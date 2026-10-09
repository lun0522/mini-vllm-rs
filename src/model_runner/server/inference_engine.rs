use crate::model_runner::SchedulerConfig;
use crate::proto::model_runner::DraftTokenCountHistogramBucket;
use crate::proto::model_runner::DraftTokenStats;
use crate::proto::model_runner::GenerationTiming;
use crate::proto::model_runner::InferenceBackendPrefixCacheTelemetry;
use crate::proto::model_runner::PrefixCacheStats;
use crate::proto::model_runner::TextGenerationFinishReason;
use crate::proto::model_runner::TextGenerationStats;
use anyhow::Context;
use anyhow::Result;
use log::error;
use log::info;
use log::warn;
use std::time::Duration;
use std::time::Instant;
use thousands::Separable;
use tokio::sync::mpsc;
use tonic::Status;

use super::model_runner::ModelRunner;
use super::request_manager::RequestExecutionResult;
use super::request_manager::RequestManager;
use super::request_manager::RequestOutcome;
use super::scheduler::Scheduler;
use super::scheduler::SchedulingDecision;
use super::text_generation;
use super::InferenceEngineMessage;
use super::InferenceRequest;

/// Coordinates request storage, scheduling, model execution, and result delivery.
pub(super) struct InferenceEngine {
    backend_id: usize,
    model_runner: ModelRunner,
    request_manager: RequestManager,
    scheduler: Scheduler,
}

impl InferenceEngine {
    pub(super) fn new(
        backend_id: usize,
        model_runner: ModelRunner,
        scheduler_config: SchedulerConfig,
    ) -> Self {
        let scheduler_config = normalize_scheduler_config(&model_runner, scheduler_config);
        let kv_cache_geometry = model_runner.kv_cache_geometry();
        info!("Scheduler config: {scheduler_config}");
        Self {
            backend_id,
            model_runner,
            request_manager: RequestManager::new(),
            scheduler: Scheduler::new(scheduler_config, kv_cache_geometry),
        }
    }

    pub(super) fn run(mut self, mut message_receiver: mpsc::Receiver<InferenceEngineMessage>) {
        loop {
            if self.scheduler.is_vacant() {
                let Some(message) = message_receiver.blocking_recv() else {
                    break;
                };
                if let Err(error) = self.handle_message(message) {
                    error!("Inference engine failed to handle a message: {error:#}");
                    continue;
                }
            }
            while let Ok(message) = message_receiver.try_recv() {
                if let Err(error) = self.handle_message(message) {
                    error!("Inference engine failed to handle a message: {error:#}");
                }
            }
            if let Err(error) = self.process_requests() {
                error!("Inference engine failed to process requests: {error:#}");
            }
        }
    }

    fn handle_message(&mut self, message: InferenceEngineMessage) -> Result<()> {
        match message {
            InferenceEngineMessage::Generate(request) => self.enqueue_request(request),
            InferenceEngineMessage::GetPrefixCacheTelemetry { response_sender } => {
                let telemetry = self.model_runner.prefix_cache_telemetry();
                let _ = response_sender.send(InferenceBackendPrefixCacheTelemetry {
                    backend_id: self.backend_id as u64,
                    target: Some(telemetry.target),
                    draft: telemetry.draft,
                });
                Ok(())
            }
        }
    }

    fn enqueue_request(&mut self, request: InferenceRequest) -> Result<()> {
        let request_id = request.generate_text.request_id;
        let input_token_count = request.generate_text.input_token_ids.len();
        let max_new_token_count = usize::try_from(request.generate_text.max_new_tokens)
            .context("max_new_tokens does not fit in usize")?;
        self.request_manager.add_request(request)?;
        self.scheduler
            .enqueue(request_id, input_token_count, max_new_token_count);
        Ok(())
    }

    /// Executes one scheduling decision.
    fn process_requests(&mut self) -> Result<()> {
        for request_id in self.scheduler.admit_queued_requests() {
            if let Err(error) = self.start_request(request_id) {
                self.finalize_request_abort(request_id, error)?;
            }
        }
        let scheduling_decision = self.scheduler.create_scheduling_decision();
        self.process_scheduling_decision(scheduling_decision)?;
        Ok(())
    }

    fn start_request(&mut self, request_id: u64) -> Result<()> {
        let started_request = {
            let model_runner = &mut self.model_runner;
            self.request_manager
                .start_execution(request_id, |request| model_runner.start_request(request))?
        };
        self.scheduler
            .update_request_state(request_id, started_request.generation_phase)?;
        info!(
            "Engine state (backend_id={}): request_id={} status=started input_tokens={} \
             generation_phase={} ignore_eos_tokens={} queue_us={}",
            self.backend_id,
            request_id,
            started_request.input_token_count,
            started_request.generation_phase,
            started_request.ignore_eos_tokens,
            started_request
                .queue_duration
                .as_micros()
                .separate_with_commas(),
        );
        Ok(())
    }

    fn process_scheduling_decision(
        &mut self,
        scheduling_decision: SchedulingDecision,
    ) -> Result<()> {
        let model_runner = &mut self.model_runner;
        let results = self
            .request_manager
            .advance_executions(&scheduling_decision.requests, |execution_batch| {
                model_runner.run_steps(execution_batch)
            });
        for RequestExecutionResult { request_id, result } in results {
            match result {
                Ok(phase) => self.complete_execution_step(request_id, phase)?,
                Err(error) if self.request_manager.has_started_execution(request_id) => {
                    let error = self.abort_model_execution(request_id, error);
                    self.finalize_request_abort(request_id, error)?;
                }
                Err(_) => self.scheduler.remove_active_request(request_id)?,
            }
        }
        Ok(())
    }

    fn complete_execution_step(
        &mut self,
        request_id: u64,
        phase: text_generation::GenerationPhase,
    ) -> Result<()> {
        self.scheduler.update_request_state(request_id, phase)?;
        if !matches!(phase, text_generation::GenerationPhase::Finished { .. }) {
            return Ok(());
        }

        let model_runner = &mut self.model_runner;
        let request_outcome = self
            .request_manager
            .finish_request(request_id, |execution_state| {
                model_runner.finish_request(execution_state)
            })?;
        self.report_request_outcome(request_id, request_outcome);
        Ok(())
    }

    /// Releases model state and attaches any cleanup failure to the execution error.
    fn abort_model_execution(&mut self, request_id: u64, error: anyhow::Error) -> anyhow::Error {
        match self.model_runner.abort_request(request_id) {
            Ok(()) => error,
            Err(cleanup_error) => {
                error.context(format!("request cleanup also failed: {cleanup_error:#}"))
            }
        }
    }

    /// Removes an aborted request from engine state and reports its error to the client.
    /// Any allocated model state must already have been released.
    fn finalize_request_abort(&mut self, request_id: u64, error: anyhow::Error) -> Result<()> {
        self.scheduler.remove_active_request(request_id)?;
        let request_outcome = self.request_manager.abort_request(request_id, error)?;
        self.report_request_outcome(request_id, request_outcome);
        Ok(())
    }

    fn report_request_outcome(&self, request_id: u64, request_outcome: RequestOutcome) {
        let completed_at = Instant::now();
        let queued_at = request_outcome.context.queued_at;
        let queue_duration = request_outcome
            .metrics
            .execution_started_at
            .duration_since(queued_at);
        let ttft_as_micros_string =
            elapsed_microseconds_string(queued_at, request_outcome.metrics.first_token_at);
        match &request_outcome.result {
            Ok(result) => {
                let client_stats = create_client_facing_generation_stats(
                    &result.stats,
                    queued_at,
                    request_outcome.metrics.first_token_at,
                    request_outcome.metrics.last_token_at,
                    completed_at,
                );
                let draft_stats = result.stats.draft_stats.as_ref();
                info!(
                    "Engine state (backend_id={}): request_id={} status=completed finish_reason={} \
                     input_tokens={} output_tokens={} queue_us={} prefill_us={} ttft_us={} \
                     decode_us={} completion_us={} \
                     target_cached_tokens={} target_newly_indexed_tokens={} \
                     draft_cached_tokens={} draft_newly_indexed_tokens={} draft_accepted={} \
                     draft_proposed={} draft_selected_histogram={}",
                    self.backend_id,
                    request_id,
                    result.stats.finish_reason,
                    result.stats.input_token_count,
                    result.stats.output_token_count,
                    queue_duration.as_micros().separate_with_commas(),
                    result
                        .stats
                        .prefill_duration
                        .as_micros()
                        .separate_with_commas(),
                    ttft_as_micros_string,
                    result
                        .stats
                        .decode_duration
                        .as_micros()
                        .separate_with_commas(),
                    completed_at
                        .duration_since(queued_at)
                        .as_micros()
                        .separate_with_commas(),
                    result.stats.target_prefix_cache_stats.restored_token_count,
                    result
                        .stats
                        .target_prefix_cache_stats
                        .newly_indexed_token_count,
                    count_to_string(
                        draft_stats.map(|stats| stats.prefix_cache_stats.restored_token_count)
                    ),
                    count_to_string(
                        draft_stats.map(|stats| stats.prefix_cache_stats.newly_indexed_token_count)
                    ),
                    count_to_string(draft_stats.map(|stats| stats.accepted_token_count)),
                    count_to_string(draft_stats.map(|stats| stats.proposed_token_count)),
                    draft_histogram_to_string(draft_stats),
                );
                request_outcome.context.send_stats(client_stats);
            }
            Err(error) => {
                let status = generation_error_status(error);
                info!(
                    "Engine state (backend_id={}): request_id={} status={} input_tokens={} \
                     output_tokens={} queue_us={} ttft_us={}",
                    self.backend_id,
                    request_id,
                    status.code(),
                    request_outcome.context.input_token_count,
                    request_outcome.metrics.output_token_count,
                    queue_duration.as_micros().separate_with_commas(),
                    ttft_as_micros_string,
                );
                request_outcome.context.send_error(status);
            }
        }
    }
}

fn normalize_scheduler_config(
    model_runner: &ModelRunner,
    mut scheduler_config: SchedulerConfig,
) -> SchedulerConfig {
    if !model_runner.supports_multiple_active_requests()
        && scheduler_config.max_active_request_count > 1
    {
        warn!("Current KV cache type limits max_active_requests to 1");
        scheduler_config.max_active_request_count = 1;
    }
    scheduler_config
}

fn generation_error_status(error: &anyhow::Error) -> Status {
    if error
        .downcast_ref::<text_generation::GenerationCancelled>()
        .is_some()
    {
        Status::cancelled(error.to_string())
    } else {
        Status::internal(format!("model runner generation failed: {error:#}"))
    }
}

fn create_client_facing_generation_stats(
    stats: &text_generation::TextGenerationStats,
    queued_at: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    completed_at: Instant,
) -> TextGenerationStats {
    let draft_token_stats = create_draft_token_stats(stats.draft_stats.as_ref());
    let target_prefix_cache_stats =
        Some(create_prefix_cache_stats(&stats.target_prefix_cache_stats));
    let draft_prefix_cache_stats = stats
        .draft_stats
        .as_ref()
        .map(|draft_stats| create_prefix_cache_stats(&draft_stats.prefix_cache_stats));
    let timing = Some(create_generation_timing(
        stats,
        queued_at,
        first_token_at,
        last_token_at,
        completed_at,
    ));
    TextGenerationStats {
        finish_reason: TextGenerationFinishReason::from(stats.finish_reason) as i32,
        input_token_count: stats.input_token_count,
        output_token_count: stats.output_token_count,
        timing,
        draft_token_stats,
        target_prefix_cache_stats,
        draft_prefix_cache_stats,
    }
}

fn create_generation_timing(
    stats: &text_generation::TextGenerationStats,
    queued_at: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    completed_at: Instant,
) -> GenerationTiming {
    GenerationTiming {
        time_to_first_token_microseconds: first_token_at.map_or(0, |first_token_at| {
            duration_to_microseconds(first_token_at.duration_since(queued_at))
        }),
        time_to_last_token_microseconds: last_token_at.map_or(0, |last_token_at| {
            duration_to_microseconds(last_token_at.duration_since(queued_at))
        }),
        prefill_duration_microseconds: duration_to_microseconds(stats.prefill_duration),
        decode_duration_microseconds: duration_to_microseconds(stats.decode_duration),
        request_completion_latency_microseconds: duration_to_microseconds(
            completed_at.duration_since(queued_at),
        ),
    }
}

fn create_prefix_cache_stats(
    stats: &text_generation::PrefixCacheGenerationStats,
) -> PrefixCacheStats {
    PrefixCacheStats {
        restored_token_count: stats.restored_token_count as u64,
        newly_indexed_token_count: stats.newly_indexed_token_count as u64,
    }
}

fn create_draft_token_stats(
    stats: Option<&text_generation::DraftGenerationStats>,
) -> Option<DraftTokenStats> {
    stats.map(|stats| DraftTokenStats {
        accepted_token_count: stats.accepted_token_count as u64,
        proposed_token_count: stats.proposed_token_count as u64,
        selected_token_count_histogram: stats
            .selected_token_count_histogram
            .iter()
            .map(
                |(&draft_token_count, &usage_count)| DraftTokenCountHistogramBucket {
                    draft_token_count: draft_token_count as u64,
                    usage_count: usage_count as u64,
                },
            )
            .collect(),
    })
}

fn elapsed_microseconds_string(started_at: Instant, finished_at: Option<Instant>) -> String {
    finished_at.map_or_else(
        || "none".to_owned(),
        |finished_at| {
            finished_at
                .duration_since(started_at)
                .as_micros()
                .separate_with_commas()
        },
    )
}

fn duration_to_microseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or_default()
}

fn count_to_string(count: Option<usize>) -> String {
    count.map_or_else(|| "none".to_owned(), |count| count.to_string())
}

fn draft_histogram_to_string(stats: Option<&text_generation::DraftGenerationStats>) -> String {
    stats.map_or_else(
        || "none".to_owned(),
        |stats| {
            stats
                .selected_token_count_histogram
                .iter()
                .map(|(token_count, usage_count)| format!("{token_count}:{usage_count}"))
                .collect::<Vec<_>>()
                .join(",")
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn reports_generation_cancellation_with_the_cancelled_status() {
        let error = anyhow::Error::new(text_generation::GenerationCancelled);
        let status = generation_error_status(&error);

        assert_eq!(status.code(), tonic::Code::Cancelled);
        assert_eq!(status.message(), "generation request was cancelled");
    }

    #[test]
    fn omits_draft_stats_for_target_only_generation() {
        let stats = text_generation::TextGenerationStats {
            finish_reason: text_generation::GenerationFinishReason::EndOfSequenceToken,
            input_token_count: 3,
            output_token_count: 2,
            target_prefix_cache_stats: text_generation::PrefixCacheGenerationStats {
                restored_token_count: 0,
                newly_indexed_token_count: 0,
            },
            draft_stats: None,
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
        };
        let now = Instant::now();

        let client_stats = create_client_facing_generation_stats(&stats, now, None, None, now);

        assert_eq!(
            client_stats.finish_reason,
            TextGenerationFinishReason::EndOfSequenceToken as i32
        );
        assert!(client_stats.draft_token_stats.is_none());
        let timing = client_stats.timing.unwrap();
        assert_eq!(timing.time_to_first_token_microseconds, 0);
        assert_eq!(timing.time_to_last_token_microseconds, 0);
    }

    #[test]
    fn converts_draft_stats_with_a_sorted_histogram() {
        let stats = text_generation::TextGenerationStats {
            finish_reason: text_generation::GenerationFinishReason::MaxNewTokensReached,
            input_token_count: 3,
            output_token_count: 2,
            target_prefix_cache_stats: text_generation::PrefixCacheGenerationStats {
                restored_token_count: 16,
                newly_indexed_token_count: 32,
            },
            draft_stats: Some(text_generation::DraftGenerationStats {
                prefix_cache_stats: text_generation::PrefixCacheGenerationStats {
                    restored_token_count: 8,
                    newly_indexed_token_count: 24,
                },
                accepted_token_count: 5,
                proposed_token_count: 8,
                selected_token_count_histogram: BTreeMap::from([(3, 2), (4, 1)]),
            }),
            prefill_duration: Duration::from_micros(123),
            decode_duration: Duration::from_micros(456),
        };
        let now = Instant::now();

        let client_stats = create_client_facing_generation_stats(
            &stats,
            now,
            Some(now + Duration::from_micros(10)),
            Some(now + Duration::from_micros(20)),
            now + Duration::from_micros(30),
        );
        let draft_stats = client_stats.draft_token_stats.unwrap();

        assert_eq!(
            client_stats.finish_reason,
            TextGenerationFinishReason::MaxNewTokensReached as i32
        );
        assert_eq!(draft_stats.accepted_token_count, 5);
        assert_eq!(draft_stats.proposed_token_count, 8);
        assert_eq!(
            client_stats
                .target_prefix_cache_stats
                .as_ref()
                .unwrap()
                .restored_token_count,
            16
        );
        assert_eq!(
            client_stats
                .target_prefix_cache_stats
                .unwrap()
                .newly_indexed_token_count,
            32
        );
        assert_eq!(
            client_stats
                .draft_prefix_cache_stats
                .as_ref()
                .unwrap()
                .restored_token_count,
            8
        );
        assert_eq!(
            client_stats
                .draft_prefix_cache_stats
                .unwrap()
                .newly_indexed_token_count,
            24
        );
        let timing = client_stats.timing.unwrap();
        assert_eq!(timing.time_to_first_token_microseconds, 10);
        assert_eq!(timing.time_to_last_token_microseconds, 20);
        assert_eq!(timing.prefill_duration_microseconds, 123);
        assert_eq!(timing.decode_duration_microseconds, 456);
        assert_eq!(timing.request_completion_latency_microseconds, 30);
        assert_eq!(draft_stats.selected_token_count_histogram.len(), 2);
        assert_eq!(
            draft_stats.selected_token_count_histogram[0],
            DraftTokenCountHistogramBucket {
                draft_token_count: 3,
                usage_count: 2,
            }
        );
        assert_eq!(
            draft_stats.selected_token_count_histogram[1],
            DraftTokenCountHistogramBucket {
                draft_token_count: 4,
                usage_count: 1,
            }
        );
    }
}
