use crate::model_runner::SchedulerConfig;
use crate::proto::model_runner::DraftTokenCountHistogramBucket;
use crate::proto::model_runner::DraftTokenStats;
use crate::proto::model_runner::GenerationTiming;
use crate::proto::model_runner::InferenceBackendPrefixCacheTelemetry;
use crate::proto::model_runner::PrefixCacheStats;
use crate::proto::model_runner::PrefixCacheTelemetry as PrefixCacheTelemetryProto;
use crate::proto::model_runner::TextGenerationFinishReason;
use crate::proto::model_runner::TextGenerationStats as TextGenerationStatsProto;
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

use super::generation_stats::DraftGenerationStats;
use super::generation_stats::PrefixCacheModelStats;
use super::generation_stats::TextGenerationStats;
use super::kv_cache::PrefixCacheTelemetry;
use super::model_executor::ModelExecutor;
use super::request_manager::RequestExecutionResult;
use super::request_manager::RequestManager;
use super::request_manager::RequestOutcome;
use super::request_orchestrator::CacheAffinityState;
use super::request_orchestrator::OperationalState;
use super::scheduler::Scheduler;
use super::scheduler::SchedulingDecision;
use super::text_generation;
use super::InferenceRequest;
use super::InferenceWorkerMessage;
use tokio::sync::watch;

/// Coordinates request storage, scheduling, model execution, and result delivery.
pub(super) struct InferenceWorker {
    backend_id: usize,
    model_executor: ModelExecutor,
    request_manager: RequestManager,
    scheduler: Scheduler,
    operational_state_sender: watch::Sender<OperationalState>,
    cache_affinity_state_sender: watch::Sender<CacheAffinityState>,
}

impl InferenceWorker {
    pub(super) fn new(
        backend_id: usize,
        model_executor: ModelExecutor,
        scheduler_config: SchedulerConfig,
        operational_state_sender: watch::Sender<OperationalState>,
        cache_affinity_state_sender: watch::Sender<CacheAffinityState>,
    ) -> Self {
        let scheduler_config = normalize_scheduler_config(&model_executor, scheduler_config);
        let kv_cache_geometry = model_executor.kv_cache_geometry();
        info!("Scheduler config: {scheduler_config}");
        Self {
            backend_id,
            model_executor,
            request_manager: RequestManager::new(),
            scheduler: Scheduler::new(scheduler_config, kv_cache_geometry),
            operational_state_sender,
            cache_affinity_state_sender,
        }
    }

    pub(super) fn run(mut self, mut message_receiver: mpsc::Receiver<InferenceWorkerMessage>) {
        self.publish_operational_state();
        self.publish_cache_affinity_state();
        loop {
            if self.scheduler.is_vacant() {
                let Some(message) = message_receiver.blocking_recv() else {
                    break;
                };
                if let Err(error) = self.handle_message(message) {
                    error!("Inference worker failed to handle a message: {error:#}");
                    continue;
                }
            }
            while let Ok(message) = message_receiver.try_recv() {
                if let Err(error) = self.handle_message(message) {
                    error!("Inference worker failed to handle a message: {error:#}");
                }
            }
            if let Err(error) = self.process_requests() {
                error!("Inference worker failed to process requests: {error:#}");
            }
        }
    }

    fn handle_message(&mut self, message: InferenceWorkerMessage) -> Result<()> {
        match message {
            InferenceWorkerMessage::Generate(request) => self.enqueue_request(request),
            InferenceWorkerMessage::GetPrefixCacheTelemetry { response_sender } => {
                let telemetry = self.model_executor.prefix_cache_telemetry();
                let _ = response_sender.send(InferenceBackendPrefixCacheTelemetry {
                    backend_id: self.backend_id as u64,
                    target: telemetry.target.map(PrefixCacheTelemetryProto::from),
                    draft: telemetry.draft.map(PrefixCacheTelemetryProto::from),
                });
                Ok(())
            }
        }
    }

    fn enqueue_request(&mut self, request: InferenceRequest) -> Result<()> {
        let scheduler = &mut self.scheduler;
        self.request_manager
            .add_request(request, |generate_text_request| {
                let max_new_token_count = usize::try_from(generate_text_request.max_new_tokens)
                    .context("max_new_tokens does not fit in usize")?;
                scheduler.enqueue_request(
                    generate_text_request.request_id,
                    &generate_text_request.input_token_ids,
                    max_new_token_count,
                )
            })?;
        self.publish_operational_state();
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
            let model_executor = &mut self.model_executor;
            self.request_manager
                .start_execution(request_id, |request| model_executor.start_request(request))?
        };
        // Cache restoration during startup may shorten prefill or move the request directly into
        // decode, so synchronize the scheduler with the execution state's actual initial phase.
        self.handle_generation_phase(request_id, started_request.generation_phase)?;
        info!(
            "Worker state (backend_id={}): request_id={} status=started input_tokens={} \
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
        let model_executor = &mut self.model_executor;
        let results = self
            .request_manager
            .advance_executions(&scheduling_decision.requests, |execution_batch| {
                model_executor.run_steps(execution_batch)
            });
        for RequestExecutionResult { request_id, result } in results {
            match result {
                Ok(phase) => {
                    if let Err(error) = self.complete_execution_step(request_id, phase) {
                        if self.request_manager.has_started_execution(request_id) {
                            self.abort_started_request(request_id, error)?;
                        } else {
                            return Err(error);
                        }
                    }
                }
                Err(error) if self.request_manager.has_started_execution(request_id) => {
                    self.abort_started_request(request_id, error)?;
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
        self.handle_generation_phase(request_id, phase)?;
        if !matches!(phase, text_generation::GenerationPhase::Finished { .. }) {
            return Ok(());
        }

        let model_executor = &mut self.model_executor;
        let request_outcome = self
            .request_manager
            .finish_request(request_id, |execution_state| {
                model_executor.finish_request(execution_state)
            })?;
        self.report_request_outcome(request_id, &request_outcome);
        if request_outcome.result.is_ok() {
            self.scheduler.complete_request(request_id)?;
        } else {
            self.scheduler.abort_request(request_id)?;
        }
        self.publish_operational_state();
        self.publish_cache_affinity_state();
        Ok(())
    }

    /// Updates scheduling state and optionally publishes a completed prefix so requests with the
    /// same input, held back to avoid duplicate prefix computation, can reuse it and begin running.
    /// This publication is an intermediate checkpoint and does not finalize the leader request.
    fn handle_generation_phase(
        &mut self,
        request_id: u64,
        phase: text_generation::GenerationPhase,
    ) -> Result<()> {
        let is_prefix_publication_required =
            self.scheduler.update_request_phase(request_id, phase)?;
        if !is_prefix_publication_required {
            return Ok(());
        }

        let committed_token_ids = self.request_manager.committed_token_ids(request_id)?;
        self.model_executor
            .publish_completed_prefix(request_id, committed_token_ids)?;
        self.publish_cache_affinity_state();
        self.scheduler.release_prefix_flight(request_id)
    }

    fn abort_started_request(&mut self, request_id: u64, error: anyhow::Error) -> Result<()> {
        let error = self.abort_model_execution(request_id, error);
        self.finalize_request_abort(request_id, error)
    }

    /// Releases model state and attaches any cleanup failure to the execution error.
    fn abort_model_execution(&mut self, request_id: u64, error: anyhow::Error) -> anyhow::Error {
        match self.model_executor.abort_request(request_id) {
            Ok(()) => error,
            Err(cleanup_error) => {
                error.context(format!("request cleanup also failed: {cleanup_error:#}"))
            }
        }
    }

    /// Removes an aborted request from worker state and reports its error to the client.
    /// Any allocated model state must already have been released.
    fn finalize_request_abort(&mut self, request_id: u64, error: anyhow::Error) -> Result<()> {
        self.scheduler.remove_active_request(request_id)?;
        let request_outcome = self.request_manager.abort_request(request_id, error)?;
        self.report_request_outcome(request_id, &request_outcome);
        self.scheduler.abort_request(request_id)?;
        self.publish_operational_state();
        self.publish_cache_affinity_state();
        Ok(())
    }

    fn publish_operational_state(&self) {
        let scheduler_load = self.scheduler.create_load_snapshot();
        self.operational_state_sender
            .send_replace(OperationalState {
                queued_request_count: scheduler_load.queued_request_count,
                active_request_count: scheduler_load.active_request_count,
                queued_remaining_prefill_token_count: scheduler_load
                    .queued_remaining_prefill_token_count,
                queued_remaining_max_output_token_count: scheduler_load
                    .queued_remaining_max_output_token_count,
                active_remaining_prefill_token_count: scheduler_load
                    .active_remaining_prefill_token_count,
                active_remaining_max_output_token_count: scheduler_load
                    .active_remaining_max_output_token_count,
            });
    }

    fn publish_cache_affinity_state(&self) {
        self.cache_affinity_state_sender
            .send_replace(CacheAffinityState::new(
                self.model_executor.create_cache_affinity_snapshot(),
            ));
    }

    fn report_request_outcome(&self, request_id: u64, request_outcome: &RequestOutcome) {
        let completed_at = Instant::now();
        let queued_at = request_outcome.context.queued_at;
        let queue_duration = request_outcome
            .metrics
            .execution_started_at
            .duration_since(queued_at);
        let ttft_as_micros_string =
            elapsed_microseconds_string(queued_at, request_outcome.metrics.first_token_at);
        match &request_outcome.result {
            Ok(stats) => {
                let client_stats = create_client_facing_generation_stats(
                    stats,
                    queued_at,
                    request_outcome.metrics.first_token_at,
                    request_outcome.metrics.last_token_at,
                    completed_at,
                );
                let draft_stats = stats.execution.draft_stats.as_ref();
                info!(
                    "Worker state (backend_id={}): request_id={} status=completed finish_reason={} \
                     input_tokens={} output_tokens={} queue_us={} prefill_us={} ttft_us={} \
                     decode_us={} completion_us={} \
                     target_cached_tokens={} target_newly_indexed_tokens={} \
                     draft_cached_tokens={} draft_newly_indexed_tokens={} draft_accepted={} \
                     draft_proposed={} draft_selected_histogram={}",
                    self.backend_id,
                    request_id,
                    stats.execution.finish_reason,
                    stats.execution.input_token_count,
                    stats.execution.output_token_count,
                    queue_duration.as_micros().separate_with_commas(),
                    stats
                        .execution
                        .prefill_duration
                        .as_micros()
                        .separate_with_commas(),
                    ttft_as_micros_string,
                    stats
                        .execution
                        .decode_duration
                        .as_micros()
                        .separate_with_commas(),
                    completed_at
                        .duration_since(queued_at)
                        .as_micros()
                        .separate_with_commas(),
                    stats.prefix_cache.target.restored_token_count,
                    stats
                        .prefix_cache
                        .target
                        .newly_indexed_token_count,
                    count_to_string(
                        stats
                            .prefix_cache
                            .draft
                            .as_ref()
                            .map(|stats| stats.restored_token_count)
                    ),
                    count_to_string(
                        stats
                            .prefix_cache
                            .draft
                            .as_ref()
                            .map(|stats| stats.newly_indexed_token_count)
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
                    "Worker state (backend_id={}): request_id={} status={} input_tokens={} \
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
    model_executor: &ModelExecutor,
    mut scheduler_config: SchedulerConfig,
) -> SchedulerConfig {
    if !model_executor.supports_multiple_active_requests()
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
    stats: &TextGenerationStats,
    queued_at: Instant,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    completed_at: Instant,
) -> TextGenerationStatsProto {
    let draft_token_stats = stats
        .execution
        .draft_stats
        .as_ref()
        .map(DraftTokenStats::from);
    let target_prefix_cache_stats = Some(PrefixCacheStats::from(&stats.prefix_cache.target));
    let draft_prefix_cache_stats = stats
        .prefix_cache
        .draft
        .as_ref()
        .map(PrefixCacheStats::from);
    let timing = Some(create_generation_timing(
        stats,
        queued_at,
        first_token_at,
        last_token_at,
        completed_at,
    ));
    TextGenerationStatsProto {
        finish_reason: TextGenerationFinishReason::from(stats.execution.finish_reason) as i32,
        input_token_count: stats.execution.input_token_count,
        output_token_count: stats.execution.output_token_count,
        timing,
        draft_token_stats,
        target_prefix_cache_stats,
        draft_prefix_cache_stats,
    }
}

fn create_generation_timing(
    stats: &TextGenerationStats,
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
        prefill_duration_microseconds: duration_to_microseconds(stats.execution.prefill_duration),
        decode_duration_microseconds: duration_to_microseconds(stats.execution.decode_duration),
        request_completion_latency_microseconds: duration_to_microseconds(
            completed_at.duration_since(queued_at),
        ),
    }
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

fn draft_histogram_to_string(stats: Option<&DraftGenerationStats>) -> String {
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

impl From<PrefixCacheTelemetry> for PrefixCacheTelemetryProto {
    fn from(telemetry: PrefixCacheTelemetry) -> Self {
        Self {
            token_capacity: telemetry.token_capacity as u64,
            current_indexed_token_count: telemetry.current_indexed_token_count as u64,
            cumulative_indexed_token_count: telemetry.cumulative_indexed_token_count as u64,
            cumulative_evicted_token_count: telemetry.cumulative_evicted_token_count as u64,
            cumulative_restored_token_count: telemetry.cumulative_restored_token_count as u64,
            cumulative_lookup_count: telemetry.cumulative_lookup_count as u64,
            cumulative_hit_count: telemetry.cumulative_hit_count as u64,
        }
    }
}

impl From<&PrefixCacheModelStats> for PrefixCacheStats {
    fn from(stats: &PrefixCacheModelStats) -> Self {
        Self {
            restored_token_count: stats.restored_token_count as u64,
            newly_indexed_token_count: stats.newly_indexed_token_count as u64,
        }
    }
}

impl From<&DraftGenerationStats> for DraftTokenStats {
    fn from(stats: &DraftGenerationStats) -> Self {
        Self {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::server::generation_stats::GenerationExecutionStats;
    use crate::model_runner::server::generation_stats::GenerationFinishReason;
    use crate::model_runner::server::generation_stats::PrefixCacheGenerationStats;
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
        let stats = TextGenerationStats {
            execution: GenerationExecutionStats {
                finish_reason: GenerationFinishReason::EndOfSequenceToken,
                input_token_count: 3,
                output_token_count: 2,
                draft_stats: None,
                prefill_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
            },
            prefix_cache: PrefixCacheGenerationStats {
                target: PrefixCacheModelStats {
                    restored_token_count: 0,
                    newly_indexed_token_count: 0,
                },
                draft: None,
            },
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
        let stats = TextGenerationStats {
            execution: GenerationExecutionStats {
                finish_reason: GenerationFinishReason::MaxNewTokensReached,
                input_token_count: 3,
                output_token_count: 2,
                draft_stats: Some(DraftGenerationStats {
                    accepted_token_count: 5,
                    proposed_token_count: 8,
                    selected_token_count_histogram: BTreeMap::from([(3, 2), (4, 1)]),
                }),
                prefill_duration: Duration::from_micros(123),
                decode_duration: Duration::from_micros(456),
            },
            prefix_cache: PrefixCacheGenerationStats {
                target: PrefixCacheModelStats {
                    restored_token_count: 16,
                    newly_indexed_token_count: 32,
                },
                draft: Some(PrefixCacheModelStats {
                    restored_token_count: 8,
                    newly_indexed_token_count: 24,
                }),
            },
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
