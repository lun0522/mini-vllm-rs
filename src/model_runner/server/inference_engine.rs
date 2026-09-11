use crate::model_runner::SchedulerConfig;
use crate::model_runner::SchedulingPolicy;
use crate::proto::model_runner::TextGenerationStats;
use crate::proto::model_runner::TokenGenerationLatency;
use anyhow::Result;
use std::time::Duration;
use std::time::Instant;
use thousands::Separable;
use tonic::Status;

use super::model_runner::ModelRunner;
use super::request_manager::FinishedRequest;
use super::request_manager::RequestManager;
use super::scheduler::Scheduler;
use super::text_generation;
use super::InferenceRequest;

/// Coordinates request storage, scheduling, model execution, and result delivery.
pub(super) struct InferenceEngine {
    model_runner: ModelRunner,
    request_manager: RequestManager,
    scheduler: Scheduler,
}

impl InferenceEngine {
    pub(super) fn new(model_runner: ModelRunner, scheduler_config: SchedulerConfig) -> Self {
        log::info!("Scheduler config: {scheduler_config}");
        if scheduler_config.scheduling_policy == SchedulingPolicy::ShortestPrefillFirst {
            log::warn!(
                "Shortest-prefill-first scheduling will take effect when continuous batching is implemented"
            );
        }
        Self {
            model_runner,
            request_manager: RequestManager::new(),
            scheduler: Scheduler::new(scheduler_config),
        }
    }

    pub(super) fn enqueue_request(&mut self, request: InferenceRequest) -> Result<()> {
        let request_id = request.generate_text.request_id;
        let input_token_count = request.generate_text.input_token_ids.len();
        self.request_manager.add_request(request)?;
        self.scheduler.enqueue(request_id, input_token_count);
        Ok(())
    }

    /// Executes every request currently admitted by the scheduler.
    pub(super) fn process_requests(&mut self) -> Result<()> {
        self.scheduler.admit_queued_requests();
        // TODO: Consult the scheduler between resumable generation steps once execution is
        // interleaved.
        while let Some(request_id) = self.scheduler.get_next_active_request() {
            self.process_request(request_id)?;
            self.scheduler.admit_queued_requests();
        }
        Ok(())
    }

    fn process_request(&mut self, request_id: u64) -> Result<()> {
        let previous_evicted_cached_token_count = self.model_runner.evicted_cached_token_count();
        let started_request = {
            let model_runner = &mut self.model_runner;
            self.request_manager.start_execution(
                request_id,
                previous_evicted_cached_token_count,
                |request| model_runner.start_request(request),
            )
        };
        let finished_request_result = match started_request {
            Ok(started_request) => {
                log::info!(
                    "Engine state: request_id={} status=started input_tokens={} ignore_eos_tokens={} queue_us={}",
                    request_id,
                    started_request.input_token_count,
                    started_request.ignore_eos_tokens,
                    started_request.queue_duration.as_micros().separate_with_commas(),
                );
                match self.run_request(request_id) {
                    Ok(finished_request) => Ok(finished_request),
                    Err(error) => match self.model_runner.abort_request(request_id) {
                        Ok(()) => Err(error),
                        Err(cleanup_error) => Err(error
                            .context(format!("request cleanup also failed: {cleanup_error:#}"))),
                    },
                }
            }
            Err(error) => Err(error),
        };
        let finished_request = match finished_request_result {
            Ok(finished_request) => finished_request,
            Err(error) => self.request_manager.abort_request(request_id, error)?,
        };
        let evicted_cached_token_count = self
            .model_runner
            .evicted_cached_token_count()
            .saturating_sub(finished_request.metrics.previous_evicted_cached_token_count);

        match &finished_request.result {
            Ok(result) => {
                let client_stats = create_client_facing_generation_stats(
                    &result.stats,
                    finished_request.context.queued_at,
                    finished_request.metrics.first_token_at,
                    finished_request.metrics.last_token_at,
                );
                let draft_stats = result.stats.draft_stats.as_ref();
                log::info!(
                    "Engine state: request_id={} status=completed input_tokens={} output_tokens={} queue_us={} \
                     prefill_us={} ttft_us={} decode_us={} target_cached_tokens={} \
                     draft_cached_tokens={} evicted_cached_tokens={} draft_accepted={} \
                     draft_proposed={}",
                    request_id,
                    finished_request.context.input_token_count,
                    client_stats.output_token_count,
                    finished_request
                        .metrics
                        .queue_duration(finished_request.context.queued_at)
                        .as_micros()
                        .separate_with_commas(),
                    result.stats.prefill_duration.as_micros().separate_with_commas(),
                    duration_to_microseconds_string(finished_request.metrics.first_token_at.map(|first_token_at| {
                        first_token_at.duration_since(finished_request.context.queued_at)
                    })),
                    result.stats.decode_duration.as_micros().separate_with_commas(),
                    result.stats.target_cached_token_count,
                    count_to_string(draft_stats.map(|stats| stats.cached_token_count)),
                    evicted_cached_token_count,
                    count_to_string(draft_stats.map(|stats| stats.accepted_token_count)),
                    count_to_string(draft_stats.map(|stats| stats.proposed_token_count)),
                );
                finished_request.context.send_stats(client_stats);
            }
            Err(error) => {
                let status = generation_error_status(error);
                log::info!(
                    "Engine state: request_id={} status={} input_tokens={} output_tokens={} queue_us={} \
                     ttft_us={} evicted_cached_tokens={}",
                    request_id,
                    if status.code() == tonic::Code::Cancelled {
                        "cancelled"
                    } else {
                        "failed"
                    },
                    finished_request.context.input_token_count,
                    finished_request.metrics.output_token_count,
                    finished_request
                        .metrics
                        .queue_duration(finished_request.context.queued_at)
                        .as_micros()
                        .separate_with_commas(),
                    duration_to_microseconds_string(finished_request.metrics.first_token_at.map(|first_token_at| {
                        first_token_at.duration_since(finished_request.context.queued_at)
                    })),
                    evicted_cached_token_count,
                );
                finished_request.context.send_error(status);
            }
        }
        Ok(())
    }

    fn run_request(&mut self, request_id: u64) -> Result<FinishedRequest> {
        loop {
            let phase = {
                let model_runner = &mut self.model_runner;
                self.request_manager
                    .advance_execution(request_id, |execution_state| {
                        model_runner.run_one_step(execution_state)
                    })?
            };
            if matches!(phase, text_generation::GenerationPhase::Finished) {
                let model_runner = &mut self.model_runner;
                return self
                    .request_manager
                    .finish_request(request_id, |execution_state| {
                        model_runner.finish_request(execution_state)
                    });
            }
        }
    }
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
) -> TextGenerationStats {
    let draft_token_acceptance_rate = match stats.draft_stats.as_ref() {
        Some(draft_stats) if draft_stats.proposed_token_count != 0 => {
            Some(draft_stats.accepted_token_count as f32 / draft_stats.proposed_token_count as f32)
        }
        _ => None,
    };
    let token_generation_latency =
        first_token_at
            .zip(last_token_at)
            .map(|(first_token_at, last_token_at)| TokenGenerationLatency {
                time_to_first_token_microseconds: duration_to_microseconds(
                    first_token_at.duration_since(queued_at),
                ),
                end_to_end_latency_microseconds: duration_to_microseconds(
                    last_token_at.duration_since(queued_at),
                ),
            });
    TextGenerationStats {
        input_token_count: stats.input_token_count,
        output_token_count: stats.output_token_count,
        token_generation_latency,
        draft_token_acceptance_rate,
    }
}

fn duration_to_microseconds_string(duration: Option<Duration>) -> String {
    duration.map_or_else(
        || "none".to_owned(),
        |duration| duration.as_micros().separate_with_commas(),
    )
}

fn duration_to_microseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or_default()
}

fn count_to_string(count: Option<usize>) -> String {
    count.map_or_else(|| "none".to_owned(), |count| count.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_generation_cancellation_with_the_cancelled_status() {
        let error = anyhow::Error::new(text_generation::GenerationCancelled);
        let status = generation_error_status(&error);

        assert_eq!(status.code(), tonic::Code::Cancelled);
        assert_eq!(status.message(), "generation request was cancelled");
    }
}
