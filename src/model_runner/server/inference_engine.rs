use crate::model_runner::SchedulerConfig;
use crate::proto::model_runner::TextGenerationStats;
use crate::proto::model_runner::TokenGenerationLatency;
use anyhow::Result;
use std::time::Duration;
use std::time::Instant;
use thousands::Separable;
use tokio::sync::mpsc;
use tonic::Status;

use super::model_runner::ModelRunner;
use super::request_manager::FinishedRequest;
use super::request_manager::RequestManager;
use super::scheduler::ScheduledRequest;
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
        let scheduler_config = normalize_scheduler_config(&model_runner, scheduler_config);
        log::info!("Scheduler config: {scheduler_config}");
        Self {
            model_runner,
            request_manager: RequestManager::new(),
            scheduler: Scheduler::new(scheduler_config),
        }
    }

    pub(super) fn run(mut self, mut inference_receiver: mpsc::Receiver<InferenceRequest>) {
        loop {
            if self.scheduler.is_vacant() {
                let Some(request) = inference_receiver.blocking_recv() else {
                    break;
                };
                if let Err(error) = self.enqueue_request(request) {
                    log::error!("Inference engine failed to enqueue a request: {error:#}");
                    continue;
                }
            }
            while let Ok(request) = inference_receiver.try_recv() {
                if let Err(error) = self.enqueue_request(request) {
                    log::error!("Inference engine failed to enqueue a request: {error:#}");
                }
            }
            if let Err(error) = self.process_requests() {
                log::error!("Inference engine failed to process requests: {error:#}");
            }
        }
    }

    fn enqueue_request(&mut self, request: InferenceRequest) -> Result<()> {
        let request_id = request.generate_text.request_id;
        let input_token_count = request.generate_text.input_token_ids.len();
        self.request_manager.add_request(request)?;
        self.scheduler.enqueue(request_id, input_token_count);
        Ok(())
    }

    /// Executes one scheduling decision.
    fn process_requests(&mut self) -> Result<()> {
        for request_id in self.scheduler.admit_queued_requests() {
            if let Err(error) = self.start_request(request_id) {
                self.abort_request(request_id, error)?;
            }
        }
        let decision = self.scheduler.create_scheduling_decision();
        for scheduled_request in decision.requests {
            self.process_scheduled_request(scheduled_request)?;
        }
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
        log::info!(
            "Engine state: request_id={} status=started input_tokens={} generation_phase={} \
             ignore_eos_tokens={} queue_us={}",
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

    fn process_scheduled_request(&mut self, scheduled_request: ScheduledRequest) -> Result<()> {
        let request_id = scheduled_request.request_id;
        let phase = {
            let model_runner = &mut self.model_runner;
            self.request_manager
                .advance_execution(request_id, |execution_state| {
                    model_runner.run_one_step(execution_state, scheduled_request.token_budget)
                })
        };
        let phase = match phase {
            Ok(phase) => phase,
            Err(error) => {
                let error = match self.model_runner.abort_request(request_id) {
                    Ok(()) => error,
                    Err(cleanup_error) => {
                        error.context(format!("request cleanup also failed: {cleanup_error:#}"))
                    }
                };
                self.abort_request(request_id, error)?;
                return Ok(());
            }
        };
        self.scheduler.update_request_state(request_id, phase)?;
        if matches!(phase, text_generation::GenerationPhase::Finished) {
            let model_runner = &mut self.model_runner;
            let finished_request = self
                .request_manager
                .finish_request(request_id, |execution_state| {
                    model_runner.finish_request(execution_state)
                })?;
            self.report_finished_request(request_id, finished_request);
        }
        Ok(())
    }

    fn abort_request(&mut self, request_id: u64, error: anyhow::Error) -> Result<()> {
        self.scheduler
            .update_request_state(request_id, text_generation::GenerationPhase::Finished)?;
        let finished_request = self.request_manager.abort_request(request_id, error)?;
        self.report_finished_request(request_id, finished_request);
        Ok(())
    }

    fn report_finished_request(&self, request_id: u64, finished_request: FinishedRequest) {
        let queued_at = finished_request.context.queued_at;
        let queue_duration = finished_request
            .metrics
            .execution_started_at
            .duration_since(queued_at);
        let ttft_as_micros_string =
            elapsed_microseconds_string(queued_at, finished_request.metrics.first_token_at);
        match &finished_request.result {
            Ok(result) => {
                let client_stats = create_client_facing_generation_stats(
                    &result.stats,
                    queued_at,
                    finished_request.metrics.first_token_at,
                    finished_request.metrics.last_token_at,
                );
                let draft_stats = result.stats.draft_stats.as_ref();
                log::info!(
                    "Engine state: request_id={} status=completed input_tokens={} output_tokens={} queue_us={} \
                     prefill_us={} ttft_us={} decode_us={} target_cached_tokens={} \
                     draft_cached_tokens={} draft_accepted={} draft_proposed={}",
                    request_id,
                    result.stats.input_token_count,
                    result.stats.output_token_count,
                    queue_duration.as_micros().separate_with_commas(),
                    result.stats.prefill_duration.as_micros().separate_with_commas(),
                    ttft_as_micros_string,
                    result.stats.decode_duration.as_micros().separate_with_commas(),
                    result.stats.target_cached_token_count,
                    count_to_string(draft_stats.map(|stats| stats.cached_token_count)),
                    count_to_string(draft_stats.map(|stats| stats.accepted_token_count)),
                    count_to_string(draft_stats.map(|stats| stats.proposed_token_count)),
                );
                finished_request.context.send_stats(client_stats);
            }
            Err(error) => {
                let status = generation_error_status(error);
                log::info!(
                    "Engine state: request_id={} status={} input_tokens={} output_tokens={} queue_us={} \
                     ttft_us={}",
                    request_id,
                    status.code(),
                    finished_request.context.input_token_count,
                    finished_request.metrics.output_token_count,
                    queue_duration.as_micros().separate_with_commas(),
                    ttft_as_micros_string,
                );
                finished_request.context.send_error(status);
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
        log::warn!("Current KV cache type limits max_active_requests to 1");
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
