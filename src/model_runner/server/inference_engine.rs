use crate::model_runner::SchedulerConfig;
use crate::proto::model_runner::TextGenerationStats;
use crate::proto::model_runner::TokenGenerationLatency;
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
        info!("Scheduler config: {scheduler_config}");
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
                    error!("Inference engine failed to enqueue a request: {error:#}");
                    continue;
                }
            }
            while let Ok(request) = inference_receiver.try_recv() {
                if let Err(error) = self.enqueue_request(request) {
                    error!("Inference engine failed to enqueue a request: {error:#}");
                }
            }
            if let Err(error) = self.process_requests() {
                error!("Inference engine failed to process requests: {error:#}");
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
                self.finalize_request_abort(request_id, error)?;
            }
        }
        self.process_scheduling_decision(self.scheduler.create_scheduling_decision())?;
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
        if !matches!(phase, text_generation::GenerationPhase::Finished) {
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
                );
                let draft_stats = result.stats.draft_stats.as_ref();
                info!(
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
                request_outcome.context.send_stats(client_stats);
            }
            Err(error) => {
                let status = generation_error_status(error);
                info!(
                    "Engine state: request_id={} status={} input_tokens={} output_tokens={} queue_us={} \
                     ttft_us={}",
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
