use crate::proto::model_runner::generate_text_event;
use crate::proto::model_runner::GenerateTextEvent;
use crate::proto::model_runner::GenerateTextRequest;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::mpsc;
use tonic::Status;

use super::text_generation;
use super::text_generation::CompletedGeneration;
use super::text_generation::GenerationPhase;
use super::text_generation::GenerationStep;
use super::text_generation::RequestExecutionState;
use super::InferenceRequest;

pub(super) struct RequestContext {
    pub(super) queued_at: Instant,
    pub(super) input_token_count: usize,
    event_sender: mpsc::Sender<Result<GenerateTextEvent, Status>>,
}

impl RequestContext {
    pub(super) fn send_stats(&self, stats: crate::proto::model_runner::TextGenerationStats) {
        let _ = send_event(&self.event_sender, generate_text_event::Event::Stats(stats));
    }

    pub(super) fn send_error(&self, status: Status) {
        let _ = self.event_sender.blocking_send(Err(status));
    }
}

enum RequestState {
    Queued {
        context: RequestContext,
        generate_text_request: GenerateTextRequest,
    },
    Executing {
        context: RequestContext,
        execution_state: Box<RequestExecutionState>,
        metrics: RequestExecutionMetrics,
    },
    FailedToStart {
        context: RequestContext,
        metrics: RequestExecutionMetrics,
    },
    FailedToExecute {
        context: RequestContext,
        metrics: RequestExecutionMetrics,
    },
    Cancelled {
        context: RequestContext,
        metrics: RequestExecutionMetrics,
    },
}

pub(super) struct RequestExecutionMetrics {
    pub(super) execution_started_at: Instant,
    pub(super) first_token_at: Option<Instant>,
    pub(super) last_token_at: Option<Instant>,
    pub(super) output_token_count: usize,
    pub(super) previous_evicted_cached_token_count: usize,
}

impl RequestExecutionMetrics {
    pub(super) fn queue_duration(&self, queued_at: Instant) -> Duration {
        self.execution_started_at.duration_since(queued_at)
    }
}

pub(super) struct StartedRequest {
    pub(super) input_token_count: usize,
    pub(super) ignore_eos_tokens: bool,
    pub(super) queue_duration: Duration,
}

pub(super) struct FinishedRequest {
    pub(super) context: RequestContext,
    pub(super) metrics: RequestExecutionMetrics,
    pub(super) result: Result<CompletedGeneration>,
}

/// Owns request payloads, execution state, response channels, and lifecycle timing.
pub(super) struct RequestManager {
    request_states: HashMap<u64, RequestState>,
}

impl RequestManager {
    pub(super) fn new() -> Self {
        Self {
            request_states: HashMap::new(),
        }
    }

    pub(super) fn add_request(&mut self, request: InferenceRequest) -> Result<()> {
        let request_id = request.generate_text.request_id;
        if self.request_states.contains_key(&request_id) {
            bail!("inference request {request_id} already exists");
        }
        self.request_states.insert(
            request_id,
            RequestState::Queued {
                context: RequestContext {
                    queued_at: request.queued_at,
                    input_token_count: request.generate_text.input_token_ids.len(),
                    event_sender: request.event_sender,
                },
                generate_text_request: request.generate_text,
            },
        );
        Ok(())
    }

    /// Transitions a queued request into execution using the supplied model initializer.
    pub(super) fn start_execution(
        &mut self,
        request_id: u64,
        previous_evicted_cached_token_count: usize,
        create_execution_state: impl FnOnce(GenerateTextRequest) -> Result<RequestExecutionState>,
    ) -> Result<StartedRequest> {
        let request = self.remove_request(request_id)?;
        let execution_started_at = Instant::now();
        let RequestState::Queued {
            context,
            generate_text_request,
        } = request
        else {
            self.request_states.insert(request_id, request);
            bail!("inference request {request_id} is not in queued state");
        };
        let ignore_eos_tokens = generate_text_request.end_of_sequence_token_ids.is_empty();
        let metrics = RequestExecutionMetrics {
            execution_started_at,
            first_token_at: None,
            last_token_at: None,
            output_token_count: 0,
            previous_evicted_cached_token_count,
        };
        match create_execution_state(generate_text_request) {
            Ok(execution_state) => {
                let started_request = StartedRequest {
                    input_token_count: context.input_token_count,
                    ignore_eos_tokens,
                    queue_duration: execution_started_at.duration_since(context.queued_at),
                };
                self.request_states.insert(
                    request_id,
                    RequestState::Executing {
                        context,
                        execution_state: Box::new(execution_state),
                        metrics,
                    },
                );
                Ok(started_request)
            }
            Err(error) => {
                self.request_states
                    .insert(request_id, RequestState::FailedToStart { context, metrics });
                Err(error)
            }
        }
    }

    /// Runs one model step, sends its output, and records request latency information.
    pub(super) fn advance_execution(
        &mut self,
        request_id: u64,
        run_one_step: impl FnOnce(&mut RequestExecutionState) -> Result<GenerationStep>,
    ) -> Result<GenerationPhase> {
        let request = self.remove_request(request_id)?;
        let RequestState::Executing {
            context,
            mut execution_state,
            mut metrics,
        } = request
        else {
            self.request_states.insert(request_id, request);
            bail!("inference request {request_id} is not executing");
        };
        if context.event_sender.is_closed() {
            self.request_states
                .insert(request_id, RequestState::Cancelled { context, metrics });
            return Err(text_generation::GenerationCancelled.into());
        }
        let step = match run_one_step(&mut execution_state) {
            Ok(step) => step,
            Err(error) => {
                self.request_states.insert(
                    request_id,
                    RequestState::FailedToExecute { context, metrics },
                );
                return Err(error);
            }
        };
        if let Err(error) = send_output_tokens(&context, &mut metrics, step.output_token_ids) {
            self.request_states
                .insert(request_id, RequestState::Cancelled { context, metrics });
            return Err(error);
        }
        self.request_states.insert(
            request_id,
            RequestState::Executing {
                context,
                execution_state,
                metrics,
            },
        );
        Ok(step.phase)
    }

    /// Removes an executing request and applies its model-side completion operation.
    pub(super) fn finish_request(
        &mut self,
        request_id: u64,
        complete_execution: impl FnOnce(RequestExecutionState) -> Result<CompletedGeneration>,
    ) -> Result<FinishedRequest> {
        let request = self.remove_request(request_id)?;
        let RequestState::Executing {
            context,
            execution_state,
            metrics,
        } = request
        else {
            self.request_states.insert(request_id, request);
            bail!("inference request {request_id} is not executing");
        };
        Ok(FinishedRequest {
            context,
            metrics,
            result: complete_execution(*execution_state),
        })
    }

    /// Removes a failed or cancelled request after any required model-side cleanup.
    pub(super) fn abort_request(
        &mut self,
        request_id: u64,
        error: anyhow::Error,
    ) -> Result<FinishedRequest> {
        let request = self.remove_request(request_id)?;
        let (context, metrics) = match request {
            RequestState::Executing {
                context, metrics, ..
            }
            | RequestState::FailedToStart { context, metrics }
            | RequestState::FailedToExecute {
                context, metrics, ..
            }
            | RequestState::Cancelled {
                context, metrics, ..
            } => (context, metrics),
            request @ RequestState::Queued { .. } => {
                self.request_states.insert(request_id, request);
                bail!("inference request {request_id} has not started");
            }
        };
        Ok(FinishedRequest {
            context,
            metrics,
            result: Err(error),
        })
    }

    fn remove_request(&mut self, request_id: u64) -> Result<RequestState> {
        self.request_states
            .remove(&request_id)
            .with_context(|| format!("inference request {request_id} does not exist"))
    }
}

fn send_output_tokens(
    context: &RequestContext,
    metrics: &mut RequestExecutionMetrics,
    token_ids: impl IntoIterator<Item = u32>,
) -> Result<()> {
    for token_id in token_ids {
        send_event(
            &context.event_sender,
            generate_text_event::Event::TokenId(token_id),
        )
        .map_err(|_| anyhow::Error::new(text_generation::GenerationCancelled))?;
        let token_sent_at = Instant::now();
        if metrics.first_token_at.is_none() {
            metrics.first_token_at = Some(token_sent_at);
        }
        metrics.last_token_at = Some(token_sent_at);
        metrics.output_token_count += 1;
    }
    Ok(())
}

fn send_event(
    event_sender: &mpsc::Sender<Result<GenerateTextEvent, Status>>,
    event: generate_text_event::Event,
) -> Result<()> {
    event_sender
        .blocking_send(Ok(GenerateTextEvent { event: Some(event) }))
        .context("generation response stream was dropped")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::server::text_generation::PrefillStartPositions;

    fn request(
        request_id: u64,
    ) -> (
        InferenceRequest,
        mpsc::Receiver<Result<GenerateTextEvent, Status>>,
    ) {
        let (event_sender, event_receiver) = mpsc::channel(2);
        (
            InferenceRequest {
                queued_at: Instant::now(),
                generate_text: GenerateTextRequest {
                    request_id,
                    input_token_ids: vec![1],
                    max_new_tokens: 1,
                    ..Default::default()
                },
                event_sender,
            },
            event_receiver,
        )
    }

    #[test]
    fn owns_request_execution_state_through_the_request_lifecycle() -> Result<()> {
        let mut requests = RequestManager::new();
        let (request, mut event_receiver) = request(7);
        requests.add_request(request)?;
        let started_request = requests.start_execution(7, 3, |generate_text| {
            RequestExecutionState::new(
                generate_text,
                4,
                PrefillStartPositions {
                    target: 0,
                    draft: None,
                },
            )
        })?;
        assert_eq!(started_request.input_token_count, 1);
        requests.advance_execution(7, |state| {
            assert_eq!(state.request_id(), 7);
            Ok(GenerationStep {
                output_token_ids: vec![42],
                phase: GenerationPhase::Finished,
            })
        })?;
        assert!(matches!(
            event_receiver.blocking_recv(),
            Some(Ok(GenerateTextEvent {
                event: Some(generate_text_event::Event::TokenId(42))
            }))
        ));
        let request = requests.finish_request(7, |state| {
            assert_eq!(state.request_id(), 7);
            Ok(CompletedGeneration {
                cached_sequence_token_ids: vec![1, 42],
                stats: text_generation::TextGenerationStats {
                    input_token_count: 1,
                    output_token_count: 1,
                    target_cached_token_count: 0,
                    draft_stats: None,
                    prefill_duration: Duration::ZERO,
                    decode_duration: Duration::ZERO,
                },
            })
        })?;

        assert!(request.result.is_ok());
        assert_eq!(request.context.input_token_count, 1);
        assert_eq!(request.metrics.output_token_count, 1);
        assert_eq!(request.metrics.previous_evicted_cached_token_count, 3);
        Ok(())
    }

    #[test]
    fn rejects_duplicate_request_ids() -> Result<()> {
        let mut requests = RequestManager::new();
        let (original, _event_receiver) = request(7);
        requests.add_request(original)?;
        let (duplicate, _duplicate_receiver) = request(7);

        let error = requests.add_request(duplicate).unwrap_err().to_string();

        assert_eq!(error, "inference request 7 already exists");
        Ok(())
    }

    #[test]
    fn treats_a_dropped_response_stream_as_cancellation() -> Result<()> {
        let mut requests = RequestManager::new();
        let (request, event_receiver) = request(7);
        requests.add_request(request)?;
        drop(event_receiver);

        requests.start_execution(7, 0, |generate_text| {
            RequestExecutionState::new(
                generate_text,
                4,
                PrefillStartPositions {
                    target: 0,
                    draft: None,
                },
            )
        })?;
        let error = match requests.advance_execution(7, |_| {
            Ok(GenerationStep {
                output_token_ids: vec![42],
                phase: GenerationPhase::Finished,
            })
        }) {
            Ok(_) => panic!("token send should fail"),
            Err(error) => error,
        };

        assert!(error
            .downcast_ref::<text_generation::GenerationCancelled>()
            .is_some());
        let error = match requests.advance_execution(7, |_| panic!("cancelled request advanced")) {
            Ok(_) => panic!("cancelled request should not advance again"),
            Err(error) => error,
        };
        assert_eq!(error.to_string(), "inference request 7 is not executing");
        Ok(())
    }
}
