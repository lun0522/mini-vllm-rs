use crate::model_runner::SchedulerConfig;
use crate::proto::model_runner::generate_text_event;
use crate::proto::model_runner::GenerateTextEvent;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::TextGenerationStats;
use crate::proto::model_runner::TokenGenerationLatency;
use anyhow::Context;
use anyhow::Result;
use std::time::Duration;
use std::time::Instant;
use thousands::Separable;
use tokio::sync::mpsc;
use tonic::Status;

use super::model_runner::ModelRunner;
use super::scheduler::Scheduler;
use super::text_generation;

pub(super) struct InferenceRequest {
    pub(super) queued_at: Instant,
    pub(super) generate_text: GenerateTextRequest,
    pub(super) event_sender: mpsc::Sender<Result<GenerateTextEvent, Status>>,
}

pub(super) fn run(
    mut model_runner: ModelRunner,
    mut inference_receiver: mpsc::Receiver<InferenceRequest>,
    scheduler_config: SchedulerConfig,
) {
    log::info!(
        "Scheduler config: max_batched_tokens={} max_active_requests={} scheduling_policy={}",
        scheduler_config.max_batched_token_count,
        scheduler_config.max_active_request_count,
        scheduler_config.scheduling_policy,
    );
    if scheduler_config.scheduling_policy
        == crate::model_runner::SchedulingPolicy::ShortestPrefillFirst
    {
        log::warn!(
            "Shortest-prefill-first scheduling will take effect when continuous batching is implemented"
        );
    }
    let mut scheduler = Scheduler::new(scheduler_config);
    // A single thread owns one model today. This queue can later feed a batching
    // scheduler or route requests to multiple device-specific inference workers.
    while let Some(request) = inference_receiver.blocking_recv() {
        scheduler.enqueue(request);
        scheduler.admit_queued_requests();
        // TODO: Build a batch from multiple active requests once model execution supports batching.
        while let Some(request) = scheduler.get_next_active_request() {
            process_request(&mut model_runner, request);
            scheduler.admit_queued_requests();
        }
    }
}

fn process_request(model_runner: &mut ModelRunner, request: InferenceRequest) {
    let InferenceRequest {
        queued_at,
        generate_text,
        event_sender,
    } = request;
    let request_id = generate_text.request_id;
    let input_token_count = generate_text.input_token_ids.len();
    let execution_started = Instant::now();
    let queue_duration = execution_started.duration_since(queued_at);
    log::info!(
        "Worker state: request_id={} status=started input_tokens={} ignore_eos_tokens={} queue_us={}",
        request_id,
        input_token_count,
        generate_text.end_of_sequence_token_ids.is_empty(),
        queue_duration.as_micros().separate_with_commas(),
    );
    let previous_evicted_cached_token_count = model_runner.evicted_cached_token_count();
    let mut first_token_at = None;
    let mut last_token_at = None;
    let mut output_token_count = 0;
    let result = model_runner.generate_text(
        generate_text,
        |token_id| {
            send_token_event(&event_sender, token_id)?;
            let token_sent_at = Instant::now();
            if first_token_at.is_none() {
                first_token_at = Some(token_sent_at);
            }
            last_token_at = Some(token_sent_at);
            output_token_count += 1;
            Ok(())
        },
        || event_sender.is_closed(),
    );
    let evicted_cached_token_count = model_runner
        .evicted_cached_token_count()
        .saturating_sub(previous_evicted_cached_token_count);
    let time_to_first_token =
        first_token_at.map(|first_token_at| first_token_at.duration_since(queued_at));

    match result {
        Ok(result) => {
            let client_stats = create_client_facing_generation_stats(
                &result.stats,
                queued_at,
                first_token_at,
                last_token_at,
            );
            let draft_stats = result.stats.draft_stats.as_ref();
            log::info!(
                "Worker state: request_id={} status=completed input_tokens={} output_tokens={} queue_us={} \
                 prefill_us={} ttft_us={} decode_us={} target_cached_tokens={} \
                 draft_cached_tokens={} evicted_cached_tokens={} draft_accepted={} \
                 draft_proposed={}",
                request_id,
                input_token_count,
                client_stats.output_token_count,
                queue_duration.as_micros().separate_with_commas(),
                result.stats.prefill_duration.as_micros().separate_with_commas(),
                duration_to_microseconds_string(time_to_first_token),
                result.stats.decode_duration.as_micros().separate_with_commas(),
                result.stats.target_cached_token_count,
                count_to_string(draft_stats.map(|stats| stats.cached_token_count)),
                evicted_cached_token_count,
                count_to_string(draft_stats.map(|stats| stats.accepted_token_count)),
                count_to_string(draft_stats.map(|stats| stats.proposed_token_count)),
            );
            let _ = send_event(
                &event_sender,
                generate_text_event::Event::Stats(client_stats),
            );
        }
        Err(error) => {
            let status = generation_error_status(&error);
            log::info!(
                "Worker state: request_id={} status={} input_tokens={} output_tokens={} queue_us={} \
                 ttft_us={} evicted_cached_tokens={}",
                request_id,
                if status.code() == tonic::Code::Cancelled {
                    "cancelled"
                } else {
                    "failed"
                },
                input_token_count,
                output_token_count,
                queue_duration.as_micros().separate_with_commas(),
                duration_to_microseconds_string(time_to_first_token),
                evicted_cached_token_count,
            );
            let _ = event_sender.blocking_send(Err(status));
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

fn send_token_event(
    event_sender: &mpsc::Sender<Result<GenerateTextEvent, Status>>,
    token_id: u32,
) -> Result<()> {
    send_event(event_sender, generate_text_event::Event::TokenId(token_id))
        .map_err(|_| text_generation::GenerationCancelled.into())
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
    use crate::proto::model_runner::TextGenerationStats;

    fn receive_event(
        receiver: &mut mpsc::Receiver<Result<GenerateTextEvent, Status>>,
    ) -> generate_text_event::Event {
        receiver
            .blocking_recv()
            .expect("event channel closed")
            .expect("generation returned an error")
            .event
            .expect("generation event was empty")
    }

    #[test]
    fn generation_result_follows_generated_tokens_with_final_stats() {
        let stats = TextGenerationStats {
            input_token_count: 3,
            output_token_count: 2,
            ..Default::default()
        };

        let (sender, mut receiver) = mpsc::channel(2);
        send_event(&sender, generate_text_event::Event::TokenId(42)).unwrap();
        send_event(&sender, generate_text_event::Event::Stats(stats)).unwrap();
        assert!(matches!(
            receive_event(&mut receiver),
            generate_text_event::Event::TokenId(42)
        ));
        assert!(matches!(
            receive_event(&mut receiver),
            generate_text_event::Event::Stats(received) if received == stats
        ));
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn reports_generation_cancellation_with_the_cancelled_status() {
        let error = anyhow::Error::new(text_generation::GenerationCancelled);
        let status = generation_error_status(&error);

        assert_eq!(status.code(), tonic::Code::Cancelled);
        assert_eq!(status.message(), "generation request was cancelled");
    }

    #[test]
    fn treats_a_dropped_generation_stream_as_cancellation() {
        let (sender, receiver) = mpsc::channel(1);
        drop(receiver);

        let error = send_token_event(&sender, 42).expect_err("token send should fail");

        assert!(error
            .downcast_ref::<text_generation::GenerationCancelled>()
            .is_some());
    }
}
