use crate::model_runner::SchedulerConfig;
use crate::model_runner::SchedulingPolicy;
use std::collections::VecDeque;

use super::inference_worker::InferenceRequest;

/// Owns requests waiting for admission and those admitted for model execution.
pub(super) struct Scheduler {
    max_active_request_count: usize,
    scheduling_policy: SchedulingPolicy,
    queued_requests: VecDeque<InferenceRequest>,
    active_requests: VecDeque<InferenceRequest>,
}

impl Scheduler {
    pub(super) fn new(config: SchedulerConfig) -> Self {
        Self {
            max_active_request_count: config.max_active_request_count,
            scheduling_policy: config.scheduling_policy,
            queued_requests: VecDeque::new(),
            active_requests: VecDeque::new(),
        }
    }

    pub(super) fn enqueue(&mut self, request: InferenceRequest) {
        self.queued_requests.push_back(request);
    }

    /// Admits queued requests according to policy until the active-request limit is reached.
    pub(super) fn admit_queued_requests(&mut self) {
        while self.active_requests.len() < self.max_active_request_count {
            let Some(queued_request_index) = self.select_queued_request_index() else {
                break;
            };
            let Some(queued_request) = self.queued_requests.remove(queued_request_index) else {
                break;
            };
            self.active_requests.push_back(queued_request);
        }
    }

    pub(super) fn get_next_active_request(&mut self) -> Option<InferenceRequest> {
        self.active_requests.pop_front()
    }

    fn select_queued_request_index(&self) -> Option<usize> {
        match self.scheduling_policy {
            SchedulingPolicy::FirstComeFirstServed => {
                if self.queued_requests.is_empty() {
                    None
                } else {
                    Some(0)
                }
            }
            SchedulingPolicy::ShortestPrefillFirst => self
                .queued_requests
                .iter()
                .enumerate()
                .min_by_key(|(_, request)| request.generate_text.input_token_ids.len())
                .map(|(index, _)| index),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::model_runner::GenerateTextEvent;
    use crate::proto::model_runner::GenerateTextRequest;
    use std::time::Instant;
    use tokio::sync::mpsc;
    use tonic::Status;

    fn scheduler(
        max_active_request_count: usize,
        scheduling_policy: SchedulingPolicy,
    ) -> Scheduler {
        Scheduler::new(SchedulerConfig {
            max_batched_token_count: 512,
            max_active_request_count,
            scheduling_policy,
        })
    }

    fn request(request_id: u64, input_token_count: usize) -> InferenceRequest {
        let (event_sender, _event_receiver) = mpsc::channel::<Result<GenerateTextEvent, Status>>(1);
        InferenceRequest {
            queued_at: Instant::now(),
            generate_text: GenerateTextRequest {
                request_id,
                input_token_ids: vec![0; input_token_count],
                ..Default::default()
            },
            event_sender,
        }
    }

    fn pop_next_request_id(scheduler: &mut Scheduler) -> Option<u64> {
        scheduler
            .get_next_active_request()
            .map(|request| request.generate_text.request_id)
    }

    #[test]
    fn admits_requests_in_arrival_order() {
        let mut scheduler = scheduler(3, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(request(1, 30));
        scheduler.enqueue(request(2, 10));
        scheduler.enqueue(request(3, 20));

        scheduler.admit_queued_requests();

        assert_eq!(pop_next_request_id(&mut scheduler), Some(1));
        assert_eq!(pop_next_request_id(&mut scheduler), Some(2));
        assert_eq!(pop_next_request_id(&mut scheduler), Some(3));
    }

    #[test]
    fn admits_shortest_prefills_first_and_preserves_tie_order() {
        let mut scheduler = scheduler(4, SchedulingPolicy::ShortestPrefillFirst);
        scheduler.enqueue(request(1, 30));
        scheduler.enqueue(request(2, 10));
        scheduler.enqueue(request(3, 20));
        scheduler.enqueue(request(4, 10));

        scheduler.admit_queued_requests();

        assert_eq!(pop_next_request_id(&mut scheduler), Some(2));
        assert_eq!(pop_next_request_id(&mut scheduler), Some(4));
        assert_eq!(pop_next_request_id(&mut scheduler), Some(3));
        assert_eq!(pop_next_request_id(&mut scheduler), Some(1));
    }

    #[test]
    fn enforces_the_active_request_limit() {
        let mut scheduler = scheduler(2, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(request(1, 10));
        scheduler.enqueue(request(2, 20));
        scheduler.enqueue(request(3, 30));

        scheduler.admit_queued_requests();

        assert_eq!(scheduler.active_requests.len(), 2);
        assert_eq!(scheduler.queued_requests.len(), 1);
        assert_eq!(pop_next_request_id(&mut scheduler), Some(1));

        scheduler.admit_queued_requests();

        assert_eq!(scheduler.active_requests.len(), 2);
        assert!(scheduler.queued_requests.is_empty());
        assert_eq!(pop_next_request_id(&mut scheduler), Some(2));
        assert_eq!(pop_next_request_id(&mut scheduler), Some(3));
    }
}
