use crate::model_runner::SchedulerConfig;
use crate::model_runner::SchedulingPolicy;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use std::cmp::Ordering;

use super::text_generation::GenerationPhase;

struct RequestSchedulingMetadata {
    request_id: u64,
    phase: SchedulingPhase,
}

#[derive(Clone, Copy)]
pub(super) enum SchedulingPhase {
    Prefill { remaining_token_count: usize },
    Decode,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct ScheduledRequest {
    pub(super) request_id: u64,
    /// Caps prefill work. Decode receives one token of budget, but speculative decoding may emit
    /// multiple accepted tokens from that single scheduled step.
    pub(super) token_budget: usize,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct SchedulingDecision {
    pub(super) requests: Vec<ScheduledRequest>,
}

/// Owns scheduling metadata for requests waiting for admission and active execution.
pub(super) struct Scheduler {
    max_batched_token_count: usize,
    max_active_request_count: usize,
    scheduling_policy: SchedulingPolicy,
    queued_requests: Vec<RequestSchedulingMetadata>,
    active_requests: Vec<RequestSchedulingMetadata>,
}

impl Scheduler {
    pub(super) fn new(config: SchedulerConfig) -> Self {
        Self {
            max_batched_token_count: config.max_batched_token_count,
            max_active_request_count: config.max_active_request_count,
            scheduling_policy: config.scheduling_policy,
            queued_requests: Vec::new(),
            active_requests: Vec::new(),
        }
    }

    pub(super) fn enqueue(&mut self, request_id: u64, input_token_count: usize) {
        self.queued_requests.push(RequestSchedulingMetadata {
            request_id,
            phase: SchedulingPhase::Prefill {
                remaining_token_count: input_token_count,
            },
        });
    }

    /// Admits queued requests according to policy until the active-request limit is reached.
    pub(super) fn admit_queued_requests(&mut self) -> Vec<u64> {
        let available_slot_count = self
            .max_active_request_count
            .saturating_sub(self.active_requests.len());
        if available_slot_count == 0 {
            return Vec::new();
        }

        self.queued_requests
            .sort_by(|left, right| Self::compare_requests(self.scheduling_policy, left, right));
        let admitted_request_count = available_slot_count.min(self.queued_requests.len());
        let admitted_request_ids = self.queued_requests[..admitted_request_count]
            .iter()
            .map(|request| request.request_id)
            .collect();
        self.active_requests
            .extend(self.queued_requests.drain(..admitted_request_count));
        admitted_request_ids
    }

    pub(super) fn is_vacant(&self) -> bool {
        self.queued_requests.is_empty() && self.active_requests.is_empty()
    }

    pub(super) fn update_request_state(
        &mut self,
        request_id: u64,
        generation_phase: GenerationPhase,
    ) -> Result<()> {
        let request_index = self
            .active_requests
            .iter()
            .position(|request| request.request_id == request_id)
            .with_context(|| format!("active request {request_id} does not exist"))?;
        let scheduling_phase = self.active_requests[request_index].phase;
        let next_scheduling_phase = match (generation_phase, scheduling_phase) {
            (GenerationPhase::Finished, _) => {
                self.active_requests.remove(request_index);
                return Ok(());
            }
            (
                GenerationPhase::Prefill {
                    remaining_token_count,
                },
                SchedulingPhase::Prefill { .. },
            ) => SchedulingPhase::Prefill {
                remaining_token_count,
            },
            (GenerationPhase::Decode { .. }, _) => SchedulingPhase::Decode,
            (GenerationPhase::Prefill { .. }, SchedulingPhase::Decode) => {
                bail!("request {request_id} cannot transition from decode back to prefill")
            }
        };
        self.active_requests[request_index].phase = next_scheduling_phase;
        Ok(())
    }

    /// Prioritizes decode work, then allocates the remaining budget among active prefills.
    /// Prefills retain admission order under FCFS and use their remaining lengths under
    /// shortest-prefill-first.
    pub(super) fn create_scheduling_decision(&self) -> SchedulingDecision {
        let mut remaining_token_budget = self.max_batched_token_count;
        let mut requests = Vec::new();
        let mut active_requests = self.active_requests.iter().collect::<Vec<_>>();
        active_requests
            .sort_by(|left, right| Self::compare_requests(self.scheduling_policy, left, right));

        for request in active_requests {
            if remaining_token_budget == 0 {
                break;
            }
            let token_budget = match request.phase {
                SchedulingPhase::Decode => 1,
                SchedulingPhase::Prefill {
                    remaining_token_count,
                } => remaining_token_count.min(remaining_token_budget),
            };
            requests.push(ScheduledRequest {
                request_id: request.request_id,
                token_budget,
            });
            remaining_token_budget -= token_budget;
        }

        SchedulingDecision { requests }
    }

    fn compare_requests(
        scheduling_policy: SchedulingPolicy,
        left: &RequestSchedulingMetadata,
        right: &RequestSchedulingMetadata,
    ) -> Ordering {
        match (left.phase, right.phase) {
            (SchedulingPhase::Decode, SchedulingPhase::Prefill { .. }) => Ordering::Less,
            (SchedulingPhase::Prefill { .. }, SchedulingPhase::Decode) => Ordering::Greater,
            (SchedulingPhase::Decode, SchedulingPhase::Decode) => Ordering::Equal,
            (
                SchedulingPhase::Prefill {
                    remaining_token_count: left_remaining_token_count,
                },
                SchedulingPhase::Prefill {
                    remaining_token_count: right_remaining_token_count,
                },
            ) => match scheduling_policy {
                SchedulingPolicy::FirstComeFirstServed => Ordering::Equal,
                SchedulingPolicy::ShortestPrefillFirst => {
                    left_remaining_token_count.cmp(&right_remaining_token_count)
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scheduler_with_token_budget(
        max_batched_token_count: usize,
        max_active_request_count: usize,
        scheduling_policy: SchedulingPolicy,
    ) -> Scheduler {
        Scheduler::new(SchedulerConfig {
            max_batched_token_count,
            max_active_request_count,
            scheduling_policy,
        })
    }

    fn scheduler(
        max_active_request_count: usize,
        scheduling_policy: SchedulingPolicy,
    ) -> Scheduler {
        scheduler_with_token_budget(512, max_active_request_count, scheduling_policy)
    }

    #[test]
    fn admits_requests_in_arrival_order() {
        let mut scheduler = scheduler(3, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(1, 30);
        scheduler.enqueue(2, 10);
        scheduler.enqueue(3, 20);

        let admitted_request_ids = scheduler.admit_queued_requests();

        assert_eq!(admitted_request_ids, vec![1, 2, 3]);
    }

    #[test]
    fn admits_shortest_prefills_first_and_preserves_tie_order() {
        let mut scheduler = scheduler(4, SchedulingPolicy::ShortestPrefillFirst);
        scheduler.enqueue(1, 30);
        scheduler.enqueue(2, 10);
        scheduler.enqueue(3, 20);
        scheduler.enqueue(4, 10);

        let admitted_request_ids = scheduler.admit_queued_requests();

        assert_eq!(admitted_request_ids, vec![2, 4, 3, 1]);
    }

    #[test]
    fn enforces_the_active_request_limit() {
        let mut scheduler = scheduler(2, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(1, 10);
        scheduler.enqueue(2, 20);
        scheduler.enqueue(3, 30);

        let admitted_request_ids = scheduler.admit_queued_requests();

        assert_eq!(admitted_request_ids, vec![1, 2]);
        assert_eq!(scheduler.active_requests.len(), 2);
        assert_eq!(scheduler.queued_requests.len(), 1);
        scheduler
            .update_request_state(1, GenerationPhase::Finished)
            .unwrap();

        let admitted_request_ids = scheduler.admit_queued_requests();

        assert_eq!(admitted_request_ids, vec![3]);
        assert_eq!(scheduler.active_requests.len(), 2);
        assert!(scheduler.queued_requests.is_empty());
        assert_eq!(scheduler.active_requests[0].request_id, 2);
        assert_eq!(scheduler.active_requests[1].request_id, 3);
    }

    #[test]
    fn limits_scheduled_prefill_work_to_the_token_budget() {
        let mut scheduler =
            scheduler_with_token_budget(12, 3, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(1, 8);
        scheduler.enqueue(2, 10);
        scheduler.enqueue(3, 4);

        scheduler.admit_queued_requests();
        let decision = scheduler.create_scheduling_decision();

        assert_eq!(
            decision,
            SchedulingDecision {
                requests: vec![
                    ScheduledRequest {
                        request_id: 1,
                        token_budget: 8,
                    },
                    ScheduledRequest {
                        request_id: 2,
                        token_budget: 4,
                    },
                ],
            }
        );
    }

    #[test]
    fn chunks_a_prefill_larger_than_the_token_budget() {
        let mut scheduler =
            scheduler_with_token_budget(8, 1, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(1, 20);

        scheduler.admit_queued_requests();
        let decision = scheduler.create_scheduling_decision();

        assert_eq!(
            decision,
            SchedulingDecision {
                requests: vec![ScheduledRequest {
                    request_id: 1,
                    token_budget: 8,
                }],
            }
        );
    }

    #[test]
    fn schedules_decode_before_prefill() {
        let mut scheduler =
            scheduler_with_token_budget(4, 3, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(1, 10);
        scheduler.enqueue(2, 10);
        scheduler.enqueue(3, 10);
        scheduler.admit_queued_requests();
        scheduler
            .update_request_state(
                2,
                GenerationPhase::Decode {
                    generated_token_count: 1,
                },
            )
            .unwrap();

        let decision = scheduler.create_scheduling_decision();

        assert_eq!(
            decision,
            SchedulingDecision {
                requests: vec![
                    ScheduledRequest {
                        request_id: 2,
                        token_budget: 1,
                    },
                    ScheduledRequest {
                        request_id: 1,
                        token_budget: 3,
                    },
                ],
            }
        );
    }

    #[test]
    fn applies_admission_policy_before_creating_a_schedule() {
        let mut scheduler =
            scheduler_with_token_budget(15, 2, SchedulingPolicy::ShortestPrefillFirst);
        scheduler.enqueue(1, 30);
        scheduler.enqueue(2, 10);
        scheduler.enqueue(3, 20);

        scheduler.admit_queued_requests();
        let decision = scheduler.create_scheduling_decision();

        assert_eq!(
            decision,
            SchedulingDecision {
                requests: vec![
                    ScheduledRequest {
                        request_id: 2,
                        token_budget: 10,
                    },
                    ScheduledRequest {
                        request_id: 3,
                        token_budget: 5,
                    },
                ],
            }
        );
    }

    #[test]
    fn schedules_active_prefills_by_remaining_token_count() {
        let mut scheduler =
            scheduler_with_token_budget(10, 2, SchedulingPolicy::ShortestPrefillFirst);
        scheduler.enqueue(1, 10);
        scheduler.enqueue(2, 20);
        scheduler.admit_queued_requests();
        scheduler
            .update_request_state(
                1,
                GenerationPhase::Prefill {
                    remaining_token_count: 9,
                },
            )
            .unwrap();
        scheduler
            .update_request_state(
                2,
                GenerationPhase::Prefill {
                    remaining_token_count: 2,
                },
            )
            .unwrap();

        let decision = scheduler.create_scheduling_decision();

        assert_eq!(
            decision,
            SchedulingDecision {
                requests: vec![
                    ScheduledRequest {
                        request_id: 2,
                        token_budget: 2,
                    },
                    ScheduledRequest {
                        request_id: 1,
                        token_budget: 8,
                    },
                ],
            }
        );
    }

    #[test]
    fn rejects_transitioning_from_decode_back_to_prefill() {
        let mut scheduler = scheduler(1, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue(1, 10);
        scheduler.admit_queued_requests();
        scheduler
            .update_request_state(
                1,
                GenerationPhase::Decode {
                    generated_token_count: 1,
                },
            )
            .unwrap();

        let error = scheduler
            .update_request_state(
                1,
                GenerationPhase::Prefill {
                    remaining_token_count: 5,
                },
            )
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "request 1 cannot transition from decode back to prefill"
        );
    }
}
