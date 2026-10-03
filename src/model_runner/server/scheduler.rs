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

struct DecodeSchedulingResult {
    remaining_token_budget: usize,
    requests: Vec<ScheduledRequest>,
}

struct PrefillSchedulingMetadata {
    request_id: u64,
    remaining_token_count: usize,
}

/// Owns scheduling metadata for requests waiting for admission and active execution.
pub(super) struct Scheduler {
    max_batched_token_count: usize,
    max_active_request_count: usize,
    scheduling_policy: SchedulingPolicy,
    next_round_robin_prefill_request_id: Option<u64>,
    queued_requests: Vec<RequestSchedulingMetadata>,
    active_requests: Vec<RequestSchedulingMetadata>,
}

impl Scheduler {
    pub(super) fn new(config: SchedulerConfig) -> Self {
        Self {
            max_batched_token_count: config.max_batched_token_count,
            max_active_request_count: config.max_active_request_count,
            scheduling_policy: config.scheduling_policy,
            next_round_robin_prefill_request_id: None,
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
            .sort_by(|left, right| Self::compare_prefills(self.scheduling_policy, left, right));
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
            (GenerationPhase::Finished { .. }, _) => {
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

    pub(super) fn remove_active_request(&mut self, request_id: u64) -> Result<()> {
        let request_index = self
            .active_requests
            .iter()
            .position(|request| request.request_id == request_id)
            .with_context(|| format!("active request {request_id} does not exist"))?;
        self.active_requests.remove(request_index);
        Ok(())
    }

    /// Allocates the next model batch according to the configured scheduling policy.
    pub(super) fn create_scheduling_decision(&mut self) -> SchedulingDecision {
        let DecodeSchedulingResult {
            remaining_token_budget,
            mut requests,
        } = self.schedule_decodes(self.max_batched_token_count);

        let mut active_prefills = self.collect_active_prefills();
        self.order_prefills(&mut active_prefills);
        let scheduled_prefills = Self::schedule_prefills(&active_prefills, remaining_token_budget);
        self.update_round_robin_prefill_cursor(&active_prefills, &scheduled_prefills);
        requests.extend(scheduled_prefills);

        SchedulingDecision { requests }
    }

    fn schedule_decodes(&self, token_budget: usize) -> DecodeSchedulingResult {
        let mut remaining_token_budget = token_budget;
        let mut requests = Vec::new();
        for request in self
            .active_requests
            .iter()
            .filter(|request| matches!(request.phase, SchedulingPhase::Decode))
        {
            if remaining_token_budget == 0 {
                break;
            }
            requests.push(ScheduledRequest {
                request_id: request.request_id,
                token_budget: 1,
            });
            remaining_token_budget -= 1;
        }
        DecodeSchedulingResult {
            remaining_token_budget,
            requests,
        }
    }

    fn collect_active_prefills(&self) -> Vec<PrefillSchedulingMetadata> {
        self.active_requests
            .iter()
            .filter_map(|request| match request.phase {
                SchedulingPhase::Prefill {
                    remaining_token_count,
                } => Some(PrefillSchedulingMetadata {
                    request_id: request.request_id,
                    remaining_token_count,
                }),
                SchedulingPhase::Decode => None,
            })
            .collect()
    }

    fn update_round_robin_prefill_cursor(
        &mut self,
        active_prefills: &[PrefillSchedulingMetadata],
        scheduled_prefills: &[ScheduledRequest],
    ) {
        if self.scheduling_policy != SchedulingPolicy::RoundRobin {
            return;
        }
        if active_prefills.is_empty() {
            self.next_round_robin_prefill_request_id = None;
            return;
        }
        let Some(last_scheduled_prefill) = scheduled_prefills.last() else {
            return;
        };

        let position = active_prefills
            .iter()
            .position(|prefill| prefill.request_id == last_scheduled_prefill.request_id)
            .expect("scheduled prefill must exist in prefill metadata");
        self.next_round_robin_prefill_request_id =
            Some(active_prefills[(position + 1) % active_prefills.len()].request_id);
    }

    fn order_prefills(&self, prefills: &mut [PrefillSchedulingMetadata]) {
        match self.scheduling_policy {
            SchedulingPolicy::FirstComeFirstServed => {}
            SchedulingPolicy::ShortestPrefillFirst => {
                prefills.sort_by_key(|prefill| prefill.remaining_token_count);
            }
            SchedulingPolicy::RoundRobin => {
                let Some(start_position) =
                    self.next_round_robin_prefill_request_id
                        .and_then(|request_id| {
                            prefills
                                .iter()
                                .position(|prefill| prefill.request_id == request_id)
                        })
                else {
                    return;
                };
                prefills.rotate_left(start_position);
            }
        }
    }

    fn schedule_prefills(
        prefills: &[PrefillSchedulingMetadata],
        mut remaining_token_budget: usize,
    ) -> Vec<ScheduledRequest> {
        let mut requests = Vec::new();
        for prefill in prefills {
            if remaining_token_budget == 0 {
                break;
            }
            let token_budget = prefill.remaining_token_count.min(remaining_token_budget);
            requests.push(ScheduledRequest {
                request_id: prefill.request_id,
                token_budget,
            });
            remaining_token_budget -= token_budget;
        }
        requests
    }

    fn compare_prefills(
        scheduling_policy: SchedulingPolicy,
        left: &RequestSchedulingMetadata,
        right: &RequestSchedulingMetadata,
    ) -> Ordering {
        match scheduling_policy {
            SchedulingPolicy::FirstComeFirstServed | SchedulingPolicy::RoundRobin => {
                Ordering::Equal
            }
            SchedulingPolicy::ShortestPrefillFirst => {
                let remaining_token_count = |request: &RequestSchedulingMetadata| {
                    let SchedulingPhase::Prefill {
                        remaining_token_count,
                    } = request.phase
                    else {
                        unreachable!("queued requests must be in the prefill phase")
                    };
                    remaining_token_count
                };
                remaining_token_count(left).cmp(&remaining_token_count(right))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::server::text_generation::GenerationFinishReason;

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
            .update_request_state(
                1,
                GenerationPhase::Finished {
                    finish_reason: GenerationFinishReason::MaxNewTokensReached,
                },
            )
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
    fn every_policy_schedules_decode_before_prefill() {
        for scheduling_policy in [
            SchedulingPolicy::FirstComeFirstServed,
            SchedulingPolicy::ShortestPrefillFirst,
            SchedulingPolicy::RoundRobin,
        ] {
            let mut scheduler = scheduler_with_token_budget(4, 3, scheduling_policy);
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
    fn rotates_prefill_priority_across_round_robin_decisions() {
        let mut scheduler = scheduler_with_token_budget(5, 3, SchedulingPolicy::RoundRobin);
        scheduler.enqueue(1, 10);
        scheduler.enqueue(2, 10);
        scheduler.enqueue(3, 10);
        scheduler.admit_queued_requests();
        scheduler
            .update_request_state(
                1,
                GenerationPhase::Decode {
                    generated_token_count: 1,
                },
            )
            .unwrap();

        let first_decision = scheduler.create_scheduling_decision();
        let second_decision = scheduler.create_scheduling_decision();

        assert_eq!(
            first_decision,
            SchedulingDecision {
                requests: vec![
                    ScheduledRequest {
                        request_id: 1,
                        token_budget: 1,
                    },
                    ScheduledRequest {
                        request_id: 2,
                        token_budget: 4,
                    },
                ],
            }
        );
        assert_eq!(
            second_decision,
            SchedulingDecision {
                requests: vec![
                    ScheduledRequest {
                        request_id: 1,
                        token_budget: 1,
                    },
                    ScheduledRequest {
                        request_id: 3,
                        token_budget: 4,
                    },
                ],
            }
        );
    }

    #[test]
    fn round_robin_uses_budget_left_by_a_completed_prefill() {
        let mut scheduler = scheduler_with_token_budget(8, 3, SchedulingPolicy::RoundRobin);
        scheduler.enqueue(1, 10);
        scheduler.enqueue(2, 2);
        scheduler.enqueue(3, 10);
        scheduler.admit_queued_requests();
        scheduler
            .update_request_state(
                1,
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
                        request_id: 1,
                        token_budget: 1,
                    },
                    ScheduledRequest {
                        request_id: 2,
                        token_budget: 2,
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
