use crate::model_runner::SchedulerConfig;
use crate::model_runner::SchedulingPolicy;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use std::cmp::Ordering;

use super::super::generation::text_generation::GenerationPhase;
use super::super::kv_cache::KvCacheGeometry;
use super::prefix_cache_flight::FlightAdmission;
use super::prefix_cache_flight::PrefixCacheFlightCoordinator;
use super::prefix_cache_flight::RequestTermination;

struct RequestSchedulingMetadata {
    request_id: u64,
    max_required_page_count: usize,
    max_new_token_count: usize,
    estimated_cached_prefix_token_count: usize,
    generated_token_count: usize,
    phase: SchedulingPhase,
    is_blocked: bool,
}

#[derive(Clone, Copy)]
pub enum SchedulingPhase {
    Prefill { remaining_token_count: usize },
    Decode,
}

#[derive(Debug, Eq, PartialEq)]
pub struct ScheduledRequest {
    pub request_id: u64,
    /// Caps prefill work. Decode receives one token of budget, but speculative decoding may emit
    /// multiple accepted tokens from that single scheduled step.
    pub token_budget: usize,
}

#[derive(Debug, Eq, PartialEq)]
pub struct SchedulingDecision {
    pub requests: Vec<ScheduledRequest>,
}

/// Aggregates raw request load facts without assigning policy-specific costs to them.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SchedulerLoadSnapshot {
    pub queued_request_count: usize,
    pub active_request_count: usize,
    pub queued_remaining_prefill_token_count: usize,
    pub queued_cached_prefix_token_count: usize,
    pub queued_remaining_max_output_token_count: usize,
    pub active_remaining_prefill_token_count: usize,
    pub active_remaining_max_output_token_count: usize,
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
pub struct Scheduler {
    max_batched_token_count: usize,
    max_active_request_count: usize,
    kv_cache_geometry: KvCacheGeometry,
    scheduling_policy: SchedulingPolicy,
    next_round_robin_prefill_request_id: Option<u64>,
    prefix_cache_flight_coordinator: Option<PrefixCacheFlightCoordinator>,
    queued_requests: Vec<RequestSchedulingMetadata>,
    active_requests: Vec<RequestSchedulingMetadata>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, kv_cache_geometry: KvCacheGeometry) -> Self {
        Self {
            max_batched_token_count: config.max_batched_token_count,
            max_active_request_count: config.max_active_request_count,
            kv_cache_geometry,
            scheduling_policy: config.scheduling_policy,
            next_round_robin_prefill_request_id: None,
            prefix_cache_flight_coordinator: PrefixCacheFlightCoordinator::from_policy(
                config.prefix_cache_coordination_policy,
            ),
            queued_requests: Vec::new(),
            active_requests: Vec::new(),
        }
    }

    pub fn enqueue_request(
        &mut self,
        request_id: u64,
        input_token_ids: &[u32],
        max_new_token_count: usize,
        estimated_cached_prefix_token_count: usize,
    ) -> Result<()> {
        let flight_admission = self
            .prefix_cache_flight_coordinator
            .as_mut()
            .map(|coordinator| coordinator.add_request(request_id, input_token_ids))
            .transpose()?;
        let input_token_count = input_token_ids.len();
        // The final generated token is returned but is never appended to the cache.
        let max_cached_token_count = input_token_count
            .checked_add(max_new_token_count)
            .and_then(|token_count| token_count.checked_sub(1))
            .expect("normalized request cache demand must fit in usize");
        self.queued_requests.push(RequestSchedulingMetadata {
            request_id,
            max_required_page_count: self
                .kv_cache_geometry
                .get_required_page_count(max_cached_token_count),
            max_new_token_count,
            estimated_cached_prefix_token_count: estimated_cached_prefix_token_count
                .min(input_token_count.saturating_sub(1)),
            generated_token_count: 0,
            phase: SchedulingPhase::Prefill {
                remaining_token_count: input_token_count,
            },
            is_blocked: flight_admission == Some(FlightAdmission::Follower),
        });
        Ok(())
    }

    /// Admits queued requests in policy order while active-request slots and cache pages remain.
    pub fn admit_queued_requests(&mut self) -> Vec<u64> {
        if self.active_requests.len() >= self.max_active_request_count {
            return Vec::new();
        }

        self.queued_requests
            .sort_by(|left, right| Self::compare_prefills(self.scheduling_policy, left, right));
        let mut available_page_count = self.get_unreserved_cache_page_count();
        let mut admitted_request_ids = Vec::new();
        let mut queued_requests = std::mem::take(&mut self.queued_requests).into_iter();
        while let Some(request) = queued_requests.next() {
            if request.is_blocked {
                self.queued_requests.push(request);
                continue;
            }
            if self.active_requests.len() >= self.max_active_request_count
                || request.max_required_page_count > available_page_count
            {
                self.queued_requests.push(request);
                self.queued_requests.extend(queued_requests);
                break;
            }
            available_page_count -= request.max_required_page_count;
            admitted_request_ids.push(request.request_id);
            self.active_requests.push(request);
        }
        admitted_request_ids
    }

    pub fn complete_request(&mut self, request_id: u64) -> Result<()> {
        self.handle_request_termination(request_id, RequestTermination::Completed)
    }

    pub fn abort_request(&mut self, request_id: u64) -> Result<()> {
        self.handle_request_termination(request_id, RequestTermination::Aborted)
    }

    /// Releases requests waiting for the leader's successfully published prefix.
    pub fn release_prefix_flight(&mut self, request_id: u64) -> Result<()> {
        let Some(coordinator) = self.prefix_cache_flight_coordinator.as_mut() else {
            return Ok(());
        };
        let runnable_request_ids = coordinator.release_after_prefix_publication(request_id)?;
        self.unblock_requests(runnable_request_ids)
    }

    pub fn is_vacant(&self) -> bool {
        self.queued_requests.is_empty() && self.active_requests.is_empty()
    }

    pub fn create_load_snapshot(&self) -> SchedulerLoadSnapshot {
        let mut snapshot = SchedulerLoadSnapshot {
            queued_request_count: self.queued_requests.len(),
            active_request_count: self.active_requests.len(),
            ..Default::default()
        };
        for request in &self.queued_requests {
            snapshot.queued_remaining_prefill_token_count +=
                Self::remaining_prefill_token_count(request);
            snapshot.queued_cached_prefix_token_count +=
                request.estimated_cached_prefix_token_count;
            snapshot.queued_remaining_max_output_token_count +=
                Self::remaining_max_output_token_count(request);
        }
        for request in &self.active_requests {
            snapshot.active_remaining_prefill_token_count +=
                Self::remaining_prefill_token_count(request);
            snapshot.active_remaining_max_output_token_count +=
                Self::remaining_max_output_token_count(request);
        }
        snapshot
    }

    /// Updates scheduling metadata and returns whether this transition requires prefix
    /// publication before the request's flight can be released.
    pub fn update_request_phase(
        &mut self,
        request_id: u64,
        generation_phase: GenerationPhase,
    ) -> Result<bool> {
        let request_index = self
            .active_requests
            .iter()
            .position(|request| request.request_id == request_id)
            .with_context(|| format!("active request {request_id} does not exist"))?;
        let scheduling_phase = self.active_requests[request_index].phase;
        if let GenerationPhase::Decode {
            generated_token_count,
        } = generation_phase
        {
            self.active_requests[request_index].generated_token_count = generated_token_count;
        }
        let next_scheduling_phase = match (generation_phase, scheduling_phase) {
            (GenerationPhase::Finished { .. }, _) => {
                self.active_requests.remove(request_index);
                return Ok(false);
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
        Ok(self.requires_prefix_publication_for_phase_transition(
            request_id,
            scheduling_phase,
            next_scheduling_phase,
        ))
    }

    pub fn remove_active_request(&mut self, request_id: u64) -> Result<()> {
        let request_index = self
            .active_requests
            .iter()
            .position(|request| request.request_id == request_id)
            .with_context(|| format!("active request {request_id} does not exist"))?;
        self.active_requests.remove(request_index);
        Ok(())
    }

    /// Allocates the next model batch according to the configured scheduling policy.
    pub fn create_scheduling_decision(&mut self) -> SchedulingDecision {
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

    /// A leader that releases followers after prefill must publish its completed prefix before the
    /// scheduler makes those followers runnable.
    fn requires_prefix_publication_for_phase_transition(
        &self,
        request_id: u64,
        scheduling_phase: SchedulingPhase,
        next_scheduling_phase: SchedulingPhase,
    ) -> bool {
        matches!(scheduling_phase, SchedulingPhase::Prefill { .. })
            && matches!(next_scheduling_phase, SchedulingPhase::Decode)
            && self
                .prefix_cache_flight_coordinator
                .as_ref()
                .is_some_and(|coordinator| coordinator.requires_prefix_publication(request_id))
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

    fn get_unreserved_cache_page_count(&self) -> usize {
        let reserved_page_count = self
            .active_requests
            .iter()
            .map(|request| request.max_required_page_count)
            .sum::<usize>();
        self.kv_cache_geometry
            .page_capacity()
            .saturating_sub(reserved_page_count)
    }

    fn remaining_prefill_token_count(request: &RequestSchedulingMetadata) -> usize {
        match request.phase {
            SchedulingPhase::Prefill {
                remaining_token_count,
            } => remaining_token_count,
            SchedulingPhase::Decode => 0,
        }
    }

    fn remaining_max_output_token_count(request: &RequestSchedulingMetadata) -> usize {
        request
            .max_new_token_count
            .saturating_sub(request.generated_token_count)
    }

    fn handle_request_termination(
        &mut self,
        request_id: u64,
        request_termination: RequestTermination,
    ) -> Result<()> {
        let Some(coordinator) = self.prefix_cache_flight_coordinator.as_mut() else {
            return Ok(());
        };
        let runnable_request_ids = coordinator.remove_request(request_id, request_termination)?;
        self.unblock_requests(runnable_request_ids)
    }

    fn unblock_requests(&mut self, runnable_request_ids: Vec<u64>) -> Result<()> {
        for runnable_request_id in runnable_request_ids {
            let request = self
                .queued_requests
                .iter_mut()
                .find(|request| request.request_id == runnable_request_id)
                .with_context(|| format!("queued request {runnable_request_id} does not exist"))?;
            ensure!(
                request.is_blocked,
                "queued request {runnable_request_id} is not blocked"
            );
            request.is_blocked = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::server::generation::generation_stats::GenerationFinishReason;
    use crate::model_runner::PrefixCacheCoordinationPolicy;

    fn scheduler_with_token_budget(
        max_batched_token_count: usize,
        max_active_request_count: usize,
        scheduling_policy: SchedulingPolicy,
    ) -> Scheduler {
        Scheduler::new(
            SchedulerConfig {
                max_batched_token_count,
                max_active_request_count,
                scheduling_policy,
                prefix_cache_coordination_policy: PrefixCacheCoordinationPolicy::Independent,
            },
            KvCacheGeometry {
                token_capacity: 4096,
                page_token_count: 16,
            },
        )
    }

    fn scheduler(
        max_active_request_count: usize,
        scheduling_policy: SchedulingPolicy,
    ) -> Scheduler {
        scheduler_with_token_budget(512, max_active_request_count, scheduling_policy)
    }

    #[test]
    fn reports_remaining_queued_and_active_tokens() -> Result<()> {
        let mut scheduler = scheduler(1, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue_request(1, &[1, 2, 3, 4], 10, 0)?;

        assert_eq!(
            scheduler.create_load_snapshot(),
            SchedulerLoadSnapshot {
                queued_request_count: 1,
                queued_remaining_prefill_token_count: 4,
                queued_remaining_max_output_token_count: 10,
                ..Default::default()
            }
        );

        assert_eq!(scheduler.admit_queued_requests(), vec![1]);
        scheduler.update_request_phase(
            1,
            GenerationPhase::Prefill {
                remaining_token_count: 2,
            },
        )?;
        assert_eq!(
            scheduler.create_load_snapshot(),
            SchedulerLoadSnapshot {
                active_request_count: 1,
                active_remaining_prefill_token_count: 2,
                active_remaining_max_output_token_count: 10,
                ..Default::default()
            }
        );

        scheduler.update_request_phase(
            1,
            GenerationPhase::Decode {
                generated_token_count: 3,
            },
        )?;
        assert_eq!(
            scheduler.create_load_snapshot(),
            SchedulerLoadSnapshot {
                active_request_count: 1,
                active_remaining_max_output_token_count: 7,
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn reports_routing_cache_estimate_only_while_request_is_queued() -> Result<()> {
        let mut scheduler = scheduler(1, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue_request(1, &[1, 2, 3, 4], 10, 3)?;

        assert_eq!(
            scheduler.create_load_snapshot(),
            SchedulerLoadSnapshot {
                queued_request_count: 1,
                queued_remaining_prefill_token_count: 4,
                queued_cached_prefix_token_count: 3,
                queued_remaining_max_output_token_count: 10,
                ..Default::default()
            }
        );

        assert_eq!(scheduler.admit_queued_requests(), vec![1]);
        assert_eq!(
            scheduler.create_load_snapshot(),
            SchedulerLoadSnapshot {
                active_request_count: 1,
                active_remaining_prefill_token_count: 4,
                active_remaining_max_output_token_count: 10,
                ..Default::default()
            }
        );
        Ok(())
    }

    #[test]
    fn admits_requests_in_arrival_order() {
        let mut scheduler = scheduler(3, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue_request(1, &[0; 30], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 20], 1, 0).unwrap();

        let admitted_request_ids = scheduler.admit_queued_requests();

        assert_eq!(admitted_request_ids, vec![1, 2, 3]);
    }

    #[test]
    fn single_flight_preserves_a_released_followers_policy_position() -> Result<()> {
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_batched_token_count: 512,
                max_active_request_count: 2,
                scheduling_policy: SchedulingPolicy::FirstComeFirstServed,
                prefix_cache_coordination_policy:
                    PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilRequestCompletion,
            },
            KvCacheGeometry {
                token_capacity: 4096,
                page_token_count: 16,
            },
        );
        scheduler.enqueue_request(1, &[10, 20], 1, 0)?;
        scheduler.enqueue_request(2, &[10, 20], 1, 0)?;
        scheduler.enqueue_request(3, &[30, 40], 1, 0)?;
        scheduler.enqueue_request(4, &[50, 60], 1, 0)?;
        assert_eq!(scheduler.admit_queued_requests(), vec![1, 3]);
        scheduler.update_request_phase(
            1,
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::MaxNewTokensReached,
            },
        )?;
        scheduler.complete_request(1)?;
        scheduler.update_request_phase(
            3,
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::MaxNewTokensReached,
            },
        )?;
        scheduler.complete_request(3)?;

        assert_eq!(scheduler.admit_queued_requests(), vec![2, 4]);
        scheduler.update_request_phase(
            2,
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::MaxNewTokensReached,
            },
        )?;
        scheduler.complete_request(2)?;
        Ok(())
    }

    #[test]
    fn prefix_publication_single_flight_unblocks_followers_before_leader_completion() -> Result<()>
    {
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_batched_token_count: 512,
                max_active_request_count: 2,
                scheduling_policy: SchedulingPolicy::FirstComeFirstServed,
                prefix_cache_coordination_policy:
                    PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilPrefixPublished,
            },
            KvCacheGeometry {
                token_capacity: 4096,
                page_token_count: 16,
            },
        );
        scheduler.enqueue_request(1, &[10, 20], 1, 0)?;
        scheduler.enqueue_request(2, &[10, 20], 1, 0)?;
        assert_eq!(scheduler.admit_queued_requests(), vec![1]);

        assert_eq!(
            scheduler.update_request_phase(
                1,
                GenerationPhase::Decode {
                    generated_token_count: 1
                }
            )?,
            true
        );
        scheduler.release_prefix_flight(1)?;

        assert_eq!(scheduler.admit_queued_requests(), vec![2]);
        assert_eq!(
            scheduler.update_request_phase(
                2,
                GenerationPhase::Decode {
                    generated_token_count: 1
                }
            )?,
            false
        );
        scheduler.update_request_phase(
            1,
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::MaxNewTokensReached,
            },
        )?;
        scheduler.complete_request(1)?;
        Ok(())
    }

    #[test]
    fn admits_shortest_prefills_first_and_preserves_tie_order() {
        let mut scheduler = scheduler(4, SchedulingPolicy::ShortestPrefillFirst);
        scheduler.enqueue_request(1, &[0; 30], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 20], 1, 0).unwrap();
        scheduler.enqueue_request(4, &[0; 10], 1, 0).unwrap();

        let admitted_request_ids = scheduler.admit_queued_requests();

        assert_eq!(admitted_request_ids, vec![2, 4, 3, 1]);
    }

    #[test]
    fn enforces_the_active_request_limit() {
        let mut scheduler = scheduler(2, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue_request(1, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 20], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 30], 1, 0).unwrap();

        let admitted_request_ids = scheduler.admit_queued_requests();

        assert_eq!(admitted_request_ids, vec![1, 2]);
        assert_eq!(scheduler.active_requests.len(), 2);
        assert_eq!(scheduler.queued_requests.len(), 1);
        scheduler
            .update_request_phase(
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
    fn excludes_the_pending_final_token_from_cache_demand() {
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_batched_token_count: 512,
                max_active_request_count: 1,
                scheduling_policy: SchedulingPolicy::FirstComeFirstServed,
                prefix_cache_coordination_policy: PrefixCacheCoordinationPolicy::Independent,
            },
            KvCacheGeometry {
                token_capacity: 8,
                page_token_count: 1,
            },
        );

        scheduler.enqueue_request(1, &[0; 3], 2, 0).unwrap();

        assert_eq!(scheduler.queued_requests[0].max_required_page_count, 4);
    }

    #[test]
    fn limits_admission_to_page_rounded_cache_reservations() {
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_batched_token_count: 512,
                max_active_request_count: 3,
                scheduling_policy: SchedulingPolicy::FirstComeFirstServed,
                prefix_cache_coordination_policy: PrefixCacheCoordinationPolicy::Independent,
            },
            KvCacheGeometry {
                token_capacity: 8,
                page_token_count: 4,
            },
        );
        scheduler.enqueue_request(1, &[0; 1], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 4], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 1], 1, 0).unwrap();

        assert_eq!(scheduler.admit_queued_requests(), vec![1, 2]);

        scheduler
            .update_request_phase(
                1,
                GenerationPhase::Finished {
                    finish_reason: GenerationFinishReason::MaxNewTokensReached,
                },
            )
            .unwrap();
        assert_eq!(scheduler.admit_queued_requests(), vec![3]);

        scheduler.enqueue_request(4, &[0; 1], 1, 0).unwrap();
        assert!(scheduler.admit_queued_requests().is_empty());
        scheduler.remove_active_request(2).unwrap();
        assert_eq!(scheduler.admit_queued_requests(), vec![4]);
    }

    #[test]
    fn preserves_admission_order_when_the_next_request_does_not_fit() {
        let mut scheduler = Scheduler::new(
            SchedulerConfig {
                max_batched_token_count: 512,
                max_active_request_count: 3,
                scheduling_policy: SchedulingPolicy::FirstComeFirstServed,
                prefix_cache_coordination_policy: PrefixCacheCoordinationPolicy::Independent,
            },
            KvCacheGeometry {
                token_capacity: 8,
                page_token_count: 4,
            },
        );
        scheduler.enqueue_request(1, &[0; 4], 1, 0).unwrap();
        assert_eq!(scheduler.admit_queued_requests(), vec![1]);
        scheduler.enqueue_request(2, &[0; 5], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 1], 1, 0).unwrap();

        assert!(scheduler.admit_queued_requests().is_empty());
    }

    #[test]
    fn limits_scheduled_prefill_work_to_the_token_budget() {
        let mut scheduler =
            scheduler_with_token_budget(12, 3, SchedulingPolicy::FirstComeFirstServed);
        scheduler.enqueue_request(1, &[0; 8], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 4], 1, 0).unwrap();

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
        scheduler.enqueue_request(1, &[0; 20], 1, 0).unwrap();

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
            scheduler.enqueue_request(1, &[0; 10], 1, 0).unwrap();
            scheduler.enqueue_request(2, &[0; 10], 1, 0).unwrap();
            scheduler.enqueue_request(3, &[0; 10], 1, 0).unwrap();
            scheduler.admit_queued_requests();
            scheduler
                .update_request_phase(
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
        scheduler.enqueue_request(1, &[0; 30], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 20], 1, 0).unwrap();

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
        scheduler.enqueue_request(1, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 20], 1, 0).unwrap();
        scheduler.admit_queued_requests();
        scheduler
            .update_request_phase(
                1,
                GenerationPhase::Prefill {
                    remaining_token_count: 9,
                },
            )
            .unwrap();
        scheduler
            .update_request_phase(
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
        scheduler.enqueue_request(1, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 10], 1, 0).unwrap();
        scheduler.admit_queued_requests();
        scheduler
            .update_request_phase(
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
        scheduler.enqueue_request(1, &[0; 10], 1, 0).unwrap();
        scheduler.enqueue_request(2, &[0; 2], 1, 0).unwrap();
        scheduler.enqueue_request(3, &[0; 10], 1, 0).unwrap();
        scheduler.admit_queued_requests();
        scheduler
            .update_request_phase(
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
        scheduler.enqueue_request(1, &[0; 10], 1, 0).unwrap();
        scheduler.admit_queued_requests();
        scheduler
            .update_request_phase(
                1,
                GenerationPhase::Decode {
                    generated_token_count: 1,
                },
            )
            .unwrap();

        let error = scheduler
            .update_request_phase(
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
