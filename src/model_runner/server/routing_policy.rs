use crate::model_runner::BackendThroughput;
use crate::model_runner::BackendThroughputProfile;
use crate::model_runner::InferenceDevice;
use std::cmp::Reverse;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use tokio::sync::watch;

use super::request_orchestrator::BackendReservedLoad;
use super::request_orchestrator::BackendStateReceivers;
use super::request_orchestrator::CacheAffinityState;
use super::request_orchestrator::OperationalState;
use super::request_orchestrator::RequestRoutingContext;
use super::request_orchestrator::RoutingCandidateDiagnostics;
use super::request_orchestrator::RoutingDecision;
use super::BackendId;

pub(super) trait BackendRoutingPolicy: Send + Sync {
    fn route_request(
        &self,
        request: RequestRoutingContext<'_>,
        reserved_loads: &[BackendReservedLoad],
    ) -> RoutingDecision;
}

pub(super) struct RoundRobinPolicy {
    backend_count: usize,
    next_backend_index: AtomicUsize,
}

impl RoundRobinPolicy {
    pub(super) fn new(backend_count: usize) -> Self {
        Self {
            backend_count,
            next_backend_index: AtomicUsize::new(0),
        }
    }
}

impl BackendRoutingPolicy for RoundRobinPolicy {
    fn route_request(
        &self,
        _request: RequestRoutingContext<'_>,
        _reserved_loads: &[BackendReservedLoad],
    ) -> RoutingDecision {
        let backend_index =
            self.next_backend_index.fetch_add(1, Ordering::Relaxed) % self.backend_count;
        RoutingDecision {
            backend_id: BackendId::from_index(backend_index),
            estimated_cached_prefix_token_count: 0,
            candidate_diagnostics: (0..self.backend_count)
                .map(|backend_index| {
                    RoutingCandidateDiagnostics::new(BackendId::from_index(backend_index))
                })
                .collect(),
        }
    }
}

pub(super) struct LeastRequestsPolicy {
    operational_state_receivers: Vec<watch::Receiver<OperationalState>>,
    selection_history: Mutex<BackendSelectionHistory>,
}

impl LeastRequestsPolicy {
    pub(super) fn new(backend_state_receivers: Vec<BackendStateReceivers>) -> Self {
        let backend_count = backend_state_receivers.len();
        Self {
            operational_state_receivers: backend_state_receivers
                .into_iter()
                .map(
                    |BackendStateReceivers {
                         operational,
                         cache_affinity,
                         ..
                     }| {
                        drop(cache_affinity);
                        operational
                    },
                )
                .collect(),
            selection_history: Mutex::new(BackendSelectionHistory::new(backend_count)),
        }
    }
}

impl BackendRoutingPolicy for LeastRequestsPolicy {
    fn route_request(
        &self,
        _request: RequestRoutingContext<'_>,
        reserved_loads: &[BackendReservedLoad],
    ) -> RoutingDecision {
        route_by_score(
            self.operational_state_receivers.len(),
            &self.selection_history,
            |backend_index, diagnostics| {
                let state = *self.operational_state_receivers[backend_index].borrow();
                diagnostics.operational_state = Some(state);
                let state = state.with_reserved_load(reserved_loads[backend_index]);
                let score = state
                    .queued_request_count
                    .saturating_add(state.active_request_count);
                BackendScore {
                    score,
                    estimated_cached_prefix_token_count: 0,
                }
            },
        )
    }
}

pub(super) struct LeastEstimatedWorkPolicy {
    operational_state_receivers: Vec<BackendOperationalStateReceiver>,
    throughput_profiles: BackendThroughputProfiles,
    selection_history: Mutex<BackendSelectionHistory>,
}

impl LeastEstimatedWorkPolicy {
    pub(super) fn new(
        backend_state_receivers: Vec<BackendStateReceivers>,
        throughput_profiles: Vec<BackendThroughputProfile>,
    ) -> Self {
        let backend_count = backend_state_receivers.len();
        Self {
            operational_state_receivers: backend_state_receivers
                .into_iter()
                .map(
                    |BackendStateReceivers {
                         device,
                         operational,
                         cache_affinity,
                     }| {
                        drop(cache_affinity);
                        BackendOperationalStateReceiver {
                            device,
                            operational_state_receiver: operational,
                        }
                    },
                )
                .collect(),
            throughput_profiles: BackendThroughputProfiles::new(throughput_profiles),
            selection_history: Mutex::new(BackendSelectionHistory::new(backend_count)),
        }
    }
}

impl BackendRoutingPolicy for LeastEstimatedWorkPolicy {
    fn route_request(
        &self,
        request: RequestRoutingContext<'_>,
        reserved_loads: &[BackendReservedLoad],
    ) -> RoutingDecision {
        // Estimate completion time by pricing prefill and decode work at their independently
        // measured rates for the backend and request workload.
        route_by_score(
            self.operational_state_receivers.len(),
            &self.selection_history,
            |backend_index, diagnostics| {
                let backend = &self.operational_state_receivers[backend_index];
                let state = *backend.operational_state_receiver.borrow();
                diagnostics.operational_state = Some(state);
                let state = state.with_reserved_load(reserved_loads[backend_index]);
                let estimated_work = EstimatedTokenWork::new(
                    &state, request, /* cached_prefix_token_count */ 0,
                );
                let throughput = self
                    .throughput_profiles
                    .estimate_throughput(request, backend.device);
                estimated_work.populate_diagnostics(diagnostics, throughput);
                BackendScore {
                    score: estimated_work.estimate_duration(throughput),
                    estimated_cached_prefix_token_count: 0,
                }
            },
        )
    }
}

pub(super) struct MostCachedPrefixPolicy {
    cache_affinity_state_receivers: Vec<watch::Receiver<CacheAffinityState>>,
    selection_history: Mutex<BackendSelectionHistory>,
}

impl MostCachedPrefixPolicy {
    pub(super) fn new(backend_state_receivers: Vec<BackendStateReceivers>) -> Self {
        let backend_count = backend_state_receivers.len();
        Self {
            cache_affinity_state_receivers: backend_state_receivers
                .into_iter()
                .map(|backend| backend.cache_affinity)
                .collect(),
            selection_history: Mutex::new(BackendSelectionHistory::new(backend_count)),
        }
    }
}

impl BackendRoutingPolicy for MostCachedPrefixPolicy {
    fn route_request(
        &self,
        request: RequestRoutingContext<'_>,
        _reserved_loads: &[BackendReservedLoad],
    ) -> RoutingDecision {
        route_by_score(
            self.cache_affinity_state_receivers.len(),
            &self.selection_history,
            |backend_index, diagnostics| {
                let state = self.cache_affinity_state_receivers[backend_index].borrow();
                let estimated_cached_prefix_token_count =
                    state.prefix_cache.as_ref().map_or(0, |prefix_cache| {
                        prefix_cache
                            .find_longest_prefix_token_count(request.restorable_input_token_ids())
                    });
                diagnostics.cache_affinity_state_epoch = Some(state.epoch);
                diagnostics.cached_prefix_token_count = Some(estimated_cached_prefix_token_count);
                BackendScore {
                    score: Reverse(estimated_cached_prefix_token_count),
                    estimated_cached_prefix_token_count,
                }
            },
        )
    }
}

pub(super) struct CacheAwareLeastEstimatedWorkPolicy {
    backend_state_receivers: Vec<BackendStateReceiver>,
    throughput_profiles: BackendThroughputProfiles,
    selection_history: Mutex<BackendSelectionHistory>,
}

impl CacheAwareLeastEstimatedWorkPolicy {
    pub(super) fn new(
        backend_state_receivers: Vec<BackendStateReceivers>,
        throughput_profiles: Vec<BackendThroughputProfile>,
    ) -> Self {
        let backend_count = backend_state_receivers.len();
        Self {
            backend_state_receivers: backend_state_receivers
                .into_iter()
                .map(|backend| BackendStateReceiver {
                    device: backend.device,
                    operational_state_receiver: backend.operational,
                    cache_affinity_state_receiver: backend.cache_affinity,
                })
                .collect(),
            throughput_profiles: BackendThroughputProfiles::new(throughput_profiles),
            selection_history: Mutex::new(BackendSelectionHistory::new(backend_count)),
        }
    }
}

impl BackendRoutingPolicy for CacheAwareLeastEstimatedWorkPolicy {
    fn route_request(
        &self,
        request: RequestRoutingContext<'_>,
        reserved_loads: &[BackendReservedLoad],
    ) -> RoutingDecision {
        // This uses the same phase-specific completion-time estimate as least-estimated-work,
        // but excludes the reusable cached prefix from the incoming request's prefill work.
        route_by_score(
            self.backend_state_receivers.len(),
            &self.selection_history,
            |backend_index, diagnostics| {
                let backend = &self.backend_state_receivers[backend_index];
                let operational_state = *backend.operational_state_receiver.borrow();
                diagnostics.operational_state = Some(operational_state);
                let operational_state =
                    operational_state.with_reserved_load(reserved_loads[backend_index]);
                let cache_affinity_state = backend.cache_affinity_state_receiver.borrow();
                let estimated_cached_prefix_token_count = cache_affinity_state
                    .prefix_cache
                    .as_ref()
                    .map_or(0, |prefix_cache| {
                        prefix_cache
                            .find_longest_prefix_token_count(request.restorable_input_token_ids())
                    });
                let estimated_work = EstimatedTokenWork::new(
                    &operational_state,
                    request,
                    estimated_cached_prefix_token_count,
                );
                let throughput = self
                    .throughput_profiles
                    .estimate_throughput(request, backend.device);
                diagnostics.cache_affinity_state_epoch = Some(cache_affinity_state.epoch);
                diagnostics.cached_prefix_token_count = Some(estimated_cached_prefix_token_count);
                estimated_work.populate_diagnostics(diagnostics, throughput);
                BackendScore {
                    score: estimated_work.estimate_duration(throughput),
                    estimated_cached_prefix_token_count,
                }
            },
        )
    }
}

struct BackendScore<Score> {
    score: Score,
    estimated_cached_prefix_token_count: usize,
}

fn route_by_score<Score: Ord>(
    backend_count: usize,
    selection_history: &Mutex<BackendSelectionHistory>,
    scorer: impl Fn(usize, &mut RoutingCandidateDiagnostics) -> BackendScore<Score>,
) -> RoutingDecision {
    let mut selection_history = selection_history
        .lock()
        .expect("backend selection history mutex is poisoned");
    let scored_candidates: Vec<_> = (0..backend_count)
        .map(|backend_index| {
            let mut diagnostics =
                RoutingCandidateDiagnostics::new(BackendId::from_index(backend_index));
            diagnostics.last_selected_epoch =
                Some(selection_history.last_selected_epoch[backend_index]);
            let score = scorer(backend_index, &mut diagnostics);
            (score, diagnostics)
        })
        .collect();
    let backend_index = scored_candidates
        .iter()
        .enumerate()
        .min_by(
            |(left_index, (left_score, _)), (right_index, (right_score, _))| {
                left_score.score.cmp(&right_score.score).then_with(|| {
                    selection_history.last_selected_epoch[*left_index]
                        .cmp(&selection_history.last_selected_epoch[*right_index])
                })
            },
        )
        .map(|(backend_index, _)| backend_index)
        .expect("the orchestrator always has at least one backend");
    selection_history.record_selection(backend_index);
    RoutingDecision {
        backend_id: BackendId::from_index(backend_index),
        estimated_cached_prefix_token_count: scored_candidates[backend_index]
            .0
            .estimated_cached_prefix_token_count,
        candidate_diagnostics: scored_candidates
            .into_iter()
            .map(|(_, diagnostics)| diagnostics)
            .collect(),
    }
}

struct BackendOperationalStateReceiver {
    device: InferenceDevice,
    operational_state_receiver: watch::Receiver<OperationalState>,
}

struct BackendStateReceiver {
    device: InferenceDevice,
    operational_state_receiver: watch::Receiver<OperationalState>,
    cache_affinity_state_receiver: watch::Receiver<CacheAffinityState>,
}

struct BackendThroughputProfiles {
    profiles: Vec<BackendThroughputProfile>,
}

impl BackendThroughputProfiles {
    fn new(profiles: Vec<BackendThroughputProfile>) -> Self {
        Self { profiles }
    }

    /// Uses the measured workload with the smallest Manhattan distance from the request.
    /// `min_by_key` preserves configuration order when multiple profiles are equally close.
    fn estimate_throughput(
        &self,
        request: RequestRoutingContext<'_>,
        device: InferenceDevice,
    ) -> BackendThroughput {
        let input_token_count = u64::try_from(request.input_token_count).unwrap_or(u64::MAX);
        let profile = self
            .profiles
            .iter()
            .min_by_key(|profile| {
                (profile.input_token_count.abs_diff(input_token_count) as u128).saturating_add(
                    profile
                        .max_output_token_count
                        .abs_diff(request.max_new_token_count) as u128,
                )
            })
            .expect("request orchestration requires at least one throughput profile");
        match device {
            InferenceDevice::Cpu => profile.cpu,
            InferenceDevice::Gpu => profile.gpu,
            InferenceDevice::Mixed => unreachable!("an inference backend has one concrete device"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct EstimatedTokenWork {
    prefill_token_count: u128,
    decode_token_count: u128,
}

impl EstimatedTokenWork {
    fn new(
        state: &OperationalState,
        request: RequestRoutingContext<'_>,
        cached_prefix_token_count: usize,
    ) -> Self {
        let uncached_input_token_count = request
            .input_token_count
            .saturating_sub(cached_prefix_token_count);
        Self {
            prefill_token_count: (state.queued_remaining_prefill_token_count as u128)
                .saturating_add(state.active_remaining_prefill_token_count as u128)
                .saturating_add(uncached_input_token_count as u128),
            decode_token_count: (state.queued_remaining_max_output_token_count as u128)
                .saturating_add(state.active_remaining_max_output_token_count as u128)
                .saturating_add(request.max_new_token_count as u128),
        }
    }

    fn estimate_duration(self, throughput: BackendThroughput) -> u128 {
        const NANOSECONDS_PER_SECOND: u128 = 1_000_000_000;
        let prefill_duration = self
            .prefill_token_count
            .saturating_mul(NANOSECONDS_PER_SECOND)
            / throughput.prefill_tokens_per_second as u128;
        let decode_duration = self
            .decode_token_count
            .saturating_mul(NANOSECONDS_PER_SECOND)
            / throughput.decode_tokens_per_second as u128;
        prefill_duration.saturating_add(decode_duration)
    }

    fn populate_diagnostics(
        self,
        diagnostics: &mut RoutingCandidateDiagnostics,
        throughput: BackendThroughput,
    ) {
        diagnostics.estimated_prefill_token_count = Some(self.prefill_token_count);
        diagnostics.estimated_decode_token_count = Some(self.decode_token_count);
        diagnostics.prefill_tokens_per_second = Some(throughput.prefill_tokens_per_second);
        diagnostics.decode_tokens_per_second = Some(throughput.decode_tokens_per_second);
    }
}

/// Tracks logical selection times for least-recently-selected tie-breaking.
struct BackendSelectionHistory {
    selection_epoch: u64,
    last_selected_epoch: Vec<u64>,
}

impl BackendSelectionHistory {
    fn new(backend_count: usize) -> Self {
        Self {
            selection_epoch: 0,
            last_selected_epoch: vec![0; backend_count],
        }
    }

    fn record_selection(&mut self, backend_index: usize) {
        self.selection_epoch += 1;
        self.last_selected_epoch[backend_index] = self.selection_epoch;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn backend_state_receiver(operational_state: OperationalState) -> BackendStateReceivers {
        backend_state_receiver_for_device(InferenceDevice::Cpu, operational_state)
    }

    fn backend_state_receiver_for_device(
        device: InferenceDevice,
        operational_state: OperationalState,
    ) -> BackendStateReceivers {
        let (_operational_sender, operational) = watch::channel(operational_state);
        let (_cache_affinity_sender, cache_affinity) =
            watch::channel(CacheAffinityState::default());
        BackendStateReceivers {
            device,
            operational,
            cache_affinity,
        }
    }

    fn backend_state_receivers(count: usize) -> Vec<BackendStateReceivers> {
        (0..count)
            .map(|_| backend_state_receiver(OperationalState::default()))
            .collect()
    }

    fn default_throughput_profiles() -> Vec<BackendThroughputProfile> {
        vec![BackendThroughputProfile::default()]
    }

    fn throughput(
        prefill_tokens_per_second: u64,
        decode_tokens_per_second: u64,
    ) -> BackendThroughput {
        BackendThroughput {
            prefill_tokens_per_second,
            decode_tokens_per_second,
        }
    }

    fn throughput_profile(
        input_token_count: u64,
        max_output_token_count: u64,
        cpu: BackendThroughput,
        gpu: BackendThroughput,
    ) -> BackendThroughputProfile {
        BackendThroughputProfile {
            input_token_count,
            max_output_token_count,
            cpu,
            gpu,
        }
    }

    fn routing_context(estimated_token_count: u64) -> RequestRoutingContext<'static> {
        RequestRoutingContext {
            input_token_ids: &[],
            input_token_count: 0,
            max_new_token_count: estimated_token_count,
        }
    }

    #[test]
    fn load_policies_score_reserved_work() {
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2],
            input_token_count: 2,
            max_new_token_count: 3,
        };
        let reserved_loads = [
            BackendReservedLoad {
                request_count: 1,
                prefill_token_count: 10,
                cached_prefix_token_count: 6,
                max_output_token_count: 7,
            },
            BackendReservedLoad::default(),
        ];
        let least_requests = LeastRequestsPolicy::new(backend_state_receivers(2));
        assert_eq!(
            least_requests
                .route_request(request, &reserved_loads)
                .backend_id,
            BackendId::from_index(1)
        );
        let policies: [(Box<dyn BackendRoutingPolicy>, usize, u128); 2] = [
            (
                Box::new(LeastEstimatedWorkPolicy::new(
                    backend_state_receivers(2),
                    default_throughput_profiles(),
                )),
                0,
                12,
            ),
            (
                Box::new(CacheAwareLeastEstimatedWorkPolicy::new(
                    backend_state_receivers(2),
                    default_throughput_profiles(),
                )),
                6,
                6,
            ),
        ];
        for (policy, cached_prefix_token_count, expected_prefill) in policies {
            let mut reserved_loads = reserved_loads;
            reserved_loads[0].cached_prefix_token_count = cached_prefix_token_count;
            let decision = policy.route_request(request, &reserved_loads);
            assert_eq!(decision.backend_id, BackendId::from_index(1));
            let diagnostics = &decision.candidate_diagnostics[0];
            assert_eq!(
                diagnostics.estimated_prefill_token_count,
                Some(expected_prefill)
            );
            assert_eq!(diagnostics.estimated_decode_token_count, Some(10));
            assert_eq!(
                diagnostics.operational_state,
                Some(OperationalState::default())
            );
        }
    }

    #[test]
    fn routes_requests_in_round_robin_order() {
        let policy = RoundRobinPolicy::new(2);

        let backend_ids: Vec<_> = (0..5)
            .map(|_| {
                policy
                    .route_request(routing_context(1), &[BackendReservedLoad::default(); 2])
                    .backend_id
            })
            .collect();

        assert_eq!(
            backend_ids,
            vec![
                BackendId::from_index(0),
                BackendId::from_index(1),
                BackendId::from_index(0),
                BackendId::from_index(1),
                BackendId::from_index(0),
            ]
        );
    }

    #[test]
    fn routes_every_request_to_the_only_backend() {
        let policy = RoundRobinPolicy::new(1);

        for _ in 0..3 {
            let decision =
                policy.route_request(routing_context(1), &[BackendReservedLoad::default(); 2]);
            assert_eq!(decision.backend_id, BackendId::from_index(0));
            assert_eq!(decision.candidate_diagnostics.len(), 1);
            assert_eq!(
                decision.candidate_diagnostics[0].backend_id,
                BackendId::from_index(0)
            );
        }
    }

    #[test]
    fn least_requests_observes_latest_operational_state_publications() {
        let (operational_sender, first_operational) = watch::channel(OperationalState::default());
        let (_first_cache_affinity_sender, first_cache_affinity) =
            watch::channel(CacheAffinityState::default());
        let policy = LeastRequestsPolicy::new(vec![
            BackendStateReceivers {
                device: InferenceDevice::Cpu,
                operational: first_operational,
                cache_affinity: first_cache_affinity,
            },
            backend_state_receiver(OperationalState {
                active_request_count: 1,
                ..Default::default()
            }),
        ]);

        assert_eq!(
            policy
                .route_request(routing_context(1), &[BackendReservedLoad::default(); 2])
                .backend_id,
            BackendId::from_index(0)
        );

        operational_sender.send_replace(OperationalState {
            epoch: 2,
            queued_request_count: 2,
            active_request_count: 1,
            ..Default::default()
        });

        let decision =
            policy.route_request(routing_context(1), &[BackendReservedLoad::default(); 2]);
        assert_eq!(decision.backend_id, BackendId::from_index(1));
        assert_eq!(
            decision.candidate_diagnostics[0]
                .operational_state
                .unwrap()
                .epoch,
            2
        );
    }

    #[test]
    fn routes_to_the_backend_with_the_fewest_requests() {
        let policy = LeastRequestsPolicy::new(vec![
            backend_state_receiver(OperationalState {
                queued_request_count: 2,
                active_request_count: 1,
                ..Default::default()
            }),
            backend_state_receiver(OperationalState {
                queued_request_count: 0,
                active_request_count: 2,
                ..Default::default()
            }),
        ]);

        assert_eq!(
            policy
                .route_request(routing_context(1), &[BackendReservedLoad::default(); 2])
                .backend_id,
            BackendId::from_index(1)
        );
    }

    #[test]
    fn least_requests_breaks_ties_by_least_recent_selection() {
        let policy = LeastRequestsPolicy::new(backend_state_receivers(2));

        let backend_ids: Vec<_> = (0..4)
            .map(|_| {
                policy
                    .route_request(routing_context(1), &[BackendReservedLoad::default(); 2])
                    .backend_id
            })
            .collect();

        assert_eq!(
            backend_ids,
            vec![
                BackendId::from_index(0),
                BackendId::from_index(1),
                BackendId::from_index(0),
                BackendId::from_index(1),
            ]
        );
    }

    #[test]
    fn least_estimated_work_accounts_for_backend_throughput() {
        let policy = LeastEstimatedWorkPolicy::new(
            vec![
                backend_state_receiver_for_device(
                    InferenceDevice::Cpu,
                    OperationalState {
                        queued_request_count: 0,
                        active_request_count: 1,
                        active_remaining_max_output_token_count: 100,
                        ..Default::default()
                    },
                ),
                backend_state_receiver_for_device(
                    InferenceDevice::Gpu,
                    OperationalState {
                        queued_request_count: 1,
                        active_request_count: 1,
                        queued_remaining_prefill_token_count: 50,
                        active_remaining_max_output_token_count: 100,
                        ..Default::default()
                    },
                ),
            ],
            default_throughput_profiles(),
        );

        // The GPU's higher phase-specific rates outweigh its larger queued workload.
        assert_eq!(
            policy
                .route_request(routing_context(100), &[BackendReservedLoad::default(); 2])
                .backend_id,
            BackendId::from_index(1)
        );
    }

    #[test]
    fn selects_the_nearest_throughput_profile_for_the_request_workload() {
        let profiles = BackendThroughputProfiles::new(vec![
            throughput_profile(128, 32, throughput(1, 11), throughput(2, 12)),
            throughput_profile(128, 256, throughput(3, 13), throughput(4, 14)),
            throughput_profile(512, 32, throughput(5, 15), throughput(6, 16)),
        ]);
        for (input_token_count, max_new_token_count, device, expected_throughput) in [
            (100, 50, InferenceDevice::Cpu, throughput(1, 11)),
            (100, 50, InferenceDevice::Gpu, throughput(2, 12)),
            (100, 220, InferenceDevice::Cpu, throughput(3, 13)),
            (450, 50, InferenceDevice::Gpu, throughput(6, 16)),
            // Equal-distance profiles preserve configuration order.
            (128, 144, InferenceDevice::Cpu, throughput(1, 11)),
        ] {
            let request = RequestRoutingContext {
                input_token_ids: &[],
                input_token_count,
                max_new_token_count,
            };

            assert_eq!(
                profiles.estimate_throughput(request, device),
                expected_throughput,
                "input_token_count={input_token_count}, max_new_token_count={max_new_token_count}"
            );
        }
    }

    #[test]
    fn least_estimated_work_uses_the_selected_throughput_profile() {
        let policy = LeastEstimatedWorkPolicy::new(
            vec![
                backend_state_receiver_for_device(
                    InferenceDevice::Cpu,
                    OperationalState::default(),
                ),
                backend_state_receiver_for_device(
                    InferenceDevice::Gpu,
                    OperationalState::default(),
                ),
            ],
            vec![
                throughput_profile(10, 10, throughput(10, 10), throughput(1, 1)),
                throughput_profile(100, 100, throughput(1, 1), throughput(10, 10)),
            ],
        );

        assert_eq!(
            policy
                .route_request(
                    RequestRoutingContext {
                        input_token_ids: &[],
                        input_token_count: 10,
                        max_new_token_count: 10,
                    },
                    &[BackendReservedLoad::default(); 2]
                )
                .backend_id,
            BackendId::from_index(0)
        );
        assert_eq!(
            policy
                .route_request(
                    RequestRoutingContext {
                        input_token_ids: &[],
                        input_token_count: 100,
                        max_new_token_count: 100,
                    },
                    &[BackendReservedLoad::default(); 2]
                )
                .backend_id,
            BackendId::from_index(1)
        );
    }

    #[test]
    fn least_estimated_work_prices_prefill_and_decode_at_independent_rates() {
        for (input_token_count, max_new_token_count, expected_backend_index) in
            [(100, 1, 0), (1, 100, 1)]
        {
            let policy = LeastEstimatedWorkPolicy::new(
                vec![
                    backend_state_receiver_for_device(
                        InferenceDevice::Cpu,
                        OperationalState::default(),
                    ),
                    backend_state_receiver_for_device(
                        InferenceDevice::Gpu,
                        OperationalState::default(),
                    ),
                ],
                vec![throughput_profile(
                    1,
                    1,
                    throughput(100, 1),
                    throughput(1, 100),
                )],
            );

            assert_eq!(
                policy
                    .route_request(
                        RequestRoutingContext {
                            input_token_ids: &[],
                            input_token_count,
                            max_new_token_count,
                        },
                        &[BackendReservedLoad::default(); 2]
                    )
                    .backend_id,
                BackendId::from_index(expected_backend_index)
            );
        }
    }

    #[test]
    fn least_estimated_work_breaks_equal_scores_by_least_recent_selection() {
        let policy = LeastEstimatedWorkPolicy::new(
            vec![
                backend_state_receiver_for_device(
                    InferenceDevice::Cpu,
                    OperationalState::default(),
                ),
                backend_state_receiver_for_device(
                    InferenceDevice::Cpu,
                    OperationalState::default(),
                ),
            ],
            default_throughput_profiles(),
        );

        let backend_ids: Vec<_> = (0..4)
            .map(|_| {
                policy
                    .route_request(routing_context(1), &[BackendReservedLoad::default(); 2])
                    .backend_id
            })
            .collect();

        assert_eq!(
            backend_ids,
            vec![
                BackendId::from_index(0),
                BackendId::from_index(1),
                BackendId::from_index(0),
                BackendId::from_index(1),
            ]
        );
    }

    #[test]
    fn most_cached_prefix_breaks_equal_matches_by_least_recent_selection() {
        let policy = MostCachedPrefixPolicy::new(backend_state_receivers(2));

        let decisions: Vec<_> = (0..4)
            .map(|_| policy.route_request(routing_context(1), &[BackendReservedLoad::default(); 2]))
            .collect();
        let backend_ids: Vec<_> = decisions
            .iter()
            .map(|decision| decision.backend_id)
            .collect();

        assert_eq!(
            backend_ids,
            vec![
                BackendId::from_index(0),
                BackendId::from_index(1),
                BackendId::from_index(0),
                BackendId::from_index(1),
            ]
        );
        assert!(decisions.iter().all(|decision| decision
            .candidate_diagnostics
            .iter()
            .all(|candidate| candidate.cache_affinity_state_epoch == Some(0)
                && candidate.cached_prefix_token_count == Some(0))));
    }

    #[test]
    fn cache_aware_least_estimated_work_accounts_for_backend_load() {
        let policy = CacheAwareLeastEstimatedWorkPolicy::new(
            vec![
                backend_state_receiver(OperationalState {
                    active_remaining_prefill_token_count: 20,
                    ..Default::default()
                }),
                backend_state_receiver(OperationalState::default()),
            ],
            default_throughput_profiles(),
        );
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            input_token_count: 10,
            max_new_token_count: 5,
        };

        let decision = policy.route_request(request, &[BackendReservedLoad::default(); 2]);

        // Backend 0: 20 existing + 10 input + 5 output = 35 tokens.
        // Backend 1: 10 input + 5 output = 15 tokens.
        assert_eq!(decision.backend_id, BackendId::from_index(1));
    }

    #[test]
    fn cache_aware_least_estimated_work_accounts_for_backend_throughput() {
        let policy = CacheAwareLeastEstimatedWorkPolicy::new(
            vec![
                backend_state_receiver_for_device(
                    InferenceDevice::Cpu,
                    OperationalState {
                        active_remaining_prefill_token_count: 3,
                        ..Default::default()
                    },
                ),
                backend_state_receiver_for_device(
                    InferenceDevice::Gpu,
                    OperationalState::default(),
                ),
            ],
            default_throughput_profiles(),
        );
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            input_token_count: 10,
            max_new_token_count: 5,
        };

        let decision = policy.route_request(request, &[BackendReservedLoad::default(); 2]);

        // The GPU's higher phase-specific rates more than offset the CPU's small load.
        assert_eq!(decision.backend_id, BackendId::from_index(1));
    }

    #[test]
    fn cache_aware_least_estimated_work_uses_the_selected_throughput_profile() {
        let policy = CacheAwareLeastEstimatedWorkPolicy::new(
            vec![
                backend_state_receiver_for_device(
                    InferenceDevice::Cpu,
                    OperationalState::default(),
                ),
                backend_state_receiver_for_device(
                    InferenceDevice::Gpu,
                    OperationalState::default(),
                ),
            ],
            vec![
                throughput_profile(10, 10, throughput(10, 10), throughput(1, 1)),
                throughput_profile(100, 100, throughput(1, 1), throughput(10, 10)),
            ],
        );

        for (input_token_count, max_new_token_count, expected_backend_index) in
            [(10, 10, 0), (100, 100, 1)]
        {
            assert_eq!(
                policy
                    .route_request(
                        RequestRoutingContext {
                            input_token_ids: &[],
                            input_token_count,
                            max_new_token_count,
                        },
                        &[BackendReservedLoad::default(); 2]
                    )
                    .backend_id,
                BackendId::from_index(expected_backend_index)
            );
        }
    }

    #[test]
    fn cache_aware_least_estimated_work_breaks_equal_scores_by_least_recent_selection() {
        let policy = CacheAwareLeastEstimatedWorkPolicy::new(
            backend_state_receivers(2),
            default_throughput_profiles(),
        );

        let backend_ids: Vec<_> = (0..4)
            .map(|_| {
                policy
                    .route_request(routing_context(1), &[BackendReservedLoad::default(); 2])
                    .backend_id
            })
            .collect();

        assert_eq!(
            backend_ids,
            vec![
                BackendId::from_index(0),
                BackendId::from_index(1),
                BackendId::from_index(0),
                BackendId::from_index(1),
            ]
        );
    }

    #[test]
    fn cached_prefix_reduces_only_the_incoming_prefill_work() {
        let state = OperationalState {
            queued_remaining_prefill_token_count: 10,
            queued_remaining_max_output_token_count: 20,
            active_remaining_prefill_token_count: 30,
            active_remaining_max_output_token_count: 40,
            ..Default::default()
        };
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3, 4, 5, 6],
            input_token_count: 6,
            max_new_token_count: 7,
        };

        assert_eq!(
            EstimatedTokenWork::new(&state, request, 4),
            EstimatedTokenWork {
                prefill_token_count: 42,
                decode_token_count: 67,
            }
        );
    }

    #[test]
    fn cached_prefix_token_count_cannot_underflow_incoming_work() {
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2],
            input_token_count: 2,
            max_new_token_count: 3,
        };

        assert_eq!(
            EstimatedTokenWork::new(&OperationalState::default(), request, 4),
            EstimatedTokenWork {
                prefill_token_count: 0,
                decode_token_count: 3,
            }
        );
    }

    #[test]
    fn estimates_prefill_and_decode_duration_with_independent_rates() {
        for (work, throughput, expected_duration_nanoseconds) in [
            (
                EstimatedTokenWork {
                    prefill_token_count: 20,
                    decode_token_count: 0,
                },
                throughput(10, 1),
                2_000_000_000,
            ),
            (
                EstimatedTokenWork {
                    prefill_token_count: 0,
                    decode_token_count: 20,
                },
                throughput(1, 10),
                2_000_000_000,
            ),
            (
                EstimatedTokenWork {
                    prefill_token_count: 20,
                    decode_token_count: 20,
                },
                throughput(10, 5),
                6_000_000_000,
            ),
        ] {
            assert_eq!(
                work.estimate_duration(throughput),
                expected_duration_nanoseconds
            );
        }
    }
}
