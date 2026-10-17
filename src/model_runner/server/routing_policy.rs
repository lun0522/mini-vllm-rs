use crate::model_runner::InferenceDevice;
use std::cmp::Reverse;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use tokio::sync::watch;

use super::request_orchestrator::BackendStateReceivers;
use super::request_orchestrator::CacheAffinityState;
use super::request_orchestrator::OperationalState;
use super::request_orchestrator::RequestRoutingContext;
use super::request_orchestrator::RoutingDecision;

// Fixed throughput weights from the batched target-only Apple M2 benchmark in
// benchmarks/metal_gemv_threshold.md: CPU 9.22 and GPU 16.25 output tokens/s.
const CPU_THROUGHPUT_WEIGHT: u64 = 922;
const GPU_THROUGHPUT_WEIGHT: u64 = 1_625;

pub(super) trait BackendRoutingPolicy: Send + Sync {
    fn route_request(&self, request: RequestRoutingContext<'_>) -> RoutingDecision;
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
    fn route_request(&self, _request: RequestRoutingContext<'_>) -> RoutingDecision {
        RoutingDecision {
            backend_index: self.next_backend_index.fetch_add(1, Ordering::Relaxed)
                % self.backend_count,
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
    fn route_request(&self, _request: RequestRoutingContext<'_>) -> RoutingDecision {
        let backend_index = route_by_score(
            self.operational_state_receivers.len(),
            &self.selection_history,
            |backend_index| {
                let state = self.operational_state_receivers[backend_index].borrow();
                state
                    .queued_request_count
                    .saturating_add(state.active_request_count)
            },
        );
        RoutingDecision { backend_index }
    }
}

pub(super) struct LeastEstimatedWorkPolicy {
    weighted_operational_state_receivers: Vec<WeightedOperationalStateReceiver>,
    selection_history: Mutex<BackendSelectionHistory>,
}

impl LeastEstimatedWorkPolicy {
    pub(super) fn new(backend_state_receivers: Vec<BackendStateReceivers>) -> Self {
        let backend_count = backend_state_receivers.len();
        Self {
            weighted_operational_state_receivers: backend_state_receivers
                .into_iter()
                .map(
                    |BackendStateReceivers {
                         device,
                         operational,
                         cache_affinity,
                     }| {
                        drop(cache_affinity);
                        WeightedOperationalStateReceiver {
                            operational_state_receiver: operational,
                            throughput_weight: estimate_throughput_weight(device),
                        }
                    },
                )
                .collect(),
            selection_history: Mutex::new(BackendSelectionHistory::new(backend_count)),
        }
    }
}

impl BackendRoutingPolicy for LeastEstimatedWorkPolicy {
    fn route_request(&self, request: RequestRoutingContext<'_>) -> RoutingDecision {
        // This is a conservative upper-bound estimate that assigns equal cost to prefill and
        // decode tokens:
        //   (remaining queued and active prefill tokens
        //      + remaining maximum queued and active output tokens
        //      + the incoming request's input and maximum output tokens)
        //   / backend throughput
        let backend_index = route_by_score(
            self.weighted_operational_state_receivers.len(),
            &self.selection_history,
            |backend_index| {
                let backend = &self.weighted_operational_state_receivers[backend_index];
                let state = backend.operational_state_receiver.borrow();
                EstimatedCompletionScore {
                    estimated_token_count: estimate_remaining_token_work(
                        &state, request, /* cached_prefix_token_count */ 0,
                    ),
                    throughput_weight: backend.throughput_weight,
                }
            },
        );
        RoutingDecision { backend_index }
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
    fn route_request(&self, request: RequestRoutingContext<'_>) -> RoutingDecision {
        let backend_index = route_by_score(
            self.cache_affinity_state_receivers.len(),
            &self.selection_history,
            |backend_index| {
                let state = self.cache_affinity_state_receivers[backend_index].borrow();
                Reverse(state.prefix_cache.as_ref().map_or(0, |prefix_cache| {
                    prefix_cache
                        .find_longest_prefix_token_count(request.restorable_input_token_ids())
                }))
            },
        );
        RoutingDecision { backend_index }
    }
}

pub(super) struct CacheAwareLeastEstimatedWorkPolicy {
    weighted_backend_state_receivers: Vec<WeightedBackendStateReceiver>,
    selection_history: Mutex<BackendSelectionHistory>,
}

impl CacheAwareLeastEstimatedWorkPolicy {
    pub(super) fn new(backend_state_receivers: Vec<BackendStateReceivers>) -> Self {
        let backend_count = backend_state_receivers.len();
        Self {
            weighted_backend_state_receivers: backend_state_receivers
                .into_iter()
                .map(|backend| WeightedBackendStateReceiver {
                    throughput_weight: estimate_throughput_weight(backend.device),
                    operational_state_receiver: backend.operational,
                    cache_affinity_state_receiver: backend.cache_affinity,
                })
                .collect(),
            selection_history: Mutex::new(BackendSelectionHistory::new(backend_count)),
        }
    }
}

impl BackendRoutingPolicy for CacheAwareLeastEstimatedWorkPolicy {
    fn route_request(&self, request: RequestRoutingContext<'_>) -> RoutingDecision {
        // This uses the same conservative estimate as least-estimated-work, but excludes the
        // reusable cached prefix from the incoming request's prefill work:
        //   (remaining queued and active prefill tokens
        //      + remaining maximum queued and active output tokens
        //      + the incoming request's uncached input and maximum output tokens)
        //   / backend throughput
        let backend_index = route_by_score(
            self.weighted_backend_state_receivers.len(),
            &self.selection_history,
            |backend_index| {
                let backend = &self.weighted_backend_state_receivers[backend_index];
                let operational_state = backend.operational_state_receiver.borrow();
                let cache_affinity_state = backend.cache_affinity_state_receiver.borrow();
                let cached_prefix_token_count =
                    cache_affinity_state
                        .prefix_cache
                        .as_ref()
                        .map_or(0, |prefix_cache| {
                            prefix_cache.find_longest_prefix_token_count(
                                request.restorable_input_token_ids(),
                            )
                        });
                EstimatedCompletionScore {
                    estimated_token_count: estimate_remaining_token_work(
                        &operational_state,
                        request,
                        cached_prefix_token_count,
                    ),
                    throughput_weight: backend.throughput_weight,
                }
            },
        );
        RoutingDecision { backend_index }
    }
}

fn estimate_remaining_token_work(
    state: &OperationalState,
    request: RequestRoutingContext<'_>,
    cached_prefix_token_count: usize,
) -> u128 {
    let uncached_input_token_count = request
        .input_token_count
        .saturating_sub(cached_prefix_token_count);
    (state.queued_remaining_prefill_token_count as u128)
        .saturating_add(state.queued_remaining_max_output_token_count as u128)
        .saturating_add(state.active_remaining_prefill_token_count as u128)
        .saturating_add(state.active_remaining_max_output_token_count as u128)
        .saturating_add(uncached_input_token_count as u128)
        .saturating_add(request.max_new_token_count as u128)
}

fn route_by_score<Score: Ord>(
    backend_count: usize,
    selection_history: &Mutex<BackendSelectionHistory>,
    scorer: impl Fn(usize) -> Score,
) -> usize {
    let mut selection_history = selection_history
        .lock()
        .expect("backend selection history mutex is poisoned");
    let backend_index = (0..backend_count)
        .min_by_key(|backend_index| {
            (
                scorer(*backend_index),
                selection_history.last_selected_epoch[*backend_index],
            )
        })
        .expect("the orchestrator always has at least one backend");
    selection_history.record_selection(backend_index);
    backend_index
}

struct WeightedOperationalStateReceiver {
    operational_state_receiver: watch::Receiver<OperationalState>,
    throughput_weight: u64,
}

struct WeightedBackendStateReceiver {
    operational_state_receiver: watch::Receiver<OperationalState>,
    cache_affinity_state_receiver: watch::Receiver<CacheAffinityState>,
    throughput_weight: u64,
}

fn estimate_throughput_weight(device: InferenceDevice) -> u64 {
    match device {
        InferenceDevice::Cpu => CPU_THROUGHPUT_WEIGHT,
        InferenceDevice::Gpu => GPU_THROUGHPUT_WEIGHT,
        InferenceDevice::Mixed => unreachable!("an inference backend has one concrete device"),
    }
}

/// Orders estimated completion times represented as token-count/throughput ratios.
struct EstimatedCompletionScore {
    estimated_token_count: u128,
    throughput_weight: u64,
}

impl PartialEq for EstimatedCompletionScore {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for EstimatedCompletionScore {}

impl PartialOrd for EstimatedCompletionScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EstimatedCompletionScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare token_count / throughput without division or floating-point rounding.
        (self.estimated_token_count * other.throughput_weight as u128)
            .cmp(&(other.estimated_token_count * self.throughput_weight as u128))
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

    fn routing_context(estimated_token_count: u64) -> RequestRoutingContext<'static> {
        RequestRoutingContext {
            input_token_ids: &[],
            input_token_count: 0,
            max_new_token_count: estimated_token_count,
        }
    }

    #[test]
    fn routes_requests_in_round_robin_order() {
        let policy = RoundRobinPolicy::new(2);

        let backend_indices: Vec<_> = (0..5)
            .map(|_| policy.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn routes_every_request_to_the_only_backend() {
        let policy = RoundRobinPolicy::new(1);

        for _ in 0..3 {
            assert_eq!(policy.route_request(routing_context(1)).backend_index, 0);
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

        assert_eq!(policy.route_request(routing_context(1)).backend_index, 0);

        operational_sender.send_replace(OperationalState {
            queued_request_count: 2,
            active_request_count: 1,
            ..Default::default()
        });

        assert_eq!(policy.route_request(routing_context(1)).backend_index, 1);
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

        assert_eq!(policy.route_request(routing_context(1)).backend_index, 1);
    }

    #[test]
    fn least_requests_breaks_ties_by_least_recent_selection() {
        let policy = LeastRequestsPolicy::new(backend_state_receivers(2));

        let backend_indices: Vec<_> = (0..4)
            .map(|_| policy.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1]);
    }

    #[test]
    fn least_estimated_work_accounts_for_backend_throughput() {
        let policy = LeastEstimatedWorkPolicy::new(vec![
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
        ]);

        // CPU: 200 / 9.22; GPU: 250 / 16.25, so the busier GPU still has less work.
        assert_eq!(policy.route_request(routing_context(100)).backend_index, 1);
    }

    #[test]
    fn least_estimated_work_breaks_equal_scores_by_least_recent_selection() {
        let policy = LeastEstimatedWorkPolicy::new(vec![
            backend_state_receiver_for_device(InferenceDevice::Cpu, OperationalState::default()),
            backend_state_receiver_for_device(InferenceDevice::Cpu, OperationalState::default()),
        ]);

        let backend_indices: Vec<_> = (0..4)
            .map(|_| policy.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1]);
    }

    #[test]
    fn most_cached_prefix_breaks_equal_matches_by_least_recent_selection() {
        let policy = MostCachedPrefixPolicy::new(backend_state_receivers(2));

        let backend_indices: Vec<_> = (0..4)
            .map(|_| policy.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1]);
    }

    #[test]
    fn cache_aware_least_estimated_work_accounts_for_backend_load() {
        let policy = CacheAwareLeastEstimatedWorkPolicy::new(vec![
            backend_state_receiver(OperationalState {
                active_remaining_prefill_token_count: 20,
                ..Default::default()
            }),
            backend_state_receiver(OperationalState::default()),
        ]);
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            input_token_count: 10,
            max_new_token_count: 5,
        };

        let decision = policy.route_request(request);

        // Backend 0: 20 existing + 10 input + 5 output = 35 tokens.
        // Backend 1: 10 input + 5 output = 15 tokens.
        assert_eq!(decision.backend_index, 1);
    }

    #[test]
    fn cache_aware_least_estimated_work_accounts_for_backend_throughput() {
        let policy = CacheAwareLeastEstimatedWorkPolicy::new(vec![
            backend_state_receiver_for_device(
                InferenceDevice::Cpu,
                OperationalState {
                    active_remaining_prefill_token_count: 3,
                    ..Default::default()
                },
            ),
            backend_state_receiver_for_device(InferenceDevice::Gpu, OperationalState::default()),
        ]);
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            input_token_count: 10,
            max_new_token_count: 5,
        };

        let decision = policy.route_request(request);

        // CPU: 18 / 9.22; GPU: 15 / 16.25, so the GPU finishes first.
        assert_eq!(decision.backend_index, 1);
    }

    #[test]
    fn cache_aware_least_estimated_work_breaks_equal_scores_by_least_recent_selection() {
        let policy = CacheAwareLeastEstimatedWorkPolicy::new(backend_state_receivers(2));

        let backend_indices: Vec<_> = (0..4)
            .map(|_| policy.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1]);
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

        assert_eq!(estimate_remaining_token_work(&state, request, 4), 109);
    }

    #[test]
    fn cached_prefix_token_count_cannot_underflow_incoming_work() {
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2],
            input_token_count: 2,
            max_new_token_count: 3,
        };

        assert_eq!(
            estimate_remaining_token_work(&OperationalState::default(), request, 4),
            3
        );
    }
}
