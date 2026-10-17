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
                        let throughput_weight = match device {
                            InferenceDevice::Cpu => CPU_THROUGHPUT_WEIGHT,
                            InferenceDevice::Gpu => GPU_THROUGHPUT_WEIGHT,
                            InferenceDevice::Mixed => {
                                unreachable!("an inference backend has one concrete device")
                            }
                        };
                        WeightedOperationalStateReceiver {
                            operational_state_receiver: operational,
                            throughput_weight,
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
                let estimated_token_count = (state.queued_remaining_prefill_token_count as u128)
                    .saturating_add(state.queued_remaining_max_output_token_count as u128)
                    .saturating_add(state.active_remaining_prefill_token_count as u128)
                    .saturating_add(state.active_remaining_max_output_token_count as u128)
                    .saturating_add(request.input_token_count as u128)
                    .saturating_add(request.max_new_token_count as u128);
                EstimatedCompletionScore {
                    estimated_token_count,
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
                    prefix_cache.find_longest_prefix_token_count(request.input_token_ids)
                }))
            },
        );
        RoutingDecision { backend_index }
    }
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
