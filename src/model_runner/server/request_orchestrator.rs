use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::RequestRoutingPolicy;
use anyhow::ensure;
use anyhow::Result;
use log::info;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use tokio::sync::watch;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct OperationalState {
    pub(super) queued_request_count: usize,
    pub(super) active_request_count: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CacheAffinityState {
    pub(super) indexed_token_count: usize,
}

pub(super) struct BackendStateReceivers {
    pub(super) operational: watch::Receiver<OperationalState>,
    pub(super) cache_affinity: watch::Receiver<CacheAffinityState>,
}

/// Selects an inference backend for each request without owning or dispatching to the backend.
pub(super) struct RequestOrchestrator {
    policy: RoutingPolicy,
}

enum RoutingPolicy {
    RoundRobin {
        backend_count: usize,
        next_backend_index: AtomicUsize,
    },
    LeastRequests {
        operational_state_receivers: Vec<watch::Receiver<OperationalState>>,
        selection_history: Mutex<BackendSelectionHistory>,
    },
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RoutingDecision {
    pub(super) backend_index: usize,
}

impl RequestOrchestrator {
    pub(super) fn new(
        backend_state_receivers: Vec<BackendStateReceivers>,
        config: RequestOrchestratorConfig,
    ) -> Result<Self> {
        let backend_count = backend_state_receivers.len();
        ensure!(
            backend_count > 0,
            "request orchestration requires a backend"
        );
        info!("Request orchestrator config: {config}");
        let policy = match config.routing_policy {
            RequestRoutingPolicy::RoundRobin => RoutingPolicy::RoundRobin {
                backend_count,
                next_backend_index: AtomicUsize::new(0),
            },
            RequestRoutingPolicy::LeastRequests => RoutingPolicy::LeastRequests {
                operational_state_receivers: backend_state_receivers
                    .into_iter()
                    .map(
                        |BackendStateReceivers {
                             operational,
                             cache_affinity,
                         }| {
                            drop(cache_affinity);
                            operational
                        },
                    )
                    .collect(),
                selection_history: Mutex::new(BackendSelectionHistory::new(backend_count)),
            },
        };
        Ok(Self { policy })
    }

    /// Selects a backend according to the configured routing policy.
    pub(super) fn route_request(&self) -> RoutingDecision {
        let backend_index = match &self.policy {
            RoutingPolicy::RoundRobin {
                backend_count,
                next_backend_index,
            } => next_backend_index.fetch_add(1, Ordering::Relaxed) % backend_count,
            RoutingPolicy::LeastRequests {
                operational_state_receivers,
                selection_history,
            } => {
                let mut selection_history = selection_history
                    .lock()
                    .expect("backend selection history mutex is poisoned");
                let backend_index = operational_state_receivers
                    .iter()
                    .enumerate()
                    .min_by_key(|(backend_index, receiver)| {
                        let state = receiver.borrow();
                        (
                            state.queued_request_count + state.active_request_count,
                            selection_history.last_selected_epoch[*backend_index],
                        )
                    })
                    .map(|(backend_index, _)| backend_index)
                    .expect("the orchestrator always has at least one backend");
                selection_history.record_selection(backend_index);
                backend_index
            }
        };
        RoutingDecision { backend_index }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend_state_receiver(operational_state: OperationalState) -> BackendStateReceivers {
        let (_operational_sender, operational) = watch::channel(operational_state);
        let (_cache_affinity_sender, cache_affinity) =
            watch::channel(CacheAffinityState::default());
        BackendStateReceivers {
            operational,
            cache_affinity,
        }
    }

    fn backend_state_receivers(count: usize) -> Vec<BackendStateReceivers> {
        (0..count)
            .map(|_| backend_state_receiver(OperationalState::default()))
            .collect()
    }

    fn round_robin_config() -> RequestOrchestratorConfig {
        RequestOrchestratorConfig {
            routing_policy: RequestRoutingPolicy::RoundRobin,
        }
    }

    #[test]
    fn routes_requests_in_round_robin_order() {
        let orchestrator =
            RequestOrchestrator::new(backend_state_receivers(2), round_robin_config()).unwrap();

        let backend_indices: Vec<_> = (0..5)
            .map(|_| orchestrator.route_request().backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn routes_every_request_to_the_only_backend() {
        let orchestrator =
            RequestOrchestrator::new(backend_state_receivers(1), round_robin_config()).unwrap();

        for _ in 0..3 {
            assert_eq!(orchestrator.route_request().backend_index, 0);
        }
    }

    #[test]
    fn rejects_an_empty_backend_set() {
        let error = RequestOrchestrator::new(backend_state_receivers(0), round_robin_config())
            .err()
            .unwrap();

        assert_eq!(
            error.to_string(),
            "request orchestration requires a backend"
        );
    }

    #[test]
    fn least_requests_observes_latest_operational_state_publications() {
        let (operational_sender, operational) = watch::channel(OperationalState::default());
        let (_cache_affinity_sender, cache_affinity) =
            watch::channel(CacheAffinityState::default());
        let orchestrator = RequestOrchestrator::new(
            vec![BackendStateReceivers {
                operational,
                cache_affinity,
            }],
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::LeastRequests,
            },
        )
        .unwrap();

        operational_sender.send_replace(OperationalState {
            queued_request_count: 2,
            active_request_count: 1,
        });
        operational_sender.send_replace(OperationalState {
            queued_request_count: 1,
            active_request_count: 1,
        });

        assert_eq!(orchestrator.route_request().backend_index, 0);
    }

    #[test]
    fn routes_to_the_backend_with_the_fewest_requests() {
        let orchestrator = RequestOrchestrator::new(
            vec![
                backend_state_receiver(OperationalState {
                    queued_request_count: 2,
                    active_request_count: 1,
                }),
                backend_state_receiver(OperationalState {
                    queued_request_count: 0,
                    active_request_count: 2,
                }),
            ],
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::LeastRequests,
            },
        )
        .unwrap();

        assert_eq!(orchestrator.route_request().backend_index, 1);
    }

    #[test]
    fn least_requests_breaks_ties_by_least_recent_selection() {
        let orchestrator = RequestOrchestrator::new(
            backend_state_receivers(2),
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::LeastRequests,
            },
        )
        .unwrap();

        let backend_indices: Vec<_> = (0..4)
            .map(|_| orchestrator.route_request().backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1]);
    }
}
