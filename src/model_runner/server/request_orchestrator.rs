use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::RequestRoutingPolicy;
use anyhow::ensure;
use anyhow::Result;
use log::info;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use tokio::sync::watch;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct OperationalState {
    pub(super) request_count: usize,
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
    backend_count: usize,
    routing_policy: RequestRoutingPolicy,
    next_backend_index: AtomicUsize,
    backend_state_receivers: Vec<BackendStateReceivers>,
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
        Ok(Self {
            backend_count,
            routing_policy: config.routing_policy,
            next_backend_index: AtomicUsize::new(0),
            backend_state_receivers,
        })
    }

    /// Selects backends in round-robin order.
    pub(super) fn route_request(&self) -> RoutingDecision {
        // Collect stable copies of the latest publications. Round robin does not use them yet,
        // but future policies can rank these observations without changing worker plumbing.
        let _backend_states = self.backend_states();
        match self.routing_policy {
            RequestRoutingPolicy::RoundRobin => RoutingDecision {
                backend_index: self.next_backend_index.fetch_add(1, Ordering::Relaxed)
                    % self.backend_count,
            },
        }
    }

    fn backend_states(&self) -> Vec<(OperationalState, CacheAffinityState)> {
        self.backend_state_receivers
            .iter()
            .map(|receivers| {
                (
                    *receivers.operational.borrow(),
                    *receivers.cache_affinity.borrow(),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend_state_receivers(count: usize) -> Vec<BackendStateReceivers> {
        (0..count)
            .map(|_| {
                let (_operational_sender, operational) =
                    watch::channel(OperationalState::default());
                let (_cache_affinity_sender, cache_affinity) =
                    watch::channel(CacheAffinityState::default());
                BackendStateReceivers {
                    operational,
                    cache_affinity,
                }
            })
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
    fn observes_latest_worker_state_publications() {
        let (operational_sender, operational) = watch::channel(OperationalState::default());
        let (cache_affinity_sender, cache_affinity) = watch::channel(CacheAffinityState::default());
        let orchestrator = RequestOrchestrator::new(
            vec![BackendStateReceivers {
                operational,
                cache_affinity,
            }],
            round_robin_config(),
        )
        .unwrap();

        operational_sender.send_replace(OperationalState { request_count: 3 });
        operational_sender.send_replace(OperationalState { request_count: 2 });
        cache_affinity_sender.send_replace(CacheAffinityState {
            indexed_token_count: 128,
        });

        assert_eq!(
            orchestrator.backend_states(),
            vec![(
                OperationalState { request_count: 2 },
                CacheAffinityState {
                    indexed_token_count: 128,
                },
            )]
        );
    }
}
