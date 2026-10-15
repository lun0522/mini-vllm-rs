use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::RequestRoutingPolicy;
use anyhow::ensure;
use anyhow::Result;
use log::info;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// Selects an inference backend for each request without owning or dispatching to the backend.
pub(super) struct RequestOrchestrator {
    backend_count: usize,
    routing_policy: RequestRoutingPolicy,
    next_backend_index: AtomicUsize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RoutingDecision {
    pub(super) backend_index: usize,
}

impl RequestOrchestrator {
    pub(super) fn new(backend_count: usize, config: RequestOrchestratorConfig) -> Result<Self> {
        ensure!(
            backend_count > 0,
            "request orchestration requires a backend"
        );
        info!("Request orchestrator config: {config}");
        Ok(Self {
            backend_count,
            routing_policy: config.routing_policy,
            next_backend_index: AtomicUsize::new(0),
        })
    }

    /// Selects backends in round-robin order.
    pub(super) fn route_request(&self) -> RoutingDecision {
        match self.routing_policy {
            RequestRoutingPolicy::RoundRobin => RoutingDecision {
                backend_index: self.next_backend_index.fetch_add(1, Ordering::Relaxed)
                    % self.backend_count,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_robin_config() -> RequestOrchestratorConfig {
        RequestOrchestratorConfig {
            routing_policy: RequestRoutingPolicy::RoundRobin,
        }
    }

    #[test]
    fn routes_requests_in_round_robin_order() {
        let orchestrator = RequestOrchestrator::new(2, round_robin_config()).unwrap();

        let backend_indices: Vec<_> = (0..5)
            .map(|_| orchestrator.route_request().backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn routes_every_request_to_the_only_backend() {
        let orchestrator = RequestOrchestrator::new(1, round_robin_config()).unwrap();

        for _ in 0..3 {
            assert_eq!(orchestrator.route_request().backend_index, 0);
        }
    }

    #[test]
    fn rejects_an_empty_backend_set() {
        let error = RequestOrchestrator::new(0, round_robin_config())
            .err()
            .unwrap();

        assert_eq!(
            error.to_string(),
            "request orchestration requires a backend"
        );
    }
}
