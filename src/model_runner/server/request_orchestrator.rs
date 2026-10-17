use crate::model_runner::InferenceDevice;
use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::RequestRoutingPolicy;
use anyhow::ensure;
use anyhow::Result;
use log::info;
use std::sync::Arc;
use tokio::sync::watch;

use super::kv_cache::PrefixCacheAffinitySnapshot;
use super::routing_policy::BackendRoutingPolicy;
use super::routing_policy::CacheAwareLeastEstimatedWorkPolicy;
use super::routing_policy::LeastEstimatedWorkPolicy;
use super::routing_policy::LeastRequestsPolicy;
use super::routing_policy::MostCachedPrefixPolicy;
use super::routing_policy::RoundRobinPolicy;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct OperationalState {
    pub(super) queued_request_count: usize,
    pub(super) active_request_count: usize,
    pub(super) queued_remaining_prefill_token_count: usize,
    pub(super) queued_remaining_max_output_token_count: usize,
    pub(super) active_remaining_prefill_token_count: usize,
    pub(super) active_remaining_max_output_token_count: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct CacheAffinityState {
    pub(super) prefix_cache: Option<Arc<PrefixCacheAffinitySnapshot>>,
}

impl CacheAffinityState {
    pub(super) fn new(prefix_cache: Option<PrefixCacheAffinitySnapshot>) -> Self {
        Self {
            prefix_cache: prefix_cache.map(Arc::new),
        }
    }
}

pub(super) struct BackendStateReceivers {
    pub(super) device: InferenceDevice,
    pub(super) operational: watch::Receiver<OperationalState>,
    pub(super) cache_affinity: watch::Receiver<CacheAffinityState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RoutingDecision {
    pub(super) backend_index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RequestRoutingContext<'a> {
    pub(super) input_token_ids: &'a [u32],
    pub(super) input_token_count: usize,
    pub(super) max_new_token_count: u64,
}

impl RequestRoutingContext<'_> {
    pub(super) fn restorable_input_token_ids(&self) -> &[u32] {
        self.input_token_ids
            .split_last()
            .map_or(&[], |(_, prefix)| prefix)
    }
}

/// Selects an inference backend for each request without owning or dispatching to the backend.
pub(super) struct RequestOrchestrator {
    policy: Box<dyn BackendRoutingPolicy>,
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
        let policy: Box<dyn BackendRoutingPolicy> = match config.routing_policy {
            RequestRoutingPolicy::RoundRobin => Box::new(RoundRobinPolicy::new(backend_count)),
            RequestRoutingPolicy::LeastRequests => {
                Box::new(LeastRequestsPolicy::new(backend_state_receivers))
            }
            RequestRoutingPolicy::LeastEstimatedWork => {
                Box::new(LeastEstimatedWorkPolicy::new(backend_state_receivers))
            }
            RequestRoutingPolicy::MostCachedPrefix => {
                Box::new(MostCachedPrefixPolicy::new(backend_state_receivers))
            }
            RequestRoutingPolicy::CacheAwareLeastEstimatedWork => Box::new(
                CacheAwareLeastEstimatedWorkPolicy::new(backend_state_receivers),
            ),
        };
        Ok(Self { policy })
    }

    /// Selects a backend according to the configured routing policy.
    pub(super) fn route_request(&self, request: RequestRoutingContext<'_>) -> RoutingDecision {
        self.policy.route_request(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excludes_the_final_input_token_from_cache_affinity_matching() {
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3],
            input_token_count: 3,
            max_new_token_count: 1,
        };

        assert_eq!(request.restorable_input_token_ids(), &[1, 2]);
    }

    #[test]
    fn rejects_an_empty_backend_set() {
        let error = RequestOrchestrator::new(
            Vec::new(),
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::RoundRobin,
            },
        )
        .err()
        .unwrap();

        assert_eq!(
            error.to_string(),
            "request orchestration requires a backend"
        );
    }
}
