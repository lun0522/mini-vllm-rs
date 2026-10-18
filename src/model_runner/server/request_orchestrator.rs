use crate::model_runner::InferenceDevice;
use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::RequestRoutingPolicy;
use anyhow::ensure;
use anyhow::Result;
use log::info;
use std::fmt;
use std::sync::Arc;
use tokio::sync::watch;

use super::kv_cache::PrefixCacheAffinitySnapshot;
use super::routing_policy::BackendRoutingPolicy;
use super::routing_policy::CacheAwareLeastEstimatedWorkPolicy;
use super::routing_policy::LeastEstimatedWorkPolicy;
use super::routing_policy::LeastRequestsPolicy;
use super::routing_policy::MostCachedPrefixPolicy;
use super::routing_policy::RoundRobinPolicy;
use super::BackendId;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct OperationalState {
    pub(super) epoch: u64,
    pub(super) queued_request_count: usize,
    pub(super) active_request_count: usize,
    pub(super) queued_remaining_prefill_token_count: usize,
    pub(super) queued_remaining_max_output_token_count: usize,
    pub(super) active_remaining_prefill_token_count: usize,
    pub(super) active_remaining_max_output_token_count: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct CacheAffinityState {
    pub(super) epoch: u64,
    pub(super) prefix_cache: Option<Arc<PrefixCacheAffinitySnapshot>>,
}

impl CacheAffinityState {
    pub(super) fn new(epoch: u64, prefix_cache: Option<PrefixCacheAffinitySnapshot>) -> Self {
        Self {
            epoch,
            prefix_cache: prefix_cache.map(Arc::new),
        }
    }
}

pub(super) struct BackendStateReceivers {
    pub(super) device: InferenceDevice,
    pub(super) operational: watch::Receiver<OperationalState>,
    pub(super) cache_affinity: watch::Receiver<CacheAffinityState>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RoutingDecision {
    pub(super) backend_id: BackendId,
    pub(super) candidate_diagnostics: Vec<RoutingCandidateDiagnostics>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RoutingCandidateDiagnostics {
    pub(super) backend_id: BackendId,
    pub(super) operational_state: Option<OperationalState>,
    pub(super) cache_affinity_state_epoch: Option<u64>,
    pub(super) cached_prefix_token_count: Option<usize>,
    pub(super) estimated_prefill_token_count: Option<u128>,
    pub(super) estimated_decode_token_count: Option<u128>,
    pub(super) prefill_tokens_per_second: Option<u64>,
    pub(super) decode_tokens_per_second: Option<u64>,
    pub(super) last_selected_epoch: Option<u64>,
}

impl RoutingCandidateDiagnostics {
    pub(super) fn new(backend_id: BackendId) -> Self {
        Self {
            backend_id,
            operational_state: None,
            cache_affinity_state_epoch: None,
            cached_prefix_token_count: None,
            estimated_prefill_token_count: None,
            estimated_decode_token_count: None,
            prefill_tokens_per_second: None,
            decode_tokens_per_second: None,
            last_selected_epoch: None,
        }
    }
}

impl fmt::Display for RoutingCandidateDiagnostics {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "backend_id={}", self.backend_id)?;
        if let Some(state) = self.operational_state {
            write!(
                formatter,
                " operational_epoch={} queued_requests={} active_requests={} \
                 queued_remaining_prefill_tokens={} queued_remaining_max_output_tokens={} \
                 active_remaining_prefill_tokens={} active_remaining_max_output_tokens={}",
                state.epoch,
                state.queued_request_count,
                state.active_request_count,
                state.queued_remaining_prefill_token_count,
                state.queued_remaining_max_output_token_count,
                state.active_remaining_prefill_token_count,
                state.active_remaining_max_output_token_count,
            )?;
        }
        if let Some(epoch) = self.cache_affinity_state_epoch {
            write!(formatter, " cache_affinity_epoch={epoch}")?;
        }
        if let Some(cached_prefix_token_count) = self.cached_prefix_token_count {
            write!(
                formatter,
                " cached_prefix_tokens={cached_prefix_token_count}"
            )?;
        }
        if let Some(estimated_prefill_token_count) = self.estimated_prefill_token_count {
            write!(
                formatter,
                " estimated_prefill_tokens={estimated_prefill_token_count}"
            )?;
        }
        if let Some(estimated_decode_token_count) = self.estimated_decode_token_count {
            write!(
                formatter,
                " estimated_decode_tokens={estimated_decode_token_count}"
            )?;
        }
        if let Some(prefill_tokens_per_second) = self.prefill_tokens_per_second {
            write!(
                formatter,
                " prefill_tokens_per_second={prefill_tokens_per_second}"
            )?;
        }
        if let Some(decode_tokens_per_second) = self.decode_tokens_per_second {
            write!(
                formatter,
                " decode_tokens_per_second={decode_tokens_per_second}"
            )?;
        }
        if let Some(last_selected_epoch) = self.last_selected_epoch {
            write!(formatter, " last_selected_epoch={last_selected_epoch}")?;
        }
        Ok(())
    }
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
        let RequestOrchestratorConfig {
            routing_policy,
            throughput_profiles,
        } = config;
        let policy: Box<dyn BackendRoutingPolicy> = match routing_policy {
            RequestRoutingPolicy::RoundRobin => Box::new(RoundRobinPolicy::new(backend_count)),
            RequestRoutingPolicy::LeastRequests => {
                Box::new(LeastRequestsPolicy::new(backend_state_receivers))
            }
            RequestRoutingPolicy::LeastEstimatedWork => Box::new(LeastEstimatedWorkPolicy::new(
                backend_state_receivers,
                throughput_profiles,
            )),
            RequestRoutingPolicy::MostCachedPrefix => {
                Box::new(MostCachedPrefixPolicy::new(backend_state_receivers))
            }
            RequestRoutingPolicy::CacheAwareLeastEstimatedWork => {
                Box::new(CacheAwareLeastEstimatedWorkPolicy::new(
                    backend_state_receivers,
                    throughput_profiles,
                ))
            }
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
                throughput_profiles: vec![Default::default()],
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
