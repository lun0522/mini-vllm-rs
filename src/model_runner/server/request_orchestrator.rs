use crate::model_runner::InferenceDevice;
use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::RequestRoutingPolicy;
use anyhow::ensure;
use anyhow::Result;
use log::info;
use std::fmt;
use std::sync::Arc;
use std::sync::Mutex;
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

/// Tracks requests routed to a backend but not yet reflected in its operational snapshot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BackendReservedLoad {
    pub(super) request_count: usize,
    pub(super) prefill_token_count: usize,
    pub(super) cache_adjusted_prefill_token_count: usize,
    pub(super) max_output_token_count: u64,
}

struct BackendReservations {
    backend_loads: Vec<BackendReservedLoad>,
}

impl BackendReservations {
    fn new(backend_count: usize) -> Self {
        Self {
            backend_loads: vec![BackendReservedLoad::default(); backend_count],
        }
    }

    fn reserve(&mut self, backend_id: BackendId, load: BackendReservedLoad) {
        let reserved_load = &mut self.backend_loads[backend_id.index()];
        reserved_load.request_count = reserved_load
            .request_count
            .saturating_add(load.request_count);
        reserved_load.prefill_token_count = reserved_load
            .prefill_token_count
            .saturating_add(load.prefill_token_count);
        reserved_load.cache_adjusted_prefill_token_count = reserved_load
            .cache_adjusted_prefill_token_count
            .saturating_add(load.cache_adjusted_prefill_token_count);
        reserved_load.max_output_token_count = reserved_load
            .max_output_token_count
            .saturating_add(load.max_output_token_count);
    }

    fn release(&mut self, backend_id: BackendId, load: BackendReservedLoad) {
        let reserved_load = &mut self.backend_loads[backend_id.index()];
        reserved_load.request_count = reserved_load
            .request_count
            .checked_sub(load.request_count)
            .expect("released request count exceeds reserved request count");
        reserved_load.prefill_token_count = reserved_load
            .prefill_token_count
            .checked_sub(load.prefill_token_count)
            .expect("released prefill token count exceeds reserved prefill token count");
        reserved_load.cache_adjusted_prefill_token_count = reserved_load
            .cache_adjusted_prefill_token_count
            .checked_sub(load.cache_adjusted_prefill_token_count)
            .expect("released cache-adjusted prefill token count exceeds reserved token count");
        reserved_load.max_output_token_count = reserved_load
            .max_output_token_count
            .checked_sub(load.max_output_token_count)
            .expect("released output token count exceeds reserved output token count");
    }
}

/// Releases a backend's reserved load when a routed request is dropped or observed by a worker.
pub(super) struct RoutingReservation {
    backend_id: BackendId,
    load: Option<BackendReservedLoad>,
    reservations: Arc<Mutex<BackendReservations>>,
}

impl RoutingReservation {
    fn new(
        backend_id: BackendId,
        load: BackendReservedLoad,
        reservations: Arc<Mutex<BackendReservations>>,
    ) -> Self {
        Self {
            backend_id,
            load: Some(load),
            reservations,
        }
    }

    /// Publishes the worker's updated load and removes the corresponding reservation atomically
    /// with respect to routing decisions.
    pub(super) fn release_after(mut self, publish_operational_state: impl FnOnce()) {
        let load = self
            .load
            .take()
            .expect("routing reservation must not be released twice");
        let mut reservations = self
            .reservations
            .lock()
            .expect("backend reservations mutex is poisoned");
        publish_operational_state();
        reservations.release(self.backend_id, load);
    }
}

impl Drop for RoutingReservation {
    fn drop(&mut self) {
        let Some(load) = self.load.take() else {
            return;
        };
        self.reservations
            .lock()
            .expect("backend reservations mutex is poisoned")
            .release(self.backend_id, load);
    }
}

/// Selects an inference backend for each request without owning or dispatching to the backend.
pub(super) struct RequestOrchestrator {
    policy: Box<dyn BackendRoutingPolicy>,
    // Keep critical sections over this mutex small because inference workers also acquire it to
    // publish updated operational state and release the corresponding reservations.
    reservations: Arc<Mutex<BackendReservations>>,
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
        Ok(Self {
            policy,
            reservations: Arc::new(Mutex::new(BackendReservations::new(backend_count))),
        })
    }

    /// Selects a backend according to the configured routing policy.
    pub(super) fn route_request(
        &self,
        request: RequestRoutingContext<'_>,
    ) -> (RoutingDecision, RoutingReservation) {
        let mut reservations = self
            .reservations
            .lock()
            .expect("backend reservations mutex is poisoned");
        let decision = self.policy.route_request(request);
        let reserved_load = BackendReservedLoad::default();
        reservations.reserve(decision.backend_id, reserved_load);
        let reservation = RoutingReservation::new(
            decision.backend_id,
            reserved_load,
            Arc::clone(&self.reservations),
        );
        (decision, reservation)
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
                    device: InferenceDevice::Cpu,
                    operational,
                    cache_affinity,
                }
            })
            .collect()
    }

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

    #[test]
    fn records_zero_reservations_without_changing_routing() {
        let orchestrator = RequestOrchestrator::new(
            backend_state_receivers(2),
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::RoundRobin,
                throughput_profiles: vec![Default::default()],
            },
        )
        .unwrap();
        let request = RequestRoutingContext {
            input_token_ids: &[1],
            input_token_count: 1,
            max_new_token_count: 1,
        };

        assert_eq!(
            orchestrator.route_request(request).0.backend_id,
            BackendId::from_index(0)
        );
        assert_eq!(
            orchestrator.route_request(request).0.backend_id,
            BackendId::from_index(1)
        );
        assert_eq!(
            orchestrator.reservations.lock().unwrap().backend_loads,
            vec![BackendReservedLoad::default(); 2]
        );
    }

    #[test]
    fn routing_reservation_releases_load_when_dropped() {
        let backend_id = BackendId::from_index(0);
        let load = BackendReservedLoad {
            request_count: 1,
            prefill_token_count: 2,
            cache_adjusted_prefill_token_count: 3,
            max_output_token_count: 4,
        };
        let reservations = Arc::new(Mutex::new(BackendReservations::new(1)));
        reservations.lock().unwrap().reserve(backend_id, load);

        drop(RoutingReservation::new(
            backend_id,
            load,
            Arc::clone(&reservations),
        ));

        assert_eq!(
            reservations.lock().unwrap().backend_loads,
            vec![BackendReservedLoad::default()]
        );
    }
}
