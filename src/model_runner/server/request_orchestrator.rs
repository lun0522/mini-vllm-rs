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

impl OperationalState {
    /// Includes dispatched requests not yet observed by the worker as additional queued work.
    /// The epoch still identifies the original worker publication.
    pub(super) fn with_reserved_load(mut self, reserved_load: BackendReservedLoad) -> Self {
        self.queued_request_count = self
            .queued_request_count
            .saturating_add(reserved_load.request_count);
        // Policies that do not estimate cache affinity reserve zero cached-prefix tokens.
        let prefill_token_count = reserved_load
            .prefill_token_count
            .saturating_sub(reserved_load.cached_prefix_token_count);
        self.queued_remaining_prefill_token_count = self
            .queued_remaining_prefill_token_count
            .saturating_add(prefill_token_count);
        self.queued_remaining_max_output_token_count =
            self.queued_remaining_max_output_token_count.saturating_add(
                usize::try_from(reserved_load.max_output_token_count).unwrap_or(usize::MAX),
            );
        self
    }
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
    /// One-based selection order within this orchestrator, independent of log emission order.
    /// Assigned during selection while the orchestrator holds the routing-state lock.
    pub(super) routing_epoch: u64,
    pub(super) backend_id: BackendId,
    pub(super) estimated_cached_prefix_token_count: usize,
    pub(super) candidate_diagnostics: Vec<RoutingCandidateDiagnostics>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RoutingCandidateDiagnostics {
    pub(super) backend_id: BackendId,
    pub(super) operational_state: Option<OperationalState>,
    pub(super) reserved_load: Option<BackendReservedLoad>,
    pub(super) cache_affinity_state_epoch: Option<u64>,
    pub(super) cached_prefix_token_count: Option<usize>,
    pub(super) estimated_prefill_token_count: Option<u128>,
    pub(super) estimated_decode_token_count: Option<u128>,
    /// The exact nanosecond score used by estimated-work policies, not elapsed time.
    pub(super) estimated_duration_ns: Option<u128>,
    pub(super) prefill_tokens_per_second: Option<u64>,
    pub(super) decode_tokens_per_second: Option<u64>,
    pub(super) last_selected_epoch: Option<u64>,
}

impl RoutingCandidateDiagnostics {
    pub(super) fn new(backend_id: BackendId) -> Self {
        Self {
            backend_id,
            operational_state: None,
            reserved_load: None,
            cache_affinity_state_epoch: None,
            cached_prefix_token_count: None,
            estimated_prefill_token_count: None,
            estimated_decode_token_count: None,
            estimated_duration_ns: None,
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
        if let Some(load) = self.reserved_load {
            write!(
                formatter,
                " reserved_requests={} reserved_prefill_tokens={} reserved_cached_prefix_tokens={} \
                 reserved_max_output_tokens={}",
                load.request_count,
                load.prefill_token_count,
                load.cached_prefix_token_count,
                load.max_output_token_count,
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
        if let Some(estimated_duration_ns) = self.estimated_duration_ns {
            write!(formatter, " estimated_duration_ns={estimated_duration_ns}")?;
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
    pub(super) backend_id_override: Option<BackendId>,
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
    pub(super) cached_prefix_token_count: usize,
    pub(super) max_output_token_count: u64,
}

impl BackendReservedLoad {
    fn new(request: RequestRoutingContext<'_>, cached_prefix_token_count: usize) -> Self {
        Self {
            request_count: 1,
            prefill_token_count: request.input_token_count,
            cached_prefix_token_count: cached_prefix_token_count.min(request.input_token_count),
            max_output_token_count: request.max_new_token_count,
        }
    }
}

/// Mutable selection and pending-load bookkeeping for one inference backend.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct BackendRoutingState {
    pub(super) reserved_load: BackendReservedLoad,
    pub(super) last_selected_epoch: u64,
}

/// Coordinates backend selection order and pending reservations under one shared lock.
pub(super) struct OrchestratorRoutingState {
    pub(super) selection_epoch: u64,
    pub(super) backends: Vec<BackendRoutingState>,
}

impl OrchestratorRoutingState {
    pub(super) fn new(backend_count: usize) -> Self {
        Self {
            selection_epoch: 0,
            backends: vec![BackendRoutingState::default(); backend_count],
        }
    }

    pub(super) fn record_backend_selection(&mut self, backend_index: usize) {
        self.selection_epoch = self
            .selection_epoch
            .checked_add(1)
            .expect("routing epoch exceeds u64");
        self.backends[backend_index].last_selected_epoch = self.selection_epoch;
    }

    fn reserve_load(&mut self, backend_id: BackendId, load: BackendReservedLoad) {
        let reserved_load = &mut self.backends[backend_id.index()].reserved_load;
        reserved_load.request_count = reserved_load
            .request_count
            .saturating_add(load.request_count);
        reserved_load.prefill_token_count = reserved_load
            .prefill_token_count
            .saturating_add(load.prefill_token_count);
        reserved_load.cached_prefix_token_count = reserved_load
            .cached_prefix_token_count
            .saturating_add(load.cached_prefix_token_count);
        reserved_load.max_output_token_count = reserved_load
            .max_output_token_count
            .saturating_add(load.max_output_token_count);
    }

    fn release_load(&mut self, backend_id: BackendId, load: BackendReservedLoad) {
        let reserved_load = &mut self.backends[backend_id.index()].reserved_load;
        reserved_load.request_count = reserved_load
            .request_count
            .checked_sub(load.request_count)
            .expect("released request count exceeds reserved request count");
        reserved_load.prefill_token_count = reserved_load
            .prefill_token_count
            .checked_sub(load.prefill_token_count)
            .expect("released prefill token count exceeds reserved prefill token count");
        reserved_load.cached_prefix_token_count = reserved_load
            .cached_prefix_token_count
            .checked_sub(load.cached_prefix_token_count)
            .expect("released cached prefix token count exceeds reserved token count");
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
    routing_state: Arc<Mutex<OrchestratorRoutingState>>,
}

impl RoutingReservation {
    fn new(
        backend_id: BackendId,
        load: BackendReservedLoad,
        routing_state: Arc<Mutex<OrchestratorRoutingState>>,
    ) -> Self {
        Self {
            backend_id,
            load: Some(load),
            routing_state,
        }
    }

    /// Publishes the worker's updated load and removes the corresponding reservation atomically
    /// with respect to routing decisions.
    pub(super) fn release_after(mut self, publish_operational_state: impl FnOnce()) {
        let load = self
            .load
            .take()
            .expect("routing reservation must not be released twice");
        let mut routing_state = self
            .routing_state
            .lock()
            .expect("orchestrator routing state mutex is poisoned");
        publish_operational_state();
        routing_state.release_load(self.backend_id, load);
    }
}

impl Drop for RoutingReservation {
    fn drop(&mut self) {
        let Some(load) = self.load.take() else {
            return;
        };
        self.routing_state
            .lock()
            .expect("orchestrator routing state mutex is poisoned")
            .release_load(self.backend_id, load);
    }
}

/// Selects an inference backend for each request without owning or dispatching to the backend.
pub(super) struct RequestOrchestrator {
    policy: Box<dyn BackendRoutingPolicy>,
    // Keep critical sections over this mutex small because inference workers also acquire it to
    // publish updated operational state and release the corresponding reservations.
    routing_state: Arc<Mutex<OrchestratorRoutingState>>,
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
            RequestRoutingPolicy::RoundRobin => Box::new(RoundRobinPolicy),
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
            routing_state: Arc::new(Mutex::new(OrchestratorRoutingState::new(backend_count))),
        })
    }

    /// Selects a backend according to the configured routing policy.
    pub(super) fn route_request(
        &self,
        request: RequestRoutingContext<'_>,
    ) -> (RoutingDecision, RoutingReservation) {
        let mut routing_state = self
            .routing_state
            .lock()
            .expect("orchestrator routing state mutex is poisoned");
        let decision = self.policy.route_request(request, &mut routing_state);
        let reserved_load =
            BackendReservedLoad::new(request, decision.estimated_cached_prefix_token_count);
        routing_state.reserve_load(decision.backend_id, reserved_load);
        let reservation = RoutingReservation::new(
            decision.backend_id,
            reserved_load,
            Arc::clone(&self.routing_state),
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
            backend_id_override: None,
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
    fn assigns_unique_routing_epochs_to_concurrent_requests() {
        for routing_policy in [
            RequestRoutingPolicy::RoundRobin,
            RequestRoutingPolicy::LeastRequests,
        ] {
            let orchestrator = RequestOrchestrator::new(
                backend_state_receivers(2),
                RequestOrchestratorConfig {
                    routing_policy,
                    throughput_profiles: vec![Default::default()],
                },
            )
            .unwrap();
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..16)
                    .map(|_| {
                        let orchestrator = &orchestrator;
                        scope.spawn(move || {
                            orchestrator.route_request(RequestRoutingContext {
                                input_token_ids: &[1],
                                input_token_count: 1,
                                max_new_token_count: 1,
                                backend_id_override: None,
                            })
                        })
                    })
                    .collect();
                let mut decisions: Vec<_> = handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect();
                decisions.sort_by_key(|(decision, _)| decision.routing_epoch);
                for (index, (decision, _)) in decisions.iter().enumerate() {
                    assert_eq!(decision.routing_epoch, index as u64 + 1);
                    assert_eq!(decision.backend_id, BackendId::from_index(index % 2));
                }
                // Releasing reservations must not reset the decision order.
                drop(decisions);
                let (next, _) = orchestrator.route_request(RequestRoutingContext {
                    input_token_ids: &[1],
                    input_token_count: 1,
                    max_new_token_count: 1,
                    backend_id_override: None,
                });
                assert_eq!(next.routing_epoch, 17);
            });
        }
    }

    #[test]
    fn records_request_load_and_releases_it_after_publication() {
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
            backend_id_override: None,
        };

        let (first_decision, first_reservation) = orchestrator.route_request(request);
        let (second_decision, second_reservation) = orchestrator.route_request(request);
        assert_eq!(first_decision.backend_id, BackendId::from_index(0));
        assert_eq!(second_decision.backend_id, BackendId::from_index(1));
        let expected_load = BackendReservedLoad {
            request_count: 1,
            prefill_token_count: 1,
            cached_prefix_token_count: 0,
            max_output_token_count: 1,
        };
        assert_eq!(
            orchestrator
                .routing_state
                .lock()
                .unwrap()
                .backends
                .iter()
                .map(|backend| backend.reserved_load)
                .collect::<Vec<_>>(),
            vec![expected_load; 2]
        );
        first_reservation.release_after(|| {
            // Publication happens before the reservation is removed.
            assert!(orchestrator.routing_state.try_lock().is_err());
        });
        drop(second_reservation);
        let routing_state = orchestrator.routing_state.lock().unwrap();
        assert_eq!(routing_state.selection_epoch, 2);
        assert_eq!(
            routing_state.backends,
            vec![
                BackendRoutingState {
                    last_selected_epoch: 1,
                    ..Default::default()
                },
                BackendRoutingState {
                    last_selected_epoch: 2,
                    ..Default::default()
                },
            ]
        );
    }

    #[test]
    fn backend_override_preserves_scoring_history_and_reservations() {
        let orchestrator = RequestOrchestrator::new(
            backend_state_receivers(2),
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::RoundRobin,
                throughput_profiles: vec![Default::default()],
            },
        )
        .unwrap();
        let overridden_backend_id = BackendId::from_index(1);

        let (decision, reservation) = orchestrator.route_request(RequestRoutingContext {
            input_token_ids: &[1, 2],
            input_token_count: 2,
            max_new_token_count: 3,
            backend_id_override: Some(overridden_backend_id),
        });

        assert_eq!(decision.backend_id, overridden_backend_id);
        assert_eq!(decision.candidate_diagnostics.len(), 2);
        let routing_state = orchestrator.routing_state.lock().unwrap();
        assert_eq!(routing_state.selection_epoch, 1);
        assert_eq!(routing_state.backends[0], BackendRoutingState::default());
        assert_eq!(
            routing_state.backends[1],
            BackendRoutingState {
                reserved_load: BackendReservedLoad {
                    request_count: 1,
                    prefill_token_count: 2,
                    cached_prefix_token_count: 0,
                    max_output_token_count: 3,
                },
                last_selected_epoch: 1,
            }
        );
        drop(routing_state);
        drop(reservation);
    }

    #[test]
    fn load_policies_account_for_unpublished_routed_requests() {
        for routing_policy in [
            RequestRoutingPolicy::LeastRequests,
            RequestRoutingPolicy::LeastEstimatedWork,
            RequestRoutingPolicy::CacheAwareLeastEstimatedWork,
        ] {
            let orchestrator = RequestOrchestrator::new(
                backend_state_receivers(2),
                RequestOrchestratorConfig {
                    routing_policy,
                    throughput_profiles: vec![Default::default()],
                },
            )
            .unwrap();
            let request = RequestRoutingContext {
                input_token_ids: &[1, 2],
                input_token_count: 2,
                max_new_token_count: 4,
                backend_id_override: None,
            };
            let (first, reservation) = orchestrator.route_request(request);
            assert_eq!(first.backend_id, BackendId::from_index(0));
            let (second, second_reservation) = orchestrator.route_request(request);
            assert_eq!(second.backend_id, BackendId::from_index(1));
            drop(second_reservation);
            // Least-recently-selected tie-breaking would choose backend 0. Its outstanding
            // reservation must instead direct the next request to backend 1.
            let (third, third_reservation) = orchestrator.route_request(request);
            assert_eq!(third.backend_id, BackendId::from_index(1));
            assert_eq!(
                third.candidate_diagnostics[0]
                    .reserved_load
                    .unwrap()
                    .request_count,
                1
            );
            assert_eq!(
                third.candidate_diagnostics[1]
                    .reserved_load
                    .unwrap()
                    .request_count,
                0
            );
            drop(third_reservation);
            drop(reservation);
        }
    }

    #[test]
    fn diagnostics_distinguish_published_and_reserved_load() {
        let diagnostics = RoutingCandidateDiagnostics {
            operational_state: Some(OperationalState {
                epoch: 7,
                queued_request_count: 2,
                active_request_count: 1,
                ..Default::default()
            }),
            reserved_load: Some(BackendReservedLoad {
                request_count: 3,
                prefill_token_count: 100,
                cached_prefix_token_count: 40,
                max_output_token_count: 20,
            }),
            ..RoutingCandidateDiagnostics::new(BackendId::from_index(0))
        };
        let log = diagnostics.to_string();
        assert!(log.contains("operational_epoch=7 queued_requests=2 active_requests=1"));
        assert!(log.contains("reserved_requests=3 reserved_prefill_tokens=100 reserved_cached_prefix_tokens=40 reserved_max_output_tokens=20"));
    }

    #[tokio::test]
    async fn dropped_dispatch_messages_release_reservations() {
        use crate::model_runner::server::InferenceRequest;
        use crate::model_runner::server::InferenceWorkerMessage;
        use crate::proto::model_runner::GenerateTextRequest;
        use std::time::Instant;
        use tokio::sync::mpsc;

        // Exercise both a failed send and a successfully queued message discarded on shutdown.
        for receiver_closed_before_send in [true, false] {
            let orchestrator = RequestOrchestrator::new(
                backend_state_receivers(1),
                RequestOrchestratorConfig {
                    routing_policy: RequestRoutingPolicy::LeastRequests,
                    throughput_profiles: vec![Default::default()],
                },
            )
            .unwrap();
            let (_, reservation) = orchestrator.route_request(RequestRoutingContext {
                input_token_ids: &[1],
                input_token_count: 1,
                max_new_token_count: 2,
                backend_id_override: None,
            });
            let (sender, receiver) = mpsc::channel(1);
            let mut receiver = Some(receiver);
            if receiver_closed_before_send {
                drop(receiver.take());
            }
            let (event_sender, _event_receiver) = mpsc::channel(1);
            let result = sender
                .send(InferenceWorkerMessage::Generate {
                    request: InferenceRequest {
                        queued_at: Instant::now(),
                        generate_text: GenerateTextRequest::default(),
                        event_sender,
                    },
                    routing_reservation: reservation,
                })
                .await;
            assert_eq!(result.is_err(), receiver_closed_before_send);
            if !receiver_closed_before_send {
                assert_eq!(
                    orchestrator.routing_state.lock().unwrap().backends[0]
                        .reserved_load
                        .request_count,
                    1
                );
            }
            // SendError owns the rejected message until it too is dropped.
            drop(result);
            drop(receiver);
            assert_eq!(
                orchestrator
                    .routing_state
                    .lock()
                    .unwrap()
                    .backends
                    .iter()
                    .map(|backend| backend.reserved_load)
                    .collect::<Vec<_>>(),
                vec![BackendReservedLoad::default()]
            );
        }
    }

    #[test]
    fn clamps_reserved_cached_prefix_to_input_length() {
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3, 4],
            input_token_count: 4,
            max_new_token_count: 8,
            backend_id_override: None,
        };
        for (cached_tokens, expected_cached_tokens) in [(0, 0), (2, 2), (4, 4), (5, 4)] {
            assert_eq!(
                BackendReservedLoad::new(request, cached_tokens),
                BackendReservedLoad {
                    request_count: 1,
                    prefill_token_count: 4,
                    cached_prefix_token_count: expected_cached_tokens,
                    max_output_token_count: 8,
                }
            );
        }
    }

    #[test]
    fn releasing_one_reservation_preserves_other_load_on_the_same_backend() {
        let backend_id = BackendId::from_index(0);
        let first_load = BackendReservedLoad {
            request_count: 1,
            prefill_token_count: 10,
            cached_prefix_token_count: 4,
            max_output_token_count: 8,
        };
        let second_load = BackendReservedLoad {
            request_count: 1,
            prefill_token_count: 20,
            cached_prefix_token_count: 6,
            max_output_token_count: 16,
        };
        let reservations = Arc::new(Mutex::new(OrchestratorRoutingState::new(1)));
        reservations
            .lock()
            .unwrap()
            .reserve_load(backend_id, first_load);
        let first = RoutingReservation::new(backend_id, first_load, Arc::clone(&reservations));
        reservations
            .lock()
            .unwrap()
            .reserve_load(backend_id, second_load);
        let second = RoutingReservation::new(backend_id, second_load, Arc::clone(&reservations));
        assert_eq!(
            reservations
                .lock()
                .unwrap()
                .backends
                .iter()
                .map(|backend| backend.reserved_load)
                .collect::<Vec<_>>(),
            vec![BackendReservedLoad {
                request_count: 2,
                prefill_token_count: 30,
                cached_prefix_token_count: 10,
                max_output_token_count: 24,
            }]
        );

        drop(first);
        assert_eq!(
            reservations
                .lock()
                .unwrap()
                .backends
                .iter()
                .map(|backend| backend.reserved_load)
                .collect::<Vec<_>>(),
            vec![second_load]
        );

        drop(second);
        assert_eq!(
            reservations
                .lock()
                .unwrap()
                .backends
                .iter()
                .map(|backend| backend.reserved_load)
                .collect::<Vec<_>>(),
            vec![BackendReservedLoad::default()]
        );
    }

    #[test]
    fn published_load_replaces_reservation_without_double_counting() {
        let (operational_sender, operational) = watch::channel(OperationalState::default());
        let (_cache_sender, cache_affinity) = watch::channel(CacheAffinityState::default());
        let orchestrator = RequestOrchestrator::new(
            vec![BackendStateReceivers {
                device: InferenceDevice::Cpu,
                operational,
                cache_affinity,
            }],
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::LeastEstimatedWork,
                throughput_profiles: vec![Default::default()],
            },
        )
        .unwrap();
        let request = RequestRoutingContext {
            input_token_ids: &[1, 2, 3],
            input_token_count: 3,
            max_new_token_count: 5,
            backend_id_override: None,
        };
        let (_, first_reservation) = orchestrator.route_request(request);
        let (before, probe_reservation) = orchestrator.route_request(request);
        let before = &before.candidate_diagnostics[0];
        assert_eq!(before.estimated_prefill_token_count, Some(6));
        assert_eq!(before.estimated_decode_token_count, Some(10));
        drop(probe_reservation);

        let published_state = OperationalState {
            epoch: 1,
            queued_request_count: 1,
            queued_remaining_prefill_token_count: 3,
            queued_remaining_max_output_token_count: 5,
            ..Default::default()
        };
        first_reservation.release_after(|| {
            operational_sender.send_replace(published_state);
        });
        assert_eq!(
            orchestrator
                .routing_state
                .lock()
                .unwrap()
                .backends
                .iter()
                .map(|backend| backend.reserved_load)
                .collect::<Vec<_>>(),
            vec![BackendReservedLoad::default()]
        );

        let (after, _reservation) = orchestrator.route_request(request);
        let after = &after.candidate_diagnostics[0];
        assert_eq!(after.operational_state, Some(published_state));
        assert_eq!(
            after.estimated_prefill_token_count,
            before.estimated_prefill_token_count
        );
        assert_eq!(
            after.estimated_decode_token_count,
            before.estimated_decode_token_count
        );
    }

    #[test]
    fn routing_reservation_releases_load_when_dropped() {
        let backend_id = BackendId::from_index(0);
        let load = BackendReservedLoad {
            request_count: 1,
            prefill_token_count: 2,
            cached_prefix_token_count: 1,
            max_output_token_count: 4,
        };
        let reservations = Arc::new(Mutex::new(OrchestratorRoutingState::new(1)));
        reservations.lock().unwrap().reserve_load(backend_id, load);

        drop(RoutingReservation::new(
            backend_id,
            load,
            Arc::clone(&reservations),
        ));

        assert_eq!(
            reservations
                .lock()
                .unwrap()
                .backends
                .iter()
                .map(|backend| backend.reserved_load)
                .collect::<Vec<_>>(),
            vec![BackendReservedLoad::default()]
        );
    }
}
