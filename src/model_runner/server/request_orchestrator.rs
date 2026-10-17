use crate::model_runner::InferenceDevice;
use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::RequestRoutingPolicy;
use anyhow::ensure;
use anyhow::Result;
use log::info;
use tokio::sync::watch;

use super::routing_policy::BackendRoutingPolicy;
use super::routing_policy::LeastEstimatedWorkPolicy;
use super::routing_policy::LeastRequestsPolicy;
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CacheAffinityState {
    pub(super) indexed_token_count: usize,
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
pub(super) struct RequestRoutingContext {
    pub(super) input_token_count: usize,
    pub(super) max_new_token_count: u64,
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
        };
        Ok(Self { policy })
    }

    /// Selects a backend according to the configured routing policy.
    pub(super) fn route_request(&self, request: RequestRoutingContext) -> RoutingDecision {
        self.policy.route_request(request)
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

    fn round_robin_config() -> RequestOrchestratorConfig {
        RequestOrchestratorConfig {
            routing_policy: RequestRoutingPolicy::RoundRobin,
        }
    }

    fn routing_context(estimated_token_count: u64) -> RequestRoutingContext {
        RequestRoutingContext {
            input_token_count: 0,
            max_new_token_count: estimated_token_count,
        }
    }

    #[test]
    fn routes_requests_in_round_robin_order() {
        let orchestrator =
            RequestOrchestrator::new(backend_state_receivers(2), round_robin_config()).unwrap();

        let backend_indices: Vec<_> = (0..5)
            .map(|_| orchestrator.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn routes_every_request_to_the_only_backend() {
        let orchestrator =
            RequestOrchestrator::new(backend_state_receivers(1), round_robin_config()).unwrap();

        for _ in 0..3 {
            assert_eq!(
                orchestrator.route_request(routing_context(1)).backend_index,
                0
            );
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
                device: InferenceDevice::Cpu,
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
            ..Default::default()
        });
        operational_sender.send_replace(OperationalState {
            queued_request_count: 1,
            active_request_count: 1,
            ..Default::default()
        });

        assert_eq!(
            orchestrator.route_request(routing_context(1)).backend_index,
            0
        );
    }

    #[test]
    fn routes_to_the_backend_with_the_fewest_requests() {
        let orchestrator = RequestOrchestrator::new(
            vec![
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
            ],
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::LeastRequests,
            },
        )
        .unwrap();

        assert_eq!(
            orchestrator.route_request(routing_context(1)).backend_index,
            1
        );
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
            .map(|_| orchestrator.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1]);
    }

    #[test]
    fn least_estimated_work_accounts_for_backend_throughput() {
        let orchestrator = RequestOrchestrator::new(
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
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::LeastEstimatedWork,
            },
        )
        .unwrap();

        // CPU: 200 / 9.22; GPU: 250 / 16.25, so the busier GPU still has less work.
        assert_eq!(
            orchestrator
                .route_request(routing_context(100))
                .backend_index,
            1
        );
    }

    #[test]
    fn least_estimated_work_breaks_equal_scores_by_least_recent_selection() {
        let orchestrator = RequestOrchestrator::new(
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
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::LeastEstimatedWork,
            },
        )
        .unwrap();

        let backend_indices: Vec<_> = (0..4)
            .map(|_| orchestrator.route_request(routing_context(1)).backend_index)
            .collect();

        assert_eq!(backend_indices, vec![0, 1, 0, 1]);
    }
}
