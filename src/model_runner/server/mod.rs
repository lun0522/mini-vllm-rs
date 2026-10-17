use crate::model_runner::ActivationDType;
use crate::model_runner::InferenceDevice;
use crate::model_runner::KvCacheConfig;
use crate::model_runner::RequestOrchestratorConfig;
use crate::model_runner::SchedulerConfig;
use crate::proto::inference_config::DraftModelRunnerConfig as DraftModelRunnerConfigProto;
use crate::proto::inference_config::DraftTokenCountPolicy as DraftTokenCountPolicyProto;
use crate::proto::model_runner::model_runner_command;
use crate::proto::model_runner::model_runner_service_server::ModelRunnerService;
use crate::proto::model_runner::model_runner_service_server::ModelRunnerServiceServer;
use crate::proto::model_runner::CommandResult;
use crate::proto::model_runner::GenerateTextEvent;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::GetModelMetadataRequest;
use crate::proto::model_runner::GetModelMetadataResponse;
use crate::proto::model_runner::GetPrefixCacheTelemetryRequest;
use crate::proto::model_runner::GetPrefixCacheTelemetryResponse;
use crate::proto::model_runner::InferenceBackendPrefixCacheTelemetry;
use crate::proto::model_runner::ModelRunnerCommand;
use crate::utils::rpc_shutdown::RpcShutdown;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use log::info;
use log::warn;
use std::thread::JoinHandle;
use std::time::Instant;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

mod cli;
mod draft_token_count;
mod generation_stats;
mod inference_worker;
mod kv_cache;
mod model_executor;
mod model_instance;
mod prefix_cache_flight;
mod request_manager;
mod request_orchestrator;
mod routing_policy;
mod scheduler;
mod text_generation;

pub(crate) use cli::ModelRunnerProcessArgs;
use draft_token_count::DraftTokenCountPolicy;
use inference_worker::InferenceWorker;
use model_executor::DraftModelConfig;
use model_executor::ModelExecutor;
use model_executor::ModelExecutorMetadata;
use request_orchestrator::BackendStateReceivers;
use request_orchestrator::CacheAffinityState;
use request_orchestrator::OperationalState;
use request_orchestrator::RequestOrchestrator;
use request_orchestrator::RequestRoutingContext;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_MODEL_RUNNER";
const INFERENCE_MESSAGE_QUEUE_CAPACITY: usize = 32;
const GENERATION_EVENT_QUEUE_CAPACITY: usize = 32;

struct InferenceBackend {
    metadata: ModelExecutorMetadata,
    thread: JoinHandle<()>,
    channels: InferenceBackendChannels,
}

struct InferenceBackendChannels {
    message_sender: mpsc::Sender<InferenceWorkerMessage>,
    state_receivers: BackendStateReceivers,
}

struct InferenceBackends {
    metadata: ModelExecutorMetadata,
    threads: Vec<JoinHandle<()>>,
    channels: Vec<InferenceBackendChannels>,
}

pub(super) enum InferenceWorkerMessage {
    Generate(InferenceRequest),
    GetPrefixCacheTelemetry {
        response_sender: oneshot::Sender<InferenceBackendPrefixCacheTelemetry>,
    },
}

pub(super) struct InferenceRequest {
    queued_at: Instant,
    generate_text: GenerateTextRequest,
    event_sender: mpsc::Sender<Result<GenerateTextEvent, Status>>,
}

pub(crate) async fn run(args: ModelRunnerProcessArgs) -> Result<()> {
    run_server(args).await
}

async fn run_server(args: ModelRunnerProcessArgs) -> Result<()> {
    args.request_orchestrator_config.validate()?;
    let inference_devices = match args.inference_device {
        InferenceDevice::Mixed => vec![InferenceDevice::Cpu, InferenceDevice::Gpu],
        specific_device => vec![specific_device],
    };
    let inference_backends: InferenceBackends =
        create_inference_backends(&args, &inference_devices)?;

    // Bind only after model initialization succeeds so the socket itself is a
    // readiness signal for the parent process.
    let listener = UnixListener::bind(&args.socket_path)
        .context("failed to bind the model runner Unix domain socket")?;
    let (shutdown, shutdown_receiver) = RpcShutdown::channel();
    let service = ModelRunnerRpcService::new(
        inference_backends.metadata,
        inference_backends.channels,
        RequestOrchestratorConfig::from(args.request_orchestrator_config),
        shutdown,
    )?;
    let server_result = tonic::transport::Server::builder()
        .add_service(ModelRunnerServiceServer::new(service))
        .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
            let _ = shutdown_receiver.await;
        })
        .await
        .context("model runner RPC server failed");

    for inference_thread in inference_backends.threads.into_iter() {
        inference_thread
            .join()
            .map_err(|_| anyhow::anyhow!("model runner inference thread panicked"))?;
    }
    server_result
}

struct ModelRunnerRpcService {
    metadata: ModelExecutorMetadata,
    inference_message_senders: Vec<mpsc::Sender<InferenceWorkerMessage>>,
    request_orchestrator: RequestOrchestrator,
    shutdown: RpcShutdown,
}

impl ModelRunnerRpcService {
    pub fn new(
        metadata: ModelExecutorMetadata,
        inference_backend_channels: Vec<InferenceBackendChannels>,
        request_orchestrator_config: RequestOrchestratorConfig,
        shutdown: RpcShutdown,
    ) -> Result<Self> {
        let mut inference_message_senders = Vec::with_capacity(inference_backend_channels.len());
        let mut backend_state_receivers = Vec::with_capacity(inference_backend_channels.len());
        for backend_channels in inference_backend_channels {
            inference_message_senders.push(backend_channels.message_sender);
            backend_state_receivers.push(backend_channels.state_receivers);
        }
        let request_orchestrator =
            RequestOrchestrator::new(backend_state_receivers, request_orchestrator_config)?;
        Ok(Self {
            metadata,
            inference_message_senders,
            request_orchestrator,
            shutdown,
        })
    }
}

#[tonic::async_trait]
impl ModelRunnerService for ModelRunnerRpcService {
    type GenerateTextStream = ReceiverStream<Result<GenerateTextEvent, Status>>;

    async fn get_model_metadata(
        &self,
        _request: Request<GetModelMetadataRequest>,
    ) -> Result<Response<GetModelMetadataResponse>, Status> {
        Ok(Response::new(self.metadata.model_metadata))
    }

    async fn get_prefix_cache_telemetry(
        &self,
        _request: Request<GetPrefixCacheTelemetryRequest>,
    ) -> Result<Response<GetPrefixCacheTelemetryResponse>, Status> {
        let mut backends = Vec::with_capacity(self.inference_message_senders.len());
        for message_sender in &self.inference_message_senders {
            let (response_sender, response_receiver) = oneshot::channel();
            message_sender
                .send(InferenceWorkerMessage::GetPrefixCacheTelemetry { response_sender })
                .await
                .map_err(|_| Status::unavailable("model runner inference thread stopped"))?;
            backends.push(
                response_receiver
                    .await
                    .map_err(|_| Status::unavailable("model runner inference thread stopped"))?,
            );
        }
        Ok(Response::new(GetPrefixCacheTelemetryResponse { backends }))
    }

    async fn generate_text(
        &self,
        request: Request<GenerateTextRequest>,
    ) -> Result<Response<Self::GenerateTextStream>, Status> {
        let mut request = request.into_inner();
        let model_context_length = get_effective_context_length(&self.metadata.model_metadata)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
        normalize_generate_text_request(
            &mut request,
            self.metadata.kv_cache_token_capacity,
            model_context_length,
        )
        .map_err(|error| Status::invalid_argument(error.to_string()))?;
        let request_id = request.request_id;
        let queued_at = Instant::now();
        let routing_decision = self
            .request_orchestrator
            .route_request(RequestRoutingContext {
                input_token_ids: &request.input_token_ids,
                input_token_count: request.input_token_ids.len(),
                max_new_token_count: request.max_new_tokens,
            });
        let (event_sender, event_receiver) = mpsc::channel(GENERATION_EVENT_QUEUE_CAPACITY);
        self.inference_message_senders[routing_decision.backend_index]
            .send(InferenceWorkerMessage::Generate(InferenceRequest {
                queued_at,
                generate_text: request,
                event_sender,
            }))
            .await
            .map_err(|_| Status::unavailable("model runner inference thread stopped"))?;
        info!("Model runner queue state: request_id={request_id} status=queued");
        Ok(Response::new(ReceiverStream::new(event_receiver)))
    }

    async fn handle_command(
        &self,
        request: Request<ModelRunnerCommand>,
    ) -> Result<Response<CommandResult>, Status> {
        let command = request
            .into_inner()
            .command
            .ok_or_else(|| Status::invalid_argument("model runner command is empty"))?;
        match command {
            model_runner_command::Command::Shutdown(_) => self.shutdown.trigger()?,
        }
        Ok(Response::new(CommandResult {}))
    }
}

fn create_inference_backends(
    args: &ModelRunnerProcessArgs,
    inference_devices: &[InferenceDevice],
) -> Result<InferenceBackends> {
    let mut maybe_metadata: Option<ModelExecutorMetadata> = None;
    let mut threads = Vec::with_capacity(inference_devices.len());
    let mut channels = Vec::with_capacity(inference_devices.len());
    for (index, device) in inference_devices.iter().enumerate() {
        let backend_id = index + 1;
        let inference_backend: InferenceBackend =
            create_inference_backend(backend_id, args, *device)?;
        if let Some(metadata) = maybe_metadata.as_mut() {
            // TODO: Track capacity per inference backend so mixed mode can use the CPU backend's
            // additional F16 cache capacity instead of advertising only the shared minimum.
            metadata.kv_cache_token_capacity = metadata
                .kv_cache_token_capacity
                .min(inference_backend.metadata.kv_cache_token_capacity);
        } else {
            maybe_metadata = Some(inference_backend.metadata);
        }
        threads.push(inference_backend.thread);
        channels.push(inference_backend.channels);
    }
    let metadata = maybe_metadata.expect("at least one model runner should have been created");
    Ok(InferenceBackends {
        metadata,
        threads,
        channels,
    })
}

fn create_inference_backend(
    backend_id: usize,
    args: &ModelRunnerProcessArgs,
    inference_device: InferenceDevice,
) -> Result<InferenceBackend> {
    let activation_dtype = normalize_activation_dtype(args.activation_dtype, inference_device);
    let draft_model_config = args
        .draft_model_runner_config
        .as_ref()
        .map(create_draft_model_config)
        .transpose()?;
    let model_executor = ModelExecutor::new(
        &args.model_path,
        draft_model_config,
        inference_device,
        activation_dtype,
        KvCacheConfig::from(args.kv_cache_config),
    )?;
    let metadata = model_executor.metadata();
    let (message_sender, message_receiver) = mpsc::channel(INFERENCE_MESSAGE_QUEUE_CAPACITY);
    let (operational_state_sender, operational_state_receiver) =
        watch::channel(OperationalState::default());
    let (cache_affinity_state_sender, cache_affinity_state_receiver) =
        watch::channel(CacheAffinityState::default());
    let scheduler_config = SchedulerConfig::from(args.scheduler_config);
    let thread = std::thread::Builder::new()
        .name(format!("inference-worker-{backend_id}"))
        .spawn(move || {
            InferenceWorker::new(
                backend_id,
                model_executor,
                scheduler_config,
                operational_state_sender,
                cache_affinity_state_sender,
            )
            .run(message_receiver);
        })
        .context(format!(
            "failed to start the model runner inference backend {backend_id}"
        ))?;
    Ok(InferenceBackend {
        metadata,
        thread,
        channels: InferenceBackendChannels {
            message_sender,
            state_receivers: BackendStateReceivers {
                device: inference_device,
                operational: operational_state_receiver,
                cache_affinity: cache_affinity_state_receiver,
            },
        },
    })
}

fn create_draft_model_config(config: &DraftModelRunnerConfigProto) -> Result<DraftModelConfig> {
    config.validate()?;
    let token_count_policy_proto: &DraftTokenCountPolicyProto = config
        .token_count_policy
        .as_ref()
        .expect("validated draft model runner configuration should contain a token-count policy");
    Ok(DraftModelConfig {
        model_path: config.model_path.clone().into(),
        token_count_policy: DraftTokenCountPolicy::try_from(token_count_policy_proto)?,
    })
}

fn get_effective_context_length(model_metadata: &GetModelMetadataResponse) -> Result<usize> {
    let target = model_metadata
        .target_model
        .as_ref()
        .context("target model context metadata is missing")?;
    let mut context_length = target.context_length;
    if let Some(draft) = &model_metadata.draft_model {
        context_length = context_length.min(draft.context_length);
    }
    usize::try_from(context_length).context("effective model context length does not fit in usize")
}

fn normalize_generate_text_request(
    request: &mut GenerateTextRequest,
    kv_cache_token_capacity: usize,
    model_context_length: usize,
) -> Result<()> {
    ensure!(
        !request.input_token_ids.is_empty(),
        "input token IDs must not be empty"
    );
    let input_token_count = request.input_token_ids.len();
    ensure!(
        input_token_count <= kv_cache_token_capacity,
        "request input contains {input_token_count} tokens but the configured KV-cache \
         capacity is {kv_cache_token_capacity}"
    );
    ensure!(
        input_token_count < model_context_length,
        "request input contains {input_token_count} tokens but the effective model context limit \
         is {model_context_length}; at least one output token must fit within the context"
    );
    ensure!(
        request.max_new_tokens > 0,
        "max_new_tokens must be greater than zero"
    );

    // The final generated token remains pending rather than being written to the cache, so one
    // output token can be generated even when the input already fills the cache.
    let maximum_new_token_count = (kv_cache_token_capacity - input_token_count)
        .checked_add(1)
        .context("maximum new token count does not fit in usize")?
        .min(model_context_length - input_token_count);
    let maximum_new_token_count = u64::try_from(maximum_new_token_count)
        .context("maximum new token count does not fit in u64")?;
    if request.max_new_tokens > maximum_new_token_count {
        warn!(
            "Requested {} new tokens for {input_token_count} input tokens with KV-cache capacity \
             {kv_cache_token_capacity} and effective context limit {model_context_length}; \
             reducing max_new_tokens to {}",
            request.max_new_tokens, maximum_new_token_count,
        );
        request.max_new_tokens = maximum_new_token_count;
    }
    Ok(())
}

fn normalize_activation_dtype(
    activation_dtype: ActivationDType,
    inference_device: InferenceDevice,
) -> ActivationDType {
    if cfg!(target_os = "macos")
        && inference_device == InferenceDevice::Gpu
        && activation_dtype == ActivationDType::F16
    {
        warn!(
            "Candle's Metal quantized matmul does not support F16 activations; using F32 instead"
        );
        ActivationDType::F32
    } else {
        activation_dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::RequestOrchestratorConfig;
    use crate::model_runner::RequestRoutingPolicy;
    use crate::proto::model_runner::model_runner_service_server::ModelRunnerService;
    use crate::proto::model_runner::GenerateTextRequest;
    use crate::proto::model_runner::GetModelMetadataResponse;
    use crate::proto::model_runner::GetPrefixCacheTelemetryRequest;
    use crate::proto::model_runner::InferenceBackendPrefixCacheTelemetry;
    use crate::proto::model_runner::ModelMetadata;
    use crate::proto::model_runner::PrefixCacheTelemetry;
    use crate::utils::rpc_shutdown::RpcShutdown;
    use tokio::sync::mpsc;
    use tokio::sync::watch;
    use tonic::Request;

    fn metadata(
        context_length: u64,
        draft_context_length: Option<u64>,
    ) -> GetModelMetadataResponse {
        GetModelMetadataResponse {
            target_model: Some(ModelMetadata {
                context_length,
                ..Default::default()
            }),
            draft_model: draft_context_length.map(|context_length| ModelMetadata {
                context_length,
                ..Default::default()
            }),
        }
    }

    fn backend_handle(
        message_sender: mpsc::Sender<InferenceWorkerMessage>,
    ) -> InferenceBackendChannels {
        let (_operational_sender, operational) = watch::channel(OperationalState::default());
        let (_cache_affinity_sender, cache_affinity) =
            watch::channel(CacheAffinityState::default());
        InferenceBackendChannels {
            message_sender,
            state_receivers: BackendStateReceivers {
                device: InferenceDevice::Cpu,
                operational,
                cache_affinity,
            },
        }
    }

    #[tokio::test]
    async fn queries_prefix_cache_telemetry_from_each_inference_backend() {
        let mut inference_backends = Vec::new();
        for (backend_id, target_token_capacity, draft_token_capacity) in
            [(1, 128, None), (2, 256, Some(64))]
        {
            let (message_sender, mut message_receiver) = mpsc::channel(1);
            inference_backends.push(backend_handle(message_sender));
            tokio::spawn(async move {
                let Some(InferenceWorkerMessage::GetPrefixCacheTelemetry { response_sender }) =
                    message_receiver.recv().await
                else {
                    panic!("expected a prefix-cache telemetry query");
                };
                let _ = response_sender.send(InferenceBackendPrefixCacheTelemetry {
                    backend_id,
                    target: Some(PrefixCacheTelemetry {
                        token_capacity: target_token_capacity,
                        ..Default::default()
                    }),
                    draft: draft_token_capacity.map(|token_capacity| PrefixCacheTelemetry {
                        token_capacity,
                        ..Default::default()
                    }),
                });
            });
        }
        let (shutdown, _shutdown_receiver) = RpcShutdown::channel();
        let service = ModelRunnerRpcService::new(
            ModelExecutorMetadata {
                model_metadata: metadata(32, None),
                kv_cache_token_capacity: 128,
            },
            inference_backends,
            RequestOrchestratorConfig {
                routing_policy: RequestRoutingPolicy::RoundRobin,
            },
            shutdown,
        )
        .unwrap();

        let response = service
            .get_prefix_cache_telemetry(Request::new(GetPrefixCacheTelemetryRequest {}))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(response.backends.len(), 2);
        assert_eq!(response.backends[0].backend_id, 1);
        assert_eq!(
            response.backends[0].target.as_ref().unwrap().token_capacity,
            128
        );
        assert!(response.backends[0].draft.is_none());
        assert_eq!(response.backends[1].backend_id, 2);
        assert_eq!(
            response.backends[1].target.as_ref().unwrap().token_capacity,
            256
        );
        assert_eq!(response.backends[1].draft.unwrap().token_capacity, 64);
    }

    #[test]
    fn limits_generation_to_the_available_cache_capacity() {
        let mut request = GenerateTextRequest {
            input_token_ids: vec![1, 2, 3],
            max_new_tokens: 3,
            ..Default::default()
        };

        normalize_generate_text_request(&mut request, 4, 32).unwrap();

        assert_eq!(request.max_new_tokens, 2);
    }

    #[test]
    fn rejects_empty_model_input() {
        let mut request = GenerateTextRequest::default();

        let error = normalize_generate_text_request(&mut request, 16, 32).unwrap_err();

        assert_eq!(error.to_string(), "input token IDs must not be empty");
    }

    #[test]
    fn rejects_zero_maximum_new_token_count() {
        let mut request = GenerateTextRequest {
            input_token_ids: vec![1],
            max_new_tokens: 0,
            ..Default::default()
        };

        let error = normalize_generate_text_request(&mut request, 16, 32).unwrap_err();

        assert_eq!(
            error.to_string(),
            "max_new_tokens must be greater than zero"
        );
    }

    #[test]
    fn rejects_input_that_exceeds_cache_capacity() {
        let mut request = GenerateTextRequest {
            input_token_ids: vec![1; 17],
            max_new_tokens: 0,
            ..Default::default()
        };

        let error = normalize_generate_text_request(&mut request, 16, 32).unwrap_err();

        assert!(error.to_string().contains("input contains 17 tokens"));
    }

    #[test]
    fn allows_one_output_token_when_input_fills_cache() {
        let mut request = GenerateTextRequest {
            input_token_ids: vec![1; 16],
            max_new_tokens: 8,
            ..Default::default()
        };

        normalize_generate_text_request(&mut request, 16, 32).unwrap();

        assert_eq!(request.max_new_tokens, 1);
    }

    #[test]
    fn limits_complete_sequences_to_target_and_draft_contexts() {
        for (target, draft, requested, expected) in [
            (5, None, u64::MAX, 2),
            (32, Some(4), 8, 1),
            (5, None, 2, 2),
            (5, None, 1, 1),
        ] {
            let mut request = GenerateTextRequest {
                input_token_ids: vec![1; 3],
                max_new_tokens: requested,
                ..Default::default()
            };
            normalize_generate_text_request(
                &mut request,
                16,
                get_effective_context_length(&metadata(target, draft)).unwrap(),
            )
            .unwrap();
            assert_eq!(request.max_new_tokens, expected);
        }
    }

    #[test]
    fn rejects_prompts_without_context_room_for_output() {
        for (target, draft, count) in [(3, None, 3), (3, None, 4), (32, Some(3), 3)] {
            let mut request = GenerateTextRequest {
                input_token_ids: vec![1; count],
                max_new_tokens: 1,
                ..Default::default()
            };
            let error = normalize_generate_text_request(
                &mut request,
                16,
                get_effective_context_length(&metadata(target, draft)).unwrap(),
            )
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("effective model context limit is 3"));
        }
    }
}
