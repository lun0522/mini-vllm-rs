use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::KvCache;
use crate::model_runner::KvCacheType;
use crate::proto::model_runner::model_runner_command;
use crate::proto::model_runner::model_runner_service_server::ModelRunnerService;
use crate::proto::model_runner::model_runner_service_server::ModelRunnerServiceServer;
use crate::proto::model_runner::CommandResult;
use crate::proto::model_runner::GenerateTextEvent;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::GetModelMetadataRequest;
use crate::proto::model_runner::GetModelMetadataResponse;
use crate::proto::model_runner::ModelRunnerCommand;
use crate::utils::rpc_shutdown::RpcShutdown;
use anyhow::Context;
use anyhow::Result;
use candle_core::Tensor;
use std::cell::RefCell;
use std::path::Path;
use std::path::PathBuf;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

mod cli;
mod inference_worker;
mod kv_cache;
mod text_generation;

pub(crate) use cli::ModelRunnerProcessArgs;
use inference_worker::InferenceRequest;
use inference_worker::ModelRunner;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_MODEL_RUNNER";
const INFERENCE_QUEUE_CAPACITY: usize = 32;
const GENERATION_EVENT_QUEUE_CAPACITY: usize = 32;

pub(super) struct ModelAndKvCache {
    pub(super) model: RefCell<LoadedModel>,
    pub(super) kv_cache: RefCell<Box<dyn KvCache>>,
}

impl ModelAndKvCache {
    fn new(model: LoadedModel, kv_cache: Box<dyn KvCache>) -> Self {
        Self {
            model: RefCell::new(model),
            kv_cache: RefCell::new(kv_cache),
        }
    }

    fn forward(&self, input: &Tensor, start_position: usize) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache = self.kv_cache.borrow_mut();
        model
            .model()
            .forward(input, start_position, kv_cache.as_mut())
    }

    fn forward_for_speculative_verification(
        &self,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache = self.kv_cache.borrow_mut();
        model
            .model()
            .forward_for_speculative_verification(input, start_position, kv_cache.as_mut())
    }

    fn truncate(&self, target_token_count: usize) -> Result<()> {
        self.kv_cache.borrow_mut().truncate(target_token_count)
    }
}

pub(crate) async fn run(args: ModelRunnerProcessArgs) -> Result<()> {
    let kv_cache_type = match args.kv_cache_type {
        KvCacheType::Contiguous => KvCacheType::Contiguous,
        KvCacheType::Paged { .. } => KvCacheType::Paged {
            per_page_token_count: args.kv_cache_page_token_count,
        },
    };
    run_server(
        &args.model_path,
        args.draft_model_path,
        args.draft_token_count,
        kv_cache_type,
        args.target_kv_cache_size_bytes,
        &args.socket_path,
    )
    .await
}

async fn run_server(
    model_path: &Path,
    draft_model_path: Option<PathBuf>,
    draft_token_count: usize,
    kv_cache_type: KvCacheType,
    target_kv_cache_size_bytes: usize,
    socket_path: &Path,
) -> Result<()> {
    // Bind only after model initialization succeeds so the socket itself is a
    // readiness signal for the parent process.
    let model_runner = ModelRunner::new(
        model_path,
        draft_model_path.as_deref(),
        draft_token_count,
        kv_cache_type,
        target_kv_cache_size_bytes,
    )?;
    let model_metadata = model_runner.model_metadata();
    let listener = UnixListener::bind(socket_path)
        .context("failed to bind the model runner Unix domain socket")?;
    let (inference_sender, inference_receiver) = mpsc::channel(INFERENCE_QUEUE_CAPACITY);
    let inference_thread = std::thread::Builder::new()
        .name("inference-worker".to_owned())
        .spawn(move || inference_worker::run(model_runner, inference_receiver))
        .context("failed to start the model runner inference thread")?;
    let (shutdown, shutdown_receiver) = RpcShutdown::channel();
    let service = ModelRunnerRpcService {
        inference_sender,
        model_metadata,
        shutdown,
    };

    let server_result = tonic::transport::Server::builder()
        .add_service(ModelRunnerServiceServer::new(service))
        .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
            let _ = shutdown_receiver.await;
        })
        .await
        .context("model runner RPC server failed");
    inference_thread
        .join()
        .map_err(|_| anyhow::anyhow!("model runner inference thread panicked"))?;
    server_result
}

struct ModelRunnerRpcService {
    inference_sender: mpsc::Sender<InferenceRequest>,
    model_metadata: GetModelMetadataResponse,
    shutdown: RpcShutdown,
}

#[tonic::async_trait]
impl ModelRunnerService for ModelRunnerRpcService {
    type GenerateTextStream = ReceiverStream<Result<GenerateTextEvent, Status>>;

    async fn get_model_metadata(
        &self,
        _request: Request<GetModelMetadataRequest>,
    ) -> Result<Response<GetModelMetadataResponse>, Status> {
        Ok(Response::new(self.model_metadata))
    }

    async fn generate_text(
        &self,
        request: Request<GenerateTextRequest>,
    ) -> Result<Response<Self::GenerateTextStream>, Status> {
        let (event_sender, event_receiver) = mpsc::channel(GENERATION_EVENT_QUEUE_CAPACITY);
        self.inference_sender
            .send(InferenceRequest {
                generate_text: request.into_inner(),
                event_sender,
            })
            .await
            .map_err(|_| Status::unavailable("model runner inference thread stopped"))?;
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
