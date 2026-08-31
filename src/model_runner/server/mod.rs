use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::model_loaders::ModelRole;
use crate::model_runner::KvCacheType;
use crate::proto::model_runner_command;
use crate::proto::model_runner_service_server::ModelRunnerService;
use crate::proto::model_runner_service_server::ModelRunnerServiceServer;
use crate::proto::CommandResult;
use crate::proto::GenerateText;
use crate::proto::GenerateTextEvent;
use crate::proto::ModelRunnerCommand;
use crate::utils::rpc_shutdown::RpcShutdown;
use anyhow::Context;
use anyhow::Result;
use std::path::Path;
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
mod tokenizer;

use cli::create_model_artifacts;
pub(crate) use cli::ModelRunnerProcessArgs;
use inference_worker::InferenceRequest;
use inference_worker::ModelRunner;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_MODEL_RUNNER";
const INFERENCE_QUEUE_CAPACITY: usize = 32;
const GENERATION_EVENT_QUEUE_CAPACITY: usize = 32;

pub(crate) async fn run(args: ModelRunnerProcessArgs) -> Result<()> {
    let model_artifacts = create_model_artifacts(args.model, ModelRole::Target)?;
    let draft_model_artifacts = args
        .draft_model
        .map(|paths| create_model_artifacts(paths, ModelRole::Draft))
        .transpose()?;
    run_server(
        &model_artifacts,
        draft_model_artifacts,
        args.draft_token_count,
        args.kv_cache_type,
        &args.socket_path,
    )
    .await
}

async fn run_server(
    model_artifacts: &ModelArtifacts,
    draft_model_artifacts: Option<ModelArtifacts>,
    draft_token_count: usize,
    kv_cache_type: KvCacheType,
    socket_path: &Path,
) -> Result<()> {
    // Bind only after model initialization succeeds so the socket itself is a
    // readiness signal for the parent process.
    let model_runner = ModelRunner::new(
        model_artifacts,
        draft_model_artifacts.as_ref(),
        draft_token_count,
        kv_cache_type,
    )?;
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
    shutdown: RpcShutdown,
}

#[tonic::async_trait]
impl ModelRunnerService for ModelRunnerRpcService {
    type GenerateTextStream = ReceiverStream<Result<GenerateTextEvent, Status>>;

    async fn generate_text(
        &self,
        request: Request<GenerateText>,
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
