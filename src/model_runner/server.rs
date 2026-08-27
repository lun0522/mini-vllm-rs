use crate::model_loaders::model_downloader::ModelFiles;
use crate::model_runner::ModelRunner;
use crate::proto::model_runner_command;
use crate::proto::model_runner_service_server::ModelRunnerService;
use crate::proto::model_runner_service_server::ModelRunnerServiceServer;
use crate::proto::CommandResult;
use crate::proto::GenerateText;
use crate::proto::ModelRunnerCommand;
use anyhow::Context;
use anyhow::Result;
use argh::FromArgs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_MODEL_RUNNER";
const INFERENCE_QUEUE_CAPACITY: usize = 32;

/// Runs model inference using files already available on disk.
#[derive(FromArgs)]
pub(crate) struct ModelRunnerProcessArgs {
    /// local tokenizer path used by the model runner worker
    #[argh(option)]
    tokenizer_path: PathBuf,
    /// local model configuration path used by the model runner worker
    #[argh(option)]
    config_path: PathBuf,
    /// local model weights path used by the model runner worker
    #[argh(option)]
    weights_path: PathBuf,
    /// unix domain socket path used by the model runner worker
    #[argh(option)]
    socket_path: PathBuf,
}

pub(crate) async fn run(args: ModelRunnerProcessArgs) -> Result<()> {
    let model_files = ModelFiles {
        tokenizer: args.tokenizer_path,
        config: args.config_path,
        weights: args.weights_path,
    };
    run_server(&model_files, &args.socket_path).await
}

async fn run_server(model_files: &ModelFiles, socket_path: &Path) -> Result<()> {
    // Bind only after model initialization succeeds so the socket itself is a
    // readiness signal for the parent process.
    let model_runner = ModelRunner::new(model_files)?;
    let listener = UnixListener::bind(socket_path)
        .context("failed to bind the model runner Unix domain socket")?;
    let (inference_sender, inference_receiver) = mpsc::channel(INFERENCE_QUEUE_CAPACITY);
    let inference_thread = std::thread::Builder::new()
        .name("inference-worker".to_owned())
        .spawn(move || run_inference_loop(model_runner, inference_receiver))
        .context("failed to start the model runner inference thread")?;
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let service = ModelRunnerRpcService {
        inference_sender,
        shutdown_sender: Mutex::new(Some(shutdown_sender)),
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
    shutdown_sender: Mutex<Option<oneshot::Sender<()>>>,
}

struct InferenceRequest {
    generate_text: GenerateText,
    result_sender: oneshot::Sender<Result<()>>,
}

fn run_inference_loop(
    mut model_runner: ModelRunner,
    mut inference_receiver: mpsc::Receiver<InferenceRequest>,
) {
    // A single thread owns one model today. This queue can later feed a batching
    // scheduler or route requests to multiple device-specific inference workers.
    while let Some(request) = inference_receiver.blocking_recv() {
        let result = model_runner.generate_text(&request.generate_text);
        let _ = request.result_sender.send(result);
    }
}

#[tonic::async_trait]
impl ModelRunnerService for ModelRunnerRpcService {
    async fn handle_command(
        &self,
        request: Request<ModelRunnerCommand>,
    ) -> Result<Response<CommandResult>, Status> {
        let command = request
            .into_inner()
            .command
            .ok_or_else(|| Status::invalid_argument("model runner command is empty"))?;
        let generate_text = match command {
            model_runner_command::Command::GenerateText(generate_text) => generate_text,
            model_runner_command::Command::Shutdown(_) => {
                let shutdown_sender = self
                    .shutdown_sender
                    .lock()
                    .map_err(|_| Status::internal("shutdown sender lock is poisoned"))?
                    .take()
                    .ok_or_else(|| Status::failed_precondition("shutdown was already requested"))?;
                shutdown_sender
                    .send(())
                    .map_err(|_| Status::internal("model runner shutdown receiver was dropped"))?;
                return Ok(Response::new(CommandResult {}));
            }
        };

        let (result_sender, result_receiver) = oneshot::channel();
        self.inference_sender
            .send(InferenceRequest {
                generate_text,
                result_sender,
            })
            .await
            .map_err(|_| Status::unavailable("model runner inference thread stopped"))?;
        result_receiver
            .await
            .map_err(|_| Status::internal("model runner inference response was dropped"))?
            .map_err(|error| Status::internal(format!("model runner command failed: {error:#}")))?;

        Ok(Response::new(CommandResult {}))
    }
}
