use crate::model_loaders::model_downloader::ModelFiles;
use crate::model_runner::ModelRunner;
use crate::proto::model_runner_command;
use crate::proto::model_runner_service_server::ModelRunnerService;
use crate::proto::model_runner_service_server::ModelRunnerServiceServer;
use crate::proto::CommandResult;
use crate::proto::ModelRunnerCommand;
use anyhow::Context;
use anyhow::Result;
use argh::FromArgs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use tokio::net::UnixListener;
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_MODEL_RUNNER";

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
    let listener = UnixListener::bind(socket_path)
        .context("failed to bind the model runner Unix domain socket")?;
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let service = ModelRunnerRpcService {
        model_runner: Arc::new(Mutex::new(ModelRunner::new(model_files)?)),
        shutdown_sender: Mutex::new(Some(shutdown_sender)),
    };

    tonic::transport::Server::builder()
        .add_service(ModelRunnerServiceServer::new(service))
        .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
            let _ = shutdown_receiver.await;
        })
        .await
        .context("model runner RPC server failed")
}

struct ModelRunnerRpcService {
    model_runner: Arc<Mutex<ModelRunner>>,
    shutdown_sender: Mutex<Option<oneshot::Sender<()>>>,
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

        let model_runner = Arc::clone(&self.model_runner);
        tokio::task::spawn_blocking(move || {
            model_runner
                .lock()
                .map_err(|_| anyhow::anyhow!("model runner lock is poisoned"))?
                .generate_text(&generate_text)
        })
        .await
        .map_err(|error| Status::internal(format!("model runner task failed: {error}")))?
        .map_err(|error| Status::internal(format!("model runner command failed: {error:#}")))?;

        Ok(Response::new(CommandResult {}))
    }
}
