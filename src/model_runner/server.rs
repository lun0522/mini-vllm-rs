use crate::model_loaders::model_downloader::ModelFiles;
use crate::model_runner::ModelRunner;
use crate::proto::generate_text_event;
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
use argh::FromArgs;
use std::path::Path;
use std::path::PathBuf;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_MODEL_RUNNER";
const INFERENCE_QUEUE_CAPACITY: usize = 32;
const GENERATION_EVENT_QUEUE_CAPACITY: usize = 32;

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

struct InferenceRequest {
    generate_text: GenerateText,
    event_sender: mpsc::Sender<Result<GenerateTextEvent, Status>>,
}

fn run_inference_loop(
    mut model_runner: ModelRunner,
    mut inference_receiver: mpsc::Receiver<InferenceRequest>,
) {
    // A single thread owns one model today. This queue can later feed a batching
    // scheduler or route requests to multiple device-specific inference workers.
    while let Some(request) = inference_receiver.blocking_recv() {
        let mut buffered_text = String::new();
        let stream_output = request.generate_text.stream_output;
        let result = model_runner.generate_text(
            &request.generate_text,
            |fragment| {
                if stream_output {
                    send_event(
                        &request.event_sender,
                        generate_text_event::Event::Text(fragment.to_owned()),
                    )
                } else {
                    buffered_text.push_str(fragment);
                    Ok(())
                }
            },
            || request.event_sender.is_closed(),
        );

        match result {
            Ok(stats) => {
                if !stream_output
                    && send_event(
                        &request.event_sender,
                        generate_text_event::Event::Text(buffered_text),
                    )
                    .is_err()
                {
                    continue;
                }
                let _ = send_event(
                    &request.event_sender,
                    generate_text_event::Event::Stats(stats),
                );
            }
            Err(error) => {
                let _ = request
                    .event_sender
                    .blocking_send(Err(Status::internal(format!(
                        "model runner generation failed: {error:#}"
                    ))));
            }
        }
    }
}

fn send_event(
    event_sender: &mpsc::Sender<Result<GenerateTextEvent, Status>>,
    event: generate_text_event::Event,
) -> Result<()> {
    event_sender
        .blocking_send(Ok(GenerateTextEvent { event: Some(event) }))
        .context("generation response stream was dropped")
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
