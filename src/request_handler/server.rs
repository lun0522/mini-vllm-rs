use crate::proto::model_runner::generate_text_event as model_runner_generate_text_event;
use crate::proto::model_runner::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::model_runner::GenerateTextEvent as ModelRunnerGenerateTextEvent;
use crate::proto::model_runner::GetModelMetadataRequest;
use crate::proto::request_handler::generate_text_event;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerService;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerServiceServer;
use crate::proto::request_handler::CommandResult;
use crate::proto::request_handler::GenerateText;
use crate::proto::request_handler::GenerateTextEvent;
use crate::proto::request_handler::Shutdown;
use crate::utils::domain_socket;
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
use tonic::transport::Channel;
use tonic::Request;
use tonic::Response;
use tonic::Status;

use super::tokenizer::IncrementalTokenDecoder;
use super::tokenizer::TokenizerWrapper;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_REQUEST_HANDLER";
const GENERATION_EVENT_QUEUE_CAPACITY: usize = 32;

/// Handles local inference requests and forwards them to the model runner.
#[derive(FromArgs)]
pub(crate) struct RequestHandlerProcessArgs {
    /// model runner Unix domain socket path
    #[argh(option)]
    model_runner_socket_path: PathBuf,
    /// target model tokenizer path
    #[argh(option)]
    tokenizer_path: PathBuf,
    /// draft model tokenizer path
    #[argh(option)]
    draft_tokenizer_path: Option<PathBuf>,
    /// request handler Unix domain socket path
    #[argh(option)]
    request_handler_socket_path: PathBuf,
}

pub(crate) async fn run(args: RequestHandlerProcessArgs) -> Result<()> {
    let mut model_runner_client = connect_to_model_runner(&args.model_runner_socket_path).await?;
    let model_metadata = model_runner_client
        .get_model_metadata(GetModelMetadataRequest {})
        .await
        .context("failed to get model metadata from the model runner")?
        .into_inner();
    let tokenizer = TokenizerWrapper::new(
        &args.tokenizer_path,
        args.draft_tokenizer_path.as_deref(),
        &model_metadata,
    )?;
    run_server(
        model_runner_client,
        tokenizer,
        &args.request_handler_socket_path,
    )
    .await
}

async fn connect_to_model_runner(socket_path: &Path) -> Result<ModelRunnerServiceClient<Channel>> {
    let channel = domain_socket::connect(socket_path)
        .await
        .context("failed to connect the request handler to the model runner")?;
    Ok(ModelRunnerServiceClient::new(channel))
}

async fn run_server(
    model_runner_client: ModelRunnerServiceClient<Channel>,
    tokenizer: TokenizerWrapper,
    socket_path: &Path,
) -> Result<()> {
    // Bind only after the upstream connection succeeds so the socket indicates
    // that the request handler is ready to forward requests.
    let listener = UnixListener::bind(socket_path)
        .context("failed to bind the request handler Unix domain socket")?;
    let (shutdown, shutdown_receiver) = RpcShutdown::channel();
    let service = RequestHandlerRpcService {
        model_runner_client,
        tokenizer,
        shutdown,
    };

    tonic::transport::Server::builder()
        .add_service(RequestHandlerServiceServer::new(service))
        .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
            let _ = shutdown_receiver.await;
        })
        .await
        .context("request handler RPC server failed")
}

struct RequestHandlerRpcService {
    model_runner_client: ModelRunnerServiceClient<Channel>,
    tokenizer: TokenizerWrapper,
    shutdown: RpcShutdown,
}

#[tonic::async_trait]
impl RequestHandlerService for RequestHandlerRpcService {
    type GenerateTextStream = ReceiverStream<Result<GenerateTextEvent, Status>>;

    async fn generate_text(
        &self,
        request: Request<GenerateText>,
    ) -> Result<Response<Self::GenerateTextStream>, Status> {
        let request = request.into_inner();
        let stream_output = request.stream_output;
        let request = self
            .tokenizer
            .create_generate_text_request(request)
            .map_err(|error| {
                Status::invalid_argument(format!("failed to process input: {error:#}"))
            })?;
        let mut model_runner_client = self.model_runner_client.clone();
        let response = model_runner_client.generate_text(request).await?;
        let (event_sender, event_receiver) = mpsc::channel(GENERATION_EVENT_QUEUE_CAPACITY);
        let decoder = self.tokenizer.create_token_decoder();
        tokio::spawn(forward_generation_events(
            response.into_inner(),
            event_sender,
            decoder,
            stream_output,
        ));
        Ok(Response::new(ReceiverStream::new(event_receiver)))
    }

    async fn shutdown(
        &self,
        _request: Request<Shutdown>,
    ) -> Result<Response<CommandResult>, Status> {
        self.shutdown.trigger()?;
        Ok(Response::new(CommandResult {}))
    }
}

async fn forward_generation_events(
    mut model_events: tonic::Streaming<ModelRunnerGenerateTextEvent>,
    event_sender: mpsc::Sender<Result<GenerateTextEvent, Status>>,
    mut decoder: IncrementalTokenDecoder,
    stream_output: bool,
) {
    let mut buffered_text = String::new();
    loop {
        let event = match model_events.message().await {
            Ok(Some(event)) => event.event,
            Ok(None) => return,
            Err(error) => {
                let _ = event_sender.send(Err(error)).await;
                return;
            }
        };
        let Some(event) = event else {
            let _ = event_sender
                .send(Err(Status::internal("model-runner event is empty")))
                .await;
            return;
        };
        match event {
            model_runner_generate_text_event::Event::TokenId(token_id) => {
                let fragment = match decoder.step(token_id) {
                    Ok(fragment) => fragment,
                    Err(error) => {
                        let _ = event_sender
                            .send(Err(Status::internal(format!(
                                "failed to decode model output: {error:#}"
                            ))))
                            .await;
                        return;
                    }
                };
                if let Some(fragment) = fragment {
                    if stream_output {
                        if event_sender
                            .send(Ok(GenerateTextEvent {
                                event: Some(generate_text_event::Event::Text(fragment)),
                            }))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    } else {
                        buffered_text.push_str(&fragment);
                    }
                }
            }
            model_runner_generate_text_event::Event::Stats(stats) => {
                if !stream_output
                    && event_sender
                        .send(Ok(GenerateTextEvent {
                            event: Some(generate_text_event::Event::Text(buffered_text)),
                        }))
                        .await
                        .is_err()
                {
                    return;
                }
                let _ = event_sender
                    .send(Ok(GenerateTextEvent {
                        event: Some(generate_text_event::Event::Stats(stats)),
                    }))
                    .await;
                return;
            }
        }
    }
}
