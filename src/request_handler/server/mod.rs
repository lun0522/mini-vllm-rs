use crate::proto::model_runner::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::model_runner::GenerateTextEvent as ModelRunnerGenerateTextEvent;
use crate::proto::model_runner::GetModelMetadataRequest;
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
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Channel;
use tonic::Request;
use tonic::Response;
use tonic::Status;

mod generation_event_processor;
mod input_preprocessing_pool;
mod tokenizer;

use generation_event_processor::GenerationEventProcessor;
use input_preprocessing_pool::InputPreprocessingPool;
use tokenizer::IncrementalTokenDecoder;
use tokenizer::TokenizerWrapper;

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
    /// number of threads used for concurrent input preprocessing
    #[argh(option)]
    input_preprocessing_thread_count: usize,
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
    let input_preprocessing_pool =
        InputPreprocessingPool::new(tokenizer, args.input_preprocessing_thread_count)?;
    run_server(
        model_runner_client,
        input_preprocessing_pool,
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
    input_preprocessing_pool: InputPreprocessingPool,
    socket_path: &Path,
) -> Result<()> {
    // Bind only after the upstream connection succeeds so the socket indicates
    // that the request handler is ready to forward requests.
    let listener = UnixListener::bind(socket_path)
        .context("failed to bind the request handler Unix domain socket")?;
    let (shutdown, shutdown_receiver) = RpcShutdown::channel();
    let service = RequestHandlerRpcService {
        model_runner_client,
        input_preprocessing_pool,
        request_id: RequestId::new(),
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
    input_preprocessing_pool: InputPreprocessingPool,
    request_id: RequestId,
    shutdown: RpcShutdown,
}

struct RequestId(AtomicU64);

impl RequestId {
    fn new() -> Self {
        Self(AtomicU64::new(1))
    }

    /// Returns the current request ID and advances to the next one.
    fn next(&self) -> Result<u64, &'static str> {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |request_id| {
                request_id.checked_add(1)
            })
            .map_err(|_| "request ID space is exhausted")
    }
}

#[tonic::async_trait]
impl RequestHandlerService for RequestHandlerRpcService {
    type GenerateTextStream = ReceiverStream<Result<GenerateTextEvent, Status>>;

    async fn generate_text(
        &self,
        request: Request<GenerateText>,
    ) -> Result<Response<Self::GenerateTextStream>, Status> {
        let mut request = request.into_inner();
        let request_id = self.request_id.next().map_err(Status::resource_exhausted)?;
        request.request_id = request_id;
        log::info!("Request handler state: request_id={request_id} status=arrived");
        let request = self
            .input_preprocessing_pool
            .preprocess(request)
            .await
            .map_err(|error| {
                Status::invalid_argument(format!("failed to process input: {error:#}"))
            })?;
        let mut model_runner_client = self.model_runner_client.clone();
        let response = model_runner_client
            .generate_text(request.generate_text_request)
            .await?;
        let (event_sender, event_receiver) = mpsc::channel(GENERATION_EVENT_QUEUE_CAPACITY);
        tokio::spawn(forward_generation_events(
            response.into_inner(),
            event_sender,
            request.token_decoder,
            request.stream_output,
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
    decoder: IncrementalTokenDecoder,
    stream_output: bool,
) {
    let mut processor = GenerationEventProcessor::new(decoder, stream_output);
    loop {
        let event = match model_events.message().await {
            Ok(Some(event)) => event,
            Ok(None) => return,
            Err(error) => {
                let _ = event_sender.send(Err(error)).await;
                return;
            }
        };

        let processed = match processor.process(event) {
            Ok(processed) => processed,
            Err(error) => {
                let _ = event_sender.send(Err(*error)).await;
                return;
            }
        };
        for event in processed.events {
            if event_sender.send(Ok(event)).await.is_err() {
                return;
            }
        }
        if processed.finished {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RequestId;

    #[test]
    fn request_id_returns_the_current_value_before_incrementing() {
        let request_id = RequestId::new();

        assert_eq!(request_id.next(), Ok(1));
        assert_eq!(request_id.next(), Ok(2));
    }
}
