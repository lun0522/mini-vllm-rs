use crate::proto::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerService;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerServiceServer;
use crate::proto::request_handler::CommandResult;
use crate::proto::request_handler::GenerateText;
use crate::proto::request_handler::Shutdown;
use crate::proto::GenerateTextEvent;
use crate::proto::GetModelMetadataRequest;
use crate::utils::domain_socket;
use crate::utils::rpc_shutdown::RpcShutdown;
use anyhow::Context;
use anyhow::Result;
use argh::FromArgs;
use std::path::Path;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Channel;
use tonic::Request;
use tonic::Response;
use tonic::Status;

use super::tokenizer::create_generate_tokens_request;
use super::tokenizer::load_and_validate_tokenizer;
use super::tokenizer::ModelArchitecture;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_REQUEST_HANDLER";

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
    let tokenizer = load_and_validate_tokenizer(
        &args.tokenizer_path,
        args.draft_tokenizer_path.as_deref(),
        &model_metadata,
    )?;
    let target_model_metadata = model_metadata
        .target_model
        .context("model runner did not report target model metadata")?;
    let architecture = ModelArchitecture::try_from(target_model_metadata.architecture)
        .context("model runner reported an invalid model architecture")?;
    run_server(
        model_runner_client,
        tokenizer,
        architecture,
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
    tokenizer: Tokenizer,
    architecture: ModelArchitecture,
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
        architecture,
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
    tokenizer: Tokenizer,
    architecture: ModelArchitecture,
    shutdown: RpcShutdown,
}

#[tonic::async_trait]
impl RequestHandlerService for RequestHandlerRpcService {
    type GenerateTextStream = tonic::Streaming<GenerateTextEvent>;

    async fn generate_text(
        &self,
        request: Request<GenerateText>,
    ) -> Result<Response<Self::GenerateTextStream>, Status> {
        let request = create_generate_tokens_request(
            &self.tokenizer,
            self.architecture,
            request.into_inner(),
        )
        .map_err(|error| Status::invalid_argument(format!("failed to process input: {error:#}")))?;
        let mut model_runner_client = self.model_runner_client.clone();
        let response = model_runner_client.generate_text(request).await?;
        Ok(Response::new(response.into_inner()))
    }

    async fn shutdown(
        &self,
        _request: Request<Shutdown>,
    ) -> Result<Response<CommandResult>, Status> {
        self.shutdown.trigger()?;
        Ok(Response::new(CommandResult {}))
    }
}
