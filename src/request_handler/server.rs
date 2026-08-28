use crate::proto::model_runner_command;
use crate::proto::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerService;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerServiceServer;
use crate::proto::CommandResult;
use crate::proto::GenerateText;
use crate::proto::ModelRunnerCommand;
use crate::proto::Shutdown;
use crate::utils::domain_socket;
use crate::utils::rpc_shutdown::RpcShutdown;
use anyhow::Context;
use anyhow::Result;
use argh::FromArgs;
use std::path::Path;
use std::path::PathBuf;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Channel;
use tonic::Request;
use tonic::Response;
use tonic::Status;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_REQUEST_HANDLER";

/// Handles local inference requests and forwards them to the model runner.
#[derive(FromArgs)]
pub(crate) struct RequestHandlerProcessArgs {
    /// model runner Unix domain socket path
    #[argh(option)]
    model_runner_socket_path: PathBuf,
    /// request handler Unix domain socket path
    #[argh(option)]
    request_handler_socket_path: PathBuf,
}

pub(crate) async fn run(args: RequestHandlerProcessArgs) -> Result<()> {
    let model_runner_client = connect_to_model_runner(&args.model_runner_socket_path).await?;
    run_server(model_runner_client, &args.request_handler_socket_path).await
}

async fn connect_to_model_runner(socket_path: &Path) -> Result<ModelRunnerServiceClient<Channel>> {
    let channel = domain_socket::connect(socket_path)
        .await
        .context("failed to connect the request handler to the model runner")?;
    Ok(ModelRunnerServiceClient::new(channel))
}

async fn run_server(
    model_runner_client: ModelRunnerServiceClient<Channel>,
    socket_path: &Path,
) -> Result<()> {
    // Bind only after the upstream connection succeeds so the socket indicates
    // that the request handler is ready to forward requests.
    let listener = UnixListener::bind(socket_path)
        .context("failed to bind the request handler Unix domain socket")?;
    let (shutdown, shutdown_receiver) = RpcShutdown::channel();
    let service = RequestHandlerRpcService {
        model_runner_client,
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
    shutdown: RpcShutdown,
}

#[tonic::async_trait]
impl RequestHandlerService for RequestHandlerRpcService {
    async fn generate_text(
        &self,
        request: Request<GenerateText>,
    ) -> Result<Response<CommandResult>, Status> {
        let mut model_runner_client = self.model_runner_client.clone();
        let response = model_runner_client
            .handle_command(ModelRunnerCommand {
                command: Some(model_runner_command::Command::GenerateText(
                    request.into_inner(),
                )),
            })
            .await
            .map_err(|error| Status::internal(format!("model runner request failed: {error}")))?;
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
