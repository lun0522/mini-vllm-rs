use crate::proto::model_runner_command;
use crate::proto::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerService;
use crate::proto::request_handler::request_handler_service_server::RequestHandlerServiceServer;
use crate::proto::CommandResult;
use crate::proto::GenerateText;
use crate::proto::ModelRunnerCommand;
use crate::proto::Shutdown;
use anyhow::Context;
use anyhow::Result;
use argh::FromArgs;
use hyper_util::rt::TokioIo;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use tokio::net::UnixListener;
use tokio::net::UnixStream;
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tonic::Request;
use tonic::Response;
use tonic::Status;
use tower::service_fn;

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
    let connector_path = socket_path.to_owned();
    let channel = Endpoint::from_static("http://localhost")
        .connect_with_connector(service_fn(move |_| {
            let connector_path = connector_path.clone();
            async move { UnixStream::connect(connector_path).await.map(TokioIo::new) }
        }))
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
    let (shutdown_sender, shutdown_receiver) = oneshot::channel();
    let service = RequestHandlerRpcService {
        model_runner_client,
        shutdown_sender: Mutex::new(Some(shutdown_sender)),
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
    shutdown_sender: Mutex<Option<oneshot::Sender<()>>>,
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
        let shutdown_sender = self
            .shutdown_sender
            .lock()
            .map_err(|_| Status::internal("shutdown sender lock is poisoned"))?
            .take()
            .ok_or_else(|| Status::failed_precondition("shutdown was already requested"))?;
        shutdown_sender
            .send(())
            .map_err(|_| Status::internal("request handler shutdown receiver was dropped"))?;
        Ok(Response::new(CommandResult {}))
    }
}
