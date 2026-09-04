use crate::proto::main_process::main_process_service_server::MainProcessService;
use crate::proto::main_process::main_process_service_server::MainProcessServiceServer;
use crate::proto::main_process::CommandResult;
use crate::proto::main_process::Shutdown;
use crate::utils::domain_socket;
use crate::utils::rpc_shutdown::RpcShutdown;
use anyhow::Context;
use anyhow::Result;
use std::path::Path;
use std::path::PathBuf;
use tokio::net::UnixListener;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

pub(crate) struct ControlServer {
    listener: Option<UnixListener>,
    socket_path: PathBuf,
}

impl ControlServer {
    pub(crate) fn bind(socket_path: &Path) -> Result<Self> {
        domain_socket::ensure_available(socket_path, "main process control socket")?;
        let listener = UnixListener::bind(socket_path)
            .context("failed to bind the main process control socket")?;
        Ok(Self {
            listener: Some(listener),
            socket_path: socket_path.to_owned(),
        })
    }

    pub(crate) async fn wait_for_shutdown(mut self) -> Result<()> {
        let listener = self
            .listener
            .take()
            .expect("control server listener must exist");
        let (shutdown, shutdown_receiver) = RpcShutdown::channel();
        let service = MainProcessRpcService { shutdown };

        tonic::transport::Server::builder()
            .add_service(MainProcessServiceServer::new(service))
            .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
                let _ = shutdown_receiver.await;
            })
            .await
            .context("main process control RPC server failed")
    }
}

impl Drop for ControlServer {
    fn drop(&mut self) {
        if let Err(error) = domain_socket::remove(&self.socket_path, "main process control socket")
        {
            log::error!("Failed to remove the main process control socket: {error:#}");
        }
    }
}

struct MainProcessRpcService {
    shutdown: RpcShutdown,
}

#[tonic::async_trait]
impl MainProcessService for MainProcessRpcService {
    async fn shutdown(
        &self,
        _request: Request<Shutdown>,
    ) -> Result<Response<CommandResult>, Status> {
        self.shutdown.trigger()?;
        Ok(Response::new(CommandResult {}))
    }
}

#[cfg(test)]
mod tests {
    use super::ControlServer;
    use crate::proto::main_process::main_process_service_client::MainProcessServiceClient;
    use crate::proto::main_process::Shutdown;
    use crate::utils::domain_socket;

    #[tokio::test]
    async fn shutdown_rpc_stops_server_and_removes_socket() {
        let directory = tempfile::tempdir_in("/tmp").unwrap();
        let socket_path = directory.path().join("control.sock");
        let server = ControlServer::bind(&socket_path).unwrap();
        let server_task = tokio::spawn(server.wait_for_shutdown());
        let channel = domain_socket::connect(&socket_path).await.unwrap();
        let mut client = MainProcessServiceClient::new(channel);

        client.shutdown(Shutdown {}).await.unwrap();

        server_task.await.unwrap().unwrap();
        assert!(!socket_path.exists());
    }
}
