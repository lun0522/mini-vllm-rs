use crate::proto::request_handler::request_handler_service_client::RequestHandlerServiceClient;
use crate::proto::CommandResult;
use crate::proto::GenerateText;
use crate::proto::Shutdown;
use crate::request_handler::server;
use anyhow::Context;
use anyhow::Result;
use hyper_util::rt::TokioIo;
use std::io::ErrorKind;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Child;
use std::process::Command;
use std::time::Duration;
use tokio::net::UnixStream;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tower::service_fn;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(1);
const CONNECTION_RETRY_DELAY: Duration = Duration::from_millis(100);

/// Owns the request handler process, RPC client, and public socket path.
pub(crate) struct RequestHandlerProcess {
    rpc_client: RequestHandlerServiceClient<Channel>,
    child_process: Child,
    socket_path: PathBuf,
}

impl RequestHandlerProcess {
    pub(crate) async fn start(
        model_runner_socket_path: &Path,
        request_handler_socket_path: PathBuf,
    ) -> Result<Self> {
        if request_handler_socket_path
            .try_exists()
            .context("failed to inspect the request handler socket path")?
        {
            anyhow::bail!(
                "request handler socket '{}' already exists",
                request_handler_socket_path.display()
            );
        }

        let mut child_process = spawn(
            model_runner_socket_path,
            request_handler_socket_path.as_path(),
        )?;
        let startup_started = tokio::time::Instant::now();
        let mut last_connection_error = None;

        while startup_started.elapsed() < STARTUP_TIMEOUT {
            let process_status = match child_process.try_wait() {
                Ok(process_status) => process_status,
                Err(error) => {
                    let _ = stop(&mut child_process);
                    let _ = remove_socket(&request_handler_socket_path);
                    return Err(error).context("failed to inspect the request handler process");
                }
            };
            if let Some(status) = process_status {
                let _ = remove_socket(&request_handler_socket_path);
                anyhow::bail!("request handler exited before accepting requests: {status}");
            }

            match connect(request_handler_socket_path.as_path()).await {
                Ok(channel) => {
                    return Ok(Self {
                        rpc_client: RequestHandlerServiceClient::new(channel),
                        child_process,
                        socket_path: request_handler_socket_path,
                    });
                }
                Err(error) => last_connection_error = Some(error.to_string()),
            }
            tokio::time::sleep(CONNECTION_RETRY_DELAY).await;
        }

        stop(&mut child_process)?;
        remove_socket(&request_handler_socket_path)?;
        anyhow::bail!(
            "timed out after {STARTUP_TIMEOUT:?} waiting for the request handler Unix domain \
             socket. Last connection error: {}",
            last_connection_error.as_deref().unwrap_or("none")
        )
    }

    pub(crate) async fn generate_text(&self, request: GenerateText) -> Result<CommandResult> {
        let mut rpc_client = self.rpc_client.clone();
        Ok(rpc_client
            .generate_text(request)
            .await
            .context("request handler generation request failed")?
            .into_inner())
    }

    pub(crate) fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub(crate) async fn shutdown(mut self) -> Result<()> {
        log::info!("Shutting down the request handler process");
        let shutdown_result = self
            .rpc_client
            .shutdown(Shutdown {})
            .await
            .context("failed to shut down the request handler");
        if let Err(error) = shutdown_result {
            stop(&mut self.child_process)?;
            remove_socket(&self.socket_path)?;
            return Err(error);
        }

        let wait_result = wait(&mut self.child_process);
        let remove_result = remove_socket(&self.socket_path);
        wait_result?;
        remove_result
    }
}

impl Drop for RequestHandlerProcess {
    fn drop(&mut self) {
        if let Err(error) = stop(&mut self.child_process) {
            log::error!("Failed to stop the request handler during cleanup: {error:#}");
        }
        if let Err(error) = remove_socket(&self.socket_path) {
            log::error!("Failed to remove the request handler socket: {error:#}");
        }
    }
}

fn spawn(model_runner_socket_path: &Path, socket_path: &Path) -> Result<Child> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    Command::new(executable)
        .process_group(0)
        .env(server::PROCESS_ENVIRONMENT_VARIABLE, "1")
        .arg("--model-runner-socket-path")
        .arg(model_runner_socket_path)
        .arg("--request-handler-socket-path")
        .arg(socket_path)
        .spawn()
        .context("failed to start the request handler process")
}

async fn connect(socket_path: &Path) -> Result<Channel> {
    let connector_path = socket_path.to_owned();
    Endpoint::from_static("http://localhost")
        .connect_timeout(CONNECTION_TIMEOUT)
        .connect_with_connector(service_fn(move |_| {
            let connector_path = connector_path.clone();
            async move { UnixStream::connect(connector_path).await.map(TokioIo::new) }
        }))
        .await
        .context("failed to connect to the request handler socket")
}

fn stop(child_process: &mut Child) -> Result<()> {
    if child_process
        .try_wait()
        .context("failed to inspect the request handler process")?
        .is_none()
    {
        child_process
            .kill()
            .context("failed to stop the request handler process")?;
        child_process
            .wait()
            .context("failed to wait for the request handler process")?;
    }
    Ok(())
}

fn wait(child_process: &mut Child) -> Result<()> {
    let status = child_process
        .wait()
        .context("failed to wait for the request handler process")?;
    if !status.success() {
        anyhow::bail!("request handler exited unsuccessfully: {status}");
    }
    Ok(())
}

fn remove_socket(socket_path: &Path) -> Result<()> {
    match std::fs::remove_file(socket_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("failed to remove the request handler socket"),
    }
}
