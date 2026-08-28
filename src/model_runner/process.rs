use crate::model_loaders::model_downloader::ModelFiles;
use crate::model_runner::server;
use crate::proto::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::ModelRunnerCommand;
use crate::proto::Shutdown;
use anyhow::Context;
use anyhow::Result;
use hyper_util::rt::TokioIo;
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

const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(1);
const CONNECTION_RETRY_DELAY: Duration = Duration::from_millis(100);

/// Owns the worker process, RPC client, and Unix domain socket lifetime.
pub(crate) struct ModelRunnerProcess {
    rpc_client: ModelRunnerServiceClient<Channel>,
    child_process: Child,
    socket_path: PathBuf,
    _socket_directory: tempfile::TempDir,
}

impl ModelRunnerProcess {
    pub(crate) async fn start(model_files: &ModelFiles) -> Result<Self> {
        let (socket_directory, socket_path) = create_socket()?;
        let mut child_process = spawn(model_files, &socket_path)?;
        let startup_started = tokio::time::Instant::now();
        let mut last_connection_error = None;

        while startup_started.elapsed() < STARTUP_TIMEOUT {
            let process_status = match child_process.try_wait() {
                Ok(process_status) => process_status,
                Err(error) => {
                    let _ = stop(&mut child_process);
                    return Err(error).context("failed to inspect the model runner process");
                }
            };
            if let Some(status) = process_status {
                anyhow::bail!("model runner exited before accepting commands: {status}");
            }

            let connector_path = socket_path.to_owned();
            let channel = Endpoint::from_static("http://localhost")
                .connect_timeout(CONNECTION_TIMEOUT)
                .connect_with_connector(service_fn(move |_| {
                    let connector_path = connector_path.clone();
                    async move { UnixStream::connect(connector_path).await.map(TokioIo::new) }
                }))
                .await;
            match channel {
                Ok(channel) => {
                    return Ok(Self {
                        rpc_client: ModelRunnerServiceClient::new(channel),
                        child_process,
                        socket_path,
                        _socket_directory: socket_directory,
                    });
                }
                Err(error) => last_connection_error = Some(error.to_string()),
            }
            tokio::time::sleep(CONNECTION_RETRY_DELAY).await;
        }

        stop(&mut child_process)?;
        anyhow::bail!(
            "timed out after {STARTUP_TIMEOUT:?} waiting for the model runner Unix domain \
             socket. Last connection error: {}",
            last_connection_error.as_deref().unwrap_or("none")
        )
    }

    pub(crate) fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub(crate) async fn shutdown(mut self) -> Result<()> {
        log::info!("Shutting down the model runner process");
        let shutdown_result = self
            .rpc_client
            .handle_command(ModelRunnerCommand {
                command: Some(crate::proto::model_runner_command::Command::Shutdown(
                    Shutdown {},
                )),
            })
            .await
            .context("failed to shut down the model runner");
        if let Err(error) = shutdown_result {
            stop(&mut self.child_process)?;
            return Err(error);
        }
        wait(&mut self.child_process)
    }
}

impl Drop for ModelRunnerProcess {
    fn drop(&mut self) {
        if let Err(error) = stop(&mut self.child_process) {
            log::error!("Failed to stop the model runner process during cleanup: {error:#}");
        }
    }
}

fn spawn(model_files: &ModelFiles, socket_path: &Path) -> Result<Child> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    Command::new(executable)
        .process_group(0)
        .env(server::PROCESS_ENVIRONMENT_VARIABLE, "1")
        .arg("--tokenizer-path")
        .arg(&model_files.tokenizer)
        .arg("--config-path")
        .arg(&model_files.config)
        .arg("--weights-path")
        .arg(&model_files.weights)
        .arg("--socket-path")
        .arg(socket_path)
        .spawn()
        .context("failed to start the model runner process")
}

fn create_socket() -> Result<(tempfile::TempDir, PathBuf)> {
    let socket_directory = tempfile::Builder::new()
        .prefix("mini-vllm-rs-")
        .tempdir_in("/tmp")
        .context("failed to create the model runner socket directory")?;
    let socket_path = socket_directory.path().join("model-runner.sock");
    Ok((socket_directory, socket_path))
}

fn stop(child_process: &mut Child) -> Result<()> {
    if child_process
        .try_wait()
        .context("failed to inspect the model runner process")?
        .is_none()
    {
        child_process
            .kill()
            .context("failed to stop the model runner process")?;
        child_process
            .wait()
            .context("failed to wait for the model runner process")?;
    }
    Ok(())
}

fn wait(child_process: &mut Child) -> Result<()> {
    let status = child_process
        .wait()
        .context("failed to wait for the model runner process")?;
    if !status.success() {
        anyhow::bail!("model runner exited unsuccessfully: {status}");
    }
    Ok(())
}
