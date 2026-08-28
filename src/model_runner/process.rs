use crate::model_loaders::model_downloader::ModelFiles;
use crate::model_runner::server;
use crate::proto::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::ModelRunnerCommand;
use crate::proto::Shutdown;
use crate::utils::child_process::ChildProcess;
use crate::utils::domain_socket;
use anyhow::Context;
use anyhow::Result;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use tonic::transport::Channel;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);

/// Owns the worker process, RPC client, and Unix domain socket lifetime.
pub(crate) struct ModelRunnerProcess {
    rpc_client: ModelRunnerServiceClient<Channel>,
    child_process: ChildProcess,
    socket_path: PathBuf,
    _socket_directory: tempfile::TempDir,
}

impl ModelRunnerProcess {
    pub(crate) async fn start(
        model_files: &ModelFiles,
        draft_model_files: Option<ModelFiles>,
        draft_token_count: usize,
    ) -> Result<Self> {
        let (socket_directory, socket_path) = create_socket()?;
        let mut child_process = spawn(
            model_files,
            draft_model_files.as_ref(),
            draft_token_count,
            &socket_path,
        )?;
        let channel =
            domain_socket::wait_for_server(&mut child_process, &socket_path, STARTUP_TIMEOUT)
                .await?;
        Ok(Self {
            rpc_client: ModelRunnerServiceClient::new(channel),
            child_process,
            socket_path,
            _socket_directory: socket_directory,
        })
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
            self.child_process.stop()?;
            return Err(error);
        }
        self.child_process.wait()
    }
}

fn spawn(
    model_files: &ModelFiles,
    draft_model_files: Option<&ModelFiles>,
    draft_token_count: usize,
    socket_path: &Path,
) -> Result<ChildProcess> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    let mut command = Command::new(executable);
    command
        .process_group(0)
        .env(server::PROCESS_ENVIRONMENT_VARIABLE, "1")
        .arg("--tokenizer-path")
        .arg(&model_files.tokenizer)
        .arg("--config-path")
        .arg(&model_files.config)
        .arg("--weights-path")
        .arg(&model_files.weights)
        .arg("--draft-token-count")
        .arg(draft_token_count.to_string());
    if let Some(draft_model_files) = draft_model_files {
        command
            .arg("--draft-tokenizer-path")
            .arg(&draft_model_files.tokenizer)
            .arg("--draft-config-path")
            .arg(&draft_model_files.config)
            .arg("--draft-weights-path")
            .arg(&draft_model_files.weights);
    }
    let child = command
        .arg("--socket-path")
        .arg(socket_path)
        .spawn()
        .context("failed to start the model runner process")?;
    Ok(ChildProcess::new(child, "model runner".to_owned()))
}

fn create_socket() -> Result<(tempfile::TempDir, PathBuf)> {
    let socket_directory = tempfile::Builder::new()
        .prefix("mini-vllm-rs-")
        .tempdir_in("/tmp")
        .context("failed to create the model runner socket directory")?;
    let socket_path = socket_directory.path().join("model-runner.sock");
    Ok((socket_directory, socket_path))
}
