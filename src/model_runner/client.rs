use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::model_runner::server;
use crate::model_runner::KvCacheType;
use crate::model_runner::DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT;
use crate::proto::model_runner::model_runner_command::Command::Shutdown as ShutdownCommand;
use crate::proto::model_runner::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::model_runner::ModelRunnerCommand;
use crate::proto::model_runner::Shutdown;
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
const SOCKET_PATH: &str = "/tmp/mini-vllm-model-runner.sock";

/// Owns the worker process, RPC client, and Unix domain socket lifetime.
pub(crate) struct ModelRunnerProcess {
    rpc_client: ModelRunnerServiceClient<Channel>,
    child_process: ChildProcess,
    socket_path: PathBuf,
}

impl ModelRunnerProcess {
    pub(crate) async fn start(
        model_artifacts: &ModelArtifacts,
        draft_model_artifacts: Option<&ModelArtifacts>,
        draft_token_count: usize,
        kv_cache_type: KvCacheType,
        target_kv_cache_size_bytes: usize,
    ) -> Result<Self> {
        let socket_path = PathBuf::from(SOCKET_PATH);
        domain_socket::ensure_available(&socket_path, "model runner socket")?;
        let mut child_process = spawn(
            model_artifacts,
            draft_model_artifacts,
            draft_token_count,
            kv_cache_type,
            target_kv_cache_size_bytes,
            &socket_path,
        )?;
        let channel =
            match domain_socket::wait_for_server(&mut child_process, &socket_path, STARTUP_TIMEOUT)
                .await
            {
                Ok(channel) => channel,
                Err(error) => {
                    let _ = child_process.stop();
                    let _ = domain_socket::remove(&socket_path, "model runner socket");
                    return Err(error);
                }
            };
        Ok(Self {
            rpc_client: ModelRunnerServiceClient::new(channel),
            child_process,
            socket_path,
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
                command: Some(ShutdownCommand(Shutdown {})),
            })
            .await
            .context("failed to shut down the model runner");
        if let Err(error) = shutdown_result {
            self.child_process.stop()?;
            domain_socket::remove(&self.socket_path, "model runner socket")?;
            return Err(error);
        }
        let wait_result = self.child_process.wait();
        let remove_result = domain_socket::remove(&self.socket_path, "model runner socket");
        wait_result?;
        remove_result
    }
}

impl Drop for ModelRunnerProcess {
    fn drop(&mut self) {
        if let Err(error) = self.child_process.stop() {
            log::error!("Failed to stop the model runner during cleanup: {error:#}");
        }
        if let Err(error) = domain_socket::remove(&self.socket_path, "model runner socket") {
            log::error!("Failed to remove the model runner socket: {error:#}");
        }
    }
}

fn spawn(
    model_artifacts: &ModelArtifacts,
    draft_model_artifacts: Option<&ModelArtifacts>,
    draft_token_count: usize,
    kv_cache_type: KvCacheType,
    target_kv_cache_size_bytes: usize,
    socket_path: &Path,
) -> Result<ChildProcess> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    let (kv_cache_type_arg, per_page_token_count) = match kv_cache_type {
        KvCacheType::Contiguous => ("contiguous", DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT),
        KvCacheType::Paged {
            per_page_token_count,
        } => ("paged", per_page_token_count),
    };
    let mut command = Command::new(executable);
    command
        .process_group(0)
        .env(server::PROCESS_ENVIRONMENT_VARIABLE, "1")
        .arg("--model-path")
        .arg(&model_artifacts.gguf)
        .arg("--draft-token-count")
        .arg(draft_token_count.to_string())
        .arg("--kv-cache-type")
        .arg(kv_cache_type_arg)
        .arg("--kv-cache-page-token-count")
        .arg(per_page_token_count.to_string())
        .arg("--target-kv-cache-size-bytes")
        .arg(target_kv_cache_size_bytes.to_string());
    if let Some(draft_model_artifacts) = draft_model_artifacts {
        command
            .arg("--draft-model-path")
            .arg(&draft_model_artifacts.gguf);
    }
    let child = command
        .arg("--socket-path")
        .arg(socket_path)
        .spawn()
        .context("failed to start the model runner process")?;
    Ok(ChildProcess::new(child, "model runner".to_owned()))
}
