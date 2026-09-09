use crate::model_runner::server;
use crate::model_runner::InferenceDevice;
use crate::model_runner::KvCacheType;
use crate::model_runner::SchedulerConfig;
use crate::models::model_downloader::ModelArtifacts;
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

pub(crate) struct ModelRunnerProcessConfig {
    pub(crate) draft_token_count: usize,
    pub(crate) inference_device: InferenceDevice,
    pub(crate) kv_cache_type: KvCacheType,
    pub(crate) target_kv_cache_size_bytes: usize,
    pub(crate) scheduler_config: SchedulerConfig,
}

impl ModelRunnerProcess {
    pub(crate) async fn start(
        model_artifacts: &ModelArtifacts,
        draft_model_artifacts: Option<&ModelArtifacts>,
        config: ModelRunnerProcessConfig,
    ) -> Result<Self> {
        let socket_path = PathBuf::from(SOCKET_PATH);
        domain_socket::ensure_available(&socket_path, "model runner socket")?;
        let mut child_process = spawn(
            model_artifacts,
            draft_model_artifacts,
            &config,
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
    config: &ModelRunnerProcessConfig,
    socket_path: &Path,
) -> Result<ChildProcess> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    let mut command = Command::new(executable);
    command
        .process_group(0)
        .env(server::PROCESS_ENVIRONMENT_VARIABLE, "1")
        .arg("--model-path")
        .arg(&model_artifacts.gguf)
        .arg("--draft-token-count")
        .arg(config.draft_token_count.to_string())
        .arg("--inference-device")
        .arg(config.inference_device.cli_value())
        .arg("--kv-cache-type")
        .arg(config.kv_cache_type.to_string())
        .arg("--target-kv-cache-size-bytes")
        .arg(config.target_kv_cache_size_bytes.to_string())
        .arg("--max-batched-token-count")
        .arg(config.scheduler_config.max_batched_token_count.to_string())
        .arg("--max-active-request-count")
        .arg(config.scheduler_config.max_active_request_count.to_string())
        .arg("--scheduling-policy")
        .arg(config.scheduler_config.scheduling_policy.to_string());
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
