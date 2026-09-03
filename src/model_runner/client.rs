use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::model_runner::server;
use crate::model_runner::KvCacheType;
use crate::model_runner::DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT;
use crate::proto::model_runner_service_client::ModelRunnerServiceClient;
use crate::proto::ModelPaths;
use crate::proto::ModelRunnerCommand;
use crate::proto::Shutdown;
use crate::utils::child_process::ChildProcess;
use crate::utils::domain_socket;
use crate::utils::textproto::format_textproto;
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
        model_artifacts: &ModelArtifacts,
        draft_model_artifacts: Option<ModelArtifacts>,
        draft_token_count: usize,
        kv_cache_type: KvCacheType,
        target_kv_cache_size_bytes: usize,
    ) -> Result<Self> {
        let (socket_directory, socket_path) = create_socket()?;
        let mut child_process = spawn(
            model_artifacts,
            draft_model_artifacts.as_ref(),
            draft_token_count,
            kv_cache_type,
            target_kv_cache_size_bytes,
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
    model_artifacts: &ModelArtifacts,
    draft_model_artifacts: Option<&ModelArtifacts>,
    draft_token_count: usize,
    kv_cache_type: KvCacheType,
    target_kv_cache_size_bytes: usize,
    socket_path: &Path,
) -> Result<ChildProcess> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    let model = format_model_paths(model_artifacts)?;
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
        .arg("--model")
        .arg(model)
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
            .arg("--draft-model")
            .arg(format_model_paths(draft_model_artifacts)?);
    }
    let child = command
        .arg("--socket-path")
        .arg(socket_path)
        .spawn()
        .context("failed to start the model runner process")?;
    Ok(ChildProcess::new(child, "model runner".to_owned()))
}

fn format_model_paths(model_artifacts: &ModelArtifacts) -> Result<String> {
    let paths = ModelPaths {
        tokenizer_path: model_artifacts
            .tokenizer
            .to_str()
            .context("tokenizer path is not valid UTF-8")?
            .to_owned(),
        gguf_path: model_artifacts
            .gguf
            .to_str()
            .context("GGUF path is not valid UTF-8")?
            .to_owned(),
    };
    format_textproto(&paths, "model_runner.ModelPaths").map_err(anyhow::Error::msg)
}

fn create_socket() -> Result<(tempfile::TempDir, PathBuf)> {
    let socket_directory = tempfile::Builder::new()
        .prefix("mini-vllm-rs-")
        .tempdir_in("/tmp")
        .context("failed to create the model runner socket directory")?;
    let socket_path = socket_directory.path().join("model-runner.sock");
    Ok((socket_directory, socket_path))
}
