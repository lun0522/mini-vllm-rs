use super::cli::MainProcessArgs;
use super::server::ControlServer;
use crate::model_runner::client::ModelRunnerProcess;
use crate::model_runner::client::ModelRunnerProcessConfig;
use crate::model_runner::SchedulerConfig;
use crate::models::model_downloader::ModelArtifacts;
use crate::models::model_downloader::ModelDownloader;
use crate::models::ModelRole;
use crate::proto::inference_config::DraftModelConfig;
use crate::proto::inference_config::DraftModelRunnerConfig;
use crate::proto::inference_config::ModelConfig;
use crate::request_handler::client::RequestHandlerProcess;
use anyhow::Context;
use anyhow::Result;
use log::error;
use log::info;
use std::path::PathBuf;

struct DraftModelArtifacts {
    config: DraftModelRunnerConfig,
    tokenizer_path: PathBuf,
}

pub(crate) async fn run(args: MainProcessArgs) -> Result<()> {
    args.validate()?;
    info!("Server configuration:\n{args}");
    let MainProcessArgs {
        model,
        draft_model,
        inference_device,
        activation_dtype,
        kv_cache_type,
        target_kv_cache_size_bytes,
        max_batched_token_count,
        max_active_request_count,
        scheduling_policy,
        input_preprocessing_thread_count,
        trace_directory,
        request_socket,
        control_socket,
    } = args;
    let model_artifacts = download_model(model, ModelRole::Target)?;
    let (draft_model_runner_config, draft_tokenizer_path) = draft_model
        .map(download_draft_model)
        .transpose()?
        .map(|draft_model| (draft_model.config, draft_model.tokenizer_path))
        .unzip();
    let runtime_directory = tempfile::Builder::new()
        .prefix("mini-vllm-")
        .tempdir_in("/tmp")
        .context("failed to create the server runtime directory")?;
    let scheduler_config = SchedulerConfig {
        max_batched_token_count,
        max_active_request_count,
        scheduling_policy,
    };
    let model_runner_process = ModelRunnerProcess::start(
        &model_artifacts,
        runtime_directory.path().join("model-runner.sock"),
        ModelRunnerProcessConfig {
            inference_device,
            activation_dtype,
            kv_cache_type,
            target_kv_cache_size_bytes,
            draft_model_runner_config,
            scheduler_config,
            trace_directory,
        },
    )
    .await?;
    let request_handler_process = match RequestHandlerProcess::start(
        model_runner_process.socket_path(),
        &model_artifacts.tokenizer,
        draft_tokenizer_path.as_deref(),
        input_preprocessing_thread_count,
        request_socket,
    )
    .await
    {
        Ok(process) => process,
        Err(error) => {
            if let Err(shutdown_error) = model_runner_process.shutdown().await {
                error!("Failed to shut down the model runner: {shutdown_error:#}");
            }
            return Err(error);
        }
    };

    let control_server = ControlServer::bind(&control_socket)?;
    let serving_result = async {
        info!(
            "Listening for local requests on {}. Send a shutdown command to {} or press Ctrl-C to stop.",
            request_handler_process.socket_path().display(),
            control_socket.display(),
        );
        tokio::select! {
            ctrl_c_result = tokio::signal::ctrl_c() => {
                ctrl_c_result.context("failed to listen for Ctrl-C")
            }
            shutdown_result = control_server.wait_for_shutdown() => shutdown_result,
        }
    }
    .await;

    let request_handler_shutdown = request_handler_process.shutdown().await;
    let model_runner_shutdown = model_runner_process.shutdown().await;
    serving_result?;
    request_handler_shutdown?;
    model_runner_shutdown
}

fn download_draft_model(config: DraftModelConfig) -> Result<DraftModelArtifacts> {
    let model = config
        .model
        .expect("validated draft model configuration should contain a model");
    let token_count_policy = config
        .token_count_policy
        .expect("validated draft model configuration should contain a token-count policy");
    let ModelArtifacts {
        gguf: model_path,
        tokenizer: tokenizer_path,
    } = download_model(model, ModelRole::Draft)?;
    let model_path = model_path.into_os_string().into_string().map_err(|path| {
        anyhow::anyhow!(
            "draft model path is not valid UTF-8: {}",
            std::path::PathBuf::from(path).display()
        )
    })?;
    Ok(DraftModelArtifacts {
        config: DraftModelRunnerConfig {
            model_path,
            token_count_policy: Some(token_count_policy),
        },
        tokenizer_path,
    })
}

fn download_model(config: ModelConfig, role: ModelRole) -> Result<ModelArtifacts> {
    ModelDownloader::new(config, role).download()
}
