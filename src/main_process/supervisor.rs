use super::cli::MainProcessArgs;
use super::server::ControlServer;
use crate::model_loaders::model_downloader::ModelDownloader;
use crate::model_loaders::ModelRole;
use crate::model_runner::client::ModelRunnerProcess;
use crate::request_handler::client::RequestHandlerProcess;
use anyhow::Context;
use anyhow::Result;
use log::error;
use log::info;

pub(crate) async fn run(args: MainProcessArgs) -> Result<()> {
    info!("Server configuration:\n{args}");
    let model_downloader = ModelDownloader::new(args.model, ModelRole::Target)?;
    let model_artifacts = model_downloader.download()?;
    let draft_model_artifacts = args
        .draft_model
        .map(|model| ModelDownloader::new(model, ModelRole::Draft))
        .transpose()?
        .map(|downloader| downloader.download())
        .transpose()?;
    let model_runner_process = ModelRunnerProcess::start(
        &model_artifacts,
        draft_model_artifacts.as_ref(),
        args.draft_token_count,
        args.kv_cache_type,
        args.target_kv_cache_size_bytes,
    )
    .await?;
    let request_handler_process = match RequestHandlerProcess::start(
        model_runner_process.socket_path(),
        &model_artifacts.tokenizer,
        draft_model_artifacts
            .as_ref()
            .map(|artifacts| artifacts.tokenizer.as_path()),
        args.request_socket,
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

    let control_server = ControlServer::bind(&args.control_socket)?;
    let serving_result = async {
        info!(
            "Listening for local requests on {}. Send a shutdown command to {} or press Ctrl-C to stop.",
            request_handler_process.socket_path().display(),
            args.control_socket.display(),
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
