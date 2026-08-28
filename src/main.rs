mod main_process;
mod model_loaders;
mod model_runner;
mod proto;
mod request_handler;
mod utils;

use crate::main_process::cli::MainProcessArgs;
use crate::main_process::example::run_example;
use crate::model_loaders::model_downloader::ModelDownloader;
use crate::model_runner::client::ModelRunnerProcess;
use crate::model_runner::server as model_runner_server;
use crate::request_handler::process::RequestHandlerProcess;
use crate::request_handler::server as request_handler_server;
use anyhow::Context;
use anyhow::Result;
use env_logger::Env;
use log::error;
use log::info;

#[tokio::main]
async fn main() {
    initialize_logging();

    let result = if std::env::var_os(model_runner_server::PROCESS_ENVIRONMENT_VARIABLE).is_some() {
        let args: model_runner_server::ModelRunnerProcessArgs = argh::from_env();
        model_runner_server::run(args).await
    } else if std::env::var_os(request_handler_server::PROCESS_ENVIRONMENT_VARIABLE).is_some() {
        let args: request_handler_server::RequestHandlerProcessArgs = argh::from_env();
        request_handler_server::run(args).await
    } else {
        let args = main_process::cli::parse();
        run_main_process(args).await
    };
    if let Err(err) = result {
        error!("{err:#}");
        std::process::exit(1);
    }
}

async fn run_main_process(args: MainProcessArgs) -> Result<()> {
    info!("Server configuration:\n{args}");
    let model_downloader = ModelDownloader::new(args.model)?;
    let model_artifacts = model_downloader.download()?;
    let draft_model_artifacts = args
        .draft_model
        .map(ModelDownloader::new)
        .transpose()?
        .map(|downloader| downloader.download())
        .transpose()?;
    let model_runner_process = ModelRunnerProcess::start(
        &model_artifacts,
        draft_model_artifacts,
        args.draft_token_count,
    )
    .await?;
    let request_handler_process =
        match RequestHandlerProcess::start(model_runner_process.socket_path(), args.request_socket)
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

    let serving_result = async {
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        if args.run_example && run_example(&request_handler_process, ctrl_c.as_mut()).await? {
            return Ok(());
        }
        info!(
            "Listening for local requests on {}. Press Ctrl-C to stop.",
            request_handler_process.socket_path().display()
        );
        ctrl_c.await.context("failed to listen for Ctrl-C")
    }
    .await;

    let request_handler_shutdown = request_handler_process.shutdown().await;
    let model_runner_shutdown = model_runner_process.shutdown().await;
    serving_result?;
    request_handler_shutdown?;
    model_runner_shutdown
}

fn initialize_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}
