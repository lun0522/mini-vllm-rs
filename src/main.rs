mod main_process;
mod model_runner;
mod models;
mod proto;
mod request_handler;
mod utils;

use crate::model_runner::server as model_runner_server;
use crate::request_handler::server as request_handler_server;
use anyhow::Context;
use env_logger::Env;
use log::error;
use log::info;
use std::fs::File;
use std::path::Path;
use std::process::ExitCode;
use tracing_chrome::ChromeLayerBuilder;
use tracing_chrome::FlushGuard;
use tracing_subscriber::layer::SubscriberExt;

#[tokio::main]
async fn main() -> ExitCode {
    initialize_logging();
    let result = run().await;
    if let Err(err) = result {
        error!("{err:#}");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

async fn run() -> anyhow::Result<()> {
    if std::env::var_os(model_runner_server::PROCESS_ENVIRONMENT_VARIABLE).is_some() {
        let args: model_runner_server::ModelRunnerProcessArgs = argh::from_env();
        let _trace_guard = args
            .trace_directory
            .as_deref()
            .map(initialize_trace_export)
            .transpose()?;
        model_runner_server::run(args).await
    } else if std::env::var_os(request_handler_server::PROCESS_ENVIRONMENT_VARIABLE).is_some() {
        let args: request_handler_server::RequestHandlerProcessArgs = argh::from_env();
        request_handler_server::run(args).await
    } else {
        let args = main_process::cli::parse();
        main_process::supervisor::run(args).await
    }
}

fn initialize_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}

fn initialize_trace_export(trace_directory: &Path) -> anyhow::Result<FlushGuard> {
    std::fs::create_dir_all(trace_directory).with_context(|| {
        format!(
            "failed to create trace directory {}",
            trace_directory.display()
        )
    })?;
    let trace_path = trace_directory.join(format!("model-runner-{}.json", std::process::id()));
    let trace_file = File::create(&trace_path)
        .with_context(|| format!("failed to create trace file {}", trace_path.display()))?;
    let (chrome_layer, guard) = ChromeLayerBuilder::new()
        .writer(trace_file)
        .include_locations(false)
        .build();
    let subscriber = tracing_subscriber::registry().with(chrome_layer);
    tracing::subscriber::set_global_default(subscriber)
        .context("failed to initialize Chrome trace export")?;
    info!("Writing Chrome trace to {}", trace_path.display());
    Ok(guard)
}
