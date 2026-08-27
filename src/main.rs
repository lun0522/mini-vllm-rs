mod model_loaders;
mod model_runner;
mod proto;
mod utils;

use crate::model_loaders::model_downloader::ModelDownloader;
use crate::model_runner::process::ModelRunnerProcess;
use crate::model_runner::server;
use crate::proto::model_runner_command;
use crate::proto::GenerateText;
use crate::proto::ModelRunnerCommand;
use anyhow::Result;
use argh::FromArgs;
use env_logger::Env;
use log::error;
use log::info;

/// runs text generation with a model from Hugging Face
#[derive(FromArgs)]
struct MainProcessArgs {
    /// model repository to load; supported examples are
    /// HuggingFaceTB/SmolLM2-360M-Instruct and Qwen/Qwen2.5-0.5B-Instruct;
    /// defaults to Qwen/Qwen2.5-0.5B-Instruct
    #[argh(option, default = "default_model_id()")]
    model_id: String,
    /// model repository revision, such as a branch, tag, or commit hash;
    /// defaults to main
    #[argh(option, default = "default_model_revision()")]
    model_revision: String,
}

fn default_model_id() -> String {
    "Qwen/Qwen2.5-0.5B-Instruct".to_owned()
}

fn default_model_revision() -> String {
    "main".to_owned()
}

#[tokio::main]
async fn main() {
    initialize_logging();

    let result = if std::env::var_os(server::PROCESS_ENVIRONMENT_VARIABLE).is_some() {
        let args: server::ModelRunnerProcessArgs = argh::from_env();
        server::run(args).await
    } else {
        let args: MainProcessArgs = argh::from_env();
        run_main_process(args).await
    };
    if let Err(err) = result {
        error!("{err:#}");
        std::process::exit(1);
    }
}

async fn run_main_process(args: MainProcessArgs) -> Result<()> {
    let generate_text = GenerateText {
        prompt: concat!(
            "Explain in detail how continuous batching improves throughput in an LLM ",
            "inference server. Compare it with static batching, describe how requests ",
            "enter and leave a running batch, and discuss the key scheduling and KV-cache ",
            "challenges an implementation must handle."
        )
        .to_owned(),
        max_new_tokens: 1024,
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        stream_output: true,
    };

    info!(
        "Inference configuration:\n\
         Model: {}\n\
         Revision: {}\n\
         Generate text: {generate_text:?}",
        args.model_id, args.model_revision
    );
    let model_downloader = ModelDownloader::new(args.model_id, args.model_revision);
    let model_files = model_downloader.download()?;
    let model_runner_process = ModelRunnerProcess::start(&model_files).await?;

    let inference_result = run_inference(&model_runner_process, generate_text).await;
    let shutdown_result = model_runner_process.shutdown().await;
    inference_result?;
    shutdown_result
}

fn initialize_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}

async fn run_inference(
    model_runner_process: &ModelRunnerProcess,
    generate_text: GenerateText,
) -> Result<()> {
    let command = ModelRunnerCommand {
        command: Some(model_runner_command::Command::GenerateText(generate_text)),
    };
    model_runner_process.handle_command(command).await
}
