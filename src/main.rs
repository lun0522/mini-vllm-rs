mod model_loaders;
mod model_runner;
mod proto;
mod utils;

use crate::model_loaders::model_downloader::ModelDownloader;
use crate::model_runner::ModelRunner;
use crate::proto::model_runner_command;
use crate::proto::GenerateText;
use crate::proto::ModelRunnerCommand;
use anyhow::Result;
use argh::FromArgs;
use env_logger::Env;
use log::error;
use log::info;
use prost::Message;

/// runs text generation with a model from Hugging Face
#[derive(FromArgs)]
struct CliArgs {
    /// model repository to load; supported examples are
    /// HuggingFaceTB/SmolLM2-360M-Instruct and Qwen/Qwen2.5-0.5B-Instruct;
    /// defaults to Qwen/Qwen2.5-0.5B-Instruct
    #[argh(option, default = "default_model_id()")]
    model_id: String,
}

fn default_model_id() -> String {
    "Qwen/Qwen2.5-0.5B-Instruct".to_owned()
}

struct ModelSettings {
    /// Hugging Face model repository to download and load.
    model_id: String,
    /// Model repository revision, such as a branch, tag, or commit hash.
    model_revision: String,
}

fn main() {
    initialize_logging();

    let args: CliArgs = argh::from_env();
    let model_settings = ModelSettings {
        model_id: args.model_id,
        model_revision: "main".to_owned(),
    };
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

    if let Err(err) = run_inference(model_settings, generate_text) {
        error!("{err:#}");
        std::process::exit(1);
    }
}

fn run_inference(model_settings: ModelSettings, generate_text: GenerateText) -> Result<()> {
    log_inference_config(&model_settings, &generate_text);
    let model_files = ModelDownloader::new(
        model_settings.model_id.clone(),
        model_settings.model_revision.clone(),
    )
    .download()?;
    let mut model_runner = ModelRunner::new(&model_files)?;
    let command = ModelRunnerCommand {
        command: Some(model_runner_command::Command::GenerateText(generate_text)),
    };
    model_runner.handle_command(&command.encode_to_vec())
}

fn initialize_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}

fn log_inference_config(model_settings: &ModelSettings, command: &GenerateText) {
    info!(
        "Inference configuration:\n\
         Model: {}\n\
         Revision: {}\n\
         Prompt: {}\n\
         Maximum new tokens: {}\n\
         Decoding: greedy\n\
         Repetition penalty: {} (last {} tokens)\n\
         Stream output: {}",
        model_settings.model_id,
        model_settings.model_revision,
        command.prompt,
        command.max_new_tokens,
        command.repeat_penalty,
        command.repeat_last_n,
        command.stream_output
    );
}
