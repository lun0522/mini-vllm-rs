mod model_loaders;
mod model_runner;
mod proto;
mod request_handler;
mod utils;

use crate::model_loaders::model_downloader::ModelDownloader;
use crate::model_runner::process::ModelRunnerProcess;
use crate::model_runner::server as model_runner_server;
use crate::proto::generate_text_event;
use crate::proto::GenerateText;
use crate::proto::GenerateTextEvent;
use crate::proto::TextGenerationStats;
use crate::request_handler::process::RequestHandlerProcess;
use crate::request_handler::server as request_handler_server;
use crate::utils::generated_text_output::create_generated_text_output;
use crate::utils::generated_text_output::GeneratedTextOutput;
use anyhow::Context;
use anyhow::Result;
use argh::FromArgs;
use env_logger::Env;
use log::error;
use log::info;
use log::warn;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

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
    /// unix domain socket exposed to local inference clients
    #[argh(option, default = "default_request_socket()")]
    request_socket: PathBuf,
    /// submit the built-in example request after startup
    #[argh(switch)]
    run_example: bool,
}

fn default_model_id() -> String {
    "Qwen/Qwen2.5-0.5B-Instruct".to_owned()
}

fn default_model_revision() -> String {
    "main".to_owned()
}

fn default_request_socket() -> PathBuf {
    PathBuf::from("/tmp/mini-vllm-rs.sock")
}

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
        let args: MainProcessArgs = argh::from_env();
        run_main_process(args).await
    };
    if let Err(err) = result {
        error!("{err:#}");
        std::process::exit(1);
    }
}

async fn run_main_process(args: MainProcessArgs) -> Result<()> {
    info!(
        "Server configuration:\n\
         Model: {}\n\
         Revision: {}\n\
         Request socket: {}\n\
         Run example: {}",
        args.model_id,
        args.model_revision,
        args.request_socket.display(),
        args.run_example
    );
    let model_downloader = ModelDownloader::new(args.model_id, args.model_revision);
    let model_files = model_downloader.download()?;
    let model_runner_process = ModelRunnerProcess::start(&model_files).await?;
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
        if args.run_example {
            let generate_text = create_example_request();
            info!("Submitting built-in example request: {generate_text:?}");
            if run_example(&request_handler_process, generate_text, ctrl_c.as_mut()).await? {
                return Ok(());
            }
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

async fn run_example(
    request_handler_process: &RequestHandlerProcess,
    generate_text: GenerateText,
    mut ctrl_c: Pin<&mut impl Future<Output = std::io::Result<()>>>,
) -> Result<bool> {
    let stream_output = generate_text.stream_output;
    let mut stream = request_handler_process.generate_text(generate_text).await?;
    let generated_text_output = create_generated_text_output(stream_output);
    generated_text_output.start();
    let mut generated_text_output = Some(generated_text_output);
    let stream_result = loop {
        tokio::select! {
            event = stream.message() => {
                match event {
                    Ok(Some(event)) => {
                        let output = generated_text_output
                            .take()
                            .context("received generation event after final statistics")?;
                        generated_text_output = handle_generation_event(
                            event,
                            output,
                        )?;
                        if generated_text_output.is_none() {
                            break Ok(false);
                        }
                    }
                    Ok(None) => break Ok(false),
                    Err(error) => {
                        break Err(error).context("generation response stream failed");
                    }
                }
            }
            signal = ctrl_c.as_mut() => {
                break signal
                    .context("failed to listen for Ctrl-C")
                    .map(|()| true);
            }
        }
    };
    if let Some(generated_text_output) = generated_text_output {
        generated_text_output.finish()?;
        warn!("Generation ended without receiving final statistics!");
    }
    stream_result
}

fn handle_generation_event(
    event: GenerateTextEvent,
    mut generated_text_output: Box<dyn GeneratedTextOutput>,
) -> Result<Option<Box<dyn GeneratedTextOutput>>> {
    let event = event.event.context("generation event is empty")?;
    match event {
        generate_text_event::Event::Text(text) => {
            generated_text_output.push_fragment(&text)?;
            Ok(Some(generated_text_output))
        }
        generate_text_event::Event::Stats(stats) => {
            generated_text_output.finish()?;
            log_generation_stats(&stats);
            Ok(None)
        }
    }
}

fn log_generation_stats(stats: &TextGenerationStats) {
    let prefill_tokens_per_second =
        format_tokens_per_second(stats.input_token_count, stats.prefill_duration_milliseconds);
    let decode_tokens_per_second = format_tokens_per_second(
        stats.output_token_count.saturating_sub(1),
        stats.decode_duration_milliseconds,
    );
    info!(
        "Generated {} tokens \
        (Prefill: {prefill_tokens_per_second} tokens/s, \
        Decode: {decode_tokens_per_second} tokens/s)",
        stats.output_token_count,
    );
}

fn format_tokens_per_second(token_count: u64, duration_milliseconds: u64) -> String {
    if duration_milliseconds == 0 {
        "unavailable".to_owned()
    } else {
        format!(
            "{:.2}",
            token_count as f64 * 1_000.0 / duration_milliseconds as f64
        )
    }
}

fn initialize_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}

fn create_example_request() -> GenerateText {
    GenerateText {
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
    }
}
