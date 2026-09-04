use crate::proto::model_runner::TextGenerationStats;
use crate::proto::request_handler::generate_text_event;
use crate::proto::request_handler::GenerateText;
use crate::proto::request_handler::GenerateTextEvent;
use crate::request_handler::client::RequestHandlerProcess;
use crate::utils::generated_text_output::create_generated_text_output;
use crate::utils::generated_text_output::GeneratedTextOutput;
use anyhow::Context;
use anyhow::Result;
use log::info;
use log::warn;
use std::future::Future;
use std::pin::Pin;

pub(crate) async fn run_example(
    request_handler_process: &RequestHandlerProcess,
    mut ctrl_c: Pin<&mut impl Future<Output = std::io::Result<()>>>,
) -> Result<bool> {
    let generate_text = create_example_request();
    info!("Submitting built-in example request: {generate_text:?}");
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
                        generated_text_output = handle_generation_event(event, output)?;
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
    let draft_acceptance = stats
        .draft_token_acceptance_rate
        .map(|rate| format!(", Draft acceptance: {:.1}%", rate * 100.0))
        .unwrap_or_default();
    info!(
        "Generated {} tokens \
        (Prefill: {prefill_tokens_per_second} tokens/s, \
        Decode: {decode_tokens_per_second} tokens/s{draft_acceptance})",
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
