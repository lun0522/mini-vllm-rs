mod model_loaders;
mod utils;

use crate::model_loaders::loaded_model::LoadedModel;
use crate::utils::generated_text_output::create_generated_text_output;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use env_logger::Env;
use log::error;
use log::info;
use std::time::Instant;

struct InferenceSettings {
    /// Hugging Face model repository to download and load.
    model_id: String,
    /// Model repository revision, such as a branch, tag, or commit hash.
    model_revision: String,
    /// User text inserted into the model's chat template.
    prompt: String,
    /// Maximum number of tokens the model may generate before stopping.
    max_new_tokens: usize,
    /// Penalty applied to previously seen tokens; `1.0` disables it.
    repeat_penalty: f32,
    /// Number of recent tokens considered when applying the repetition penalty.
    repeat_last_n: usize,
    /// Whether generated text is printed incrementally instead of buffered.
    stream_output: bool,
}

fn main() {
    initialize_logging();

    let settings = InferenceSettings {
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_owned(),
        model_revision: "main".to_owned(),
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

    if let Err(err) = run_inference(&settings) {
        error!("{err:#}");
        std::process::exit(1);
    }
}

fn run_inference(settings: &InferenceSettings) -> Result<()> {
    let device = get_inference_device()?;
    let mut loaded_model = LoadedModel::new(
        settings.model_id.as_str(),
        settings.model_revision.as_str(),
        device,
    )?;
    log_inference_config(settings, loaded_model.device(), loaded_model.dtype());
    generate_text(settings, &mut loaded_model)
}

fn initialize_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}

fn generate_text(settings: &InferenceSettings, loaded_model: &mut LoadedModel) -> Result<()> {
    let (model, tokenizer, device) = loaded_model.start_inference();
    let model_prompt = format_chat_prompt(&settings.prompt);
    let encoding = tokenizer
        .encode(model_prompt, true)
        .map_err(anyhow::Error::msg)
        .context("failed to tokenize the prompt")?;
    let mut tokens = encoding.get_ids().to_vec();
    let prompt_token_count = tokens.len();
    let eos_tokens: Vec<u32> = ["<|endoftext|>", "<|im_end|>"]
        .iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect();
    let mut logits_processor = LogitsProcessor::new(0, None, None);
    let mut token_stream = tokenizer.decode_stream(true);
    let mut generated_text_output = create_generated_text_output(settings.stream_output);

    let generation_started = Instant::now();
    generated_text_output.start();
    for step in 0..settings.max_new_tokens {
        // Feed the full prompt once, then one token at a time. Qwen's Candle
        // implementation retains the KV cache between these calls.
        let context_size = if step == 0 { tokens.len() } else { 1 };
        let start_pos = tokens.len() - context_size;
        let input = Tensor::new(&tokens[start_pos..], device)?.unsqueeze(0)?;
        let logits = model.forward(&input, start_pos)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let repeat_start = tokens.len().saturating_sub(settings.repeat_last_n);
        let logits =
            apply_repeat_penalty(&logits, settings.repeat_penalty, &tokens[repeat_start..])?;
        let next_token = logits_processor.sample(&logits)?;

        if eos_tokens.contains(&next_token) {
            break;
        }
        tokens.push(next_token);
        if let Some(fragment) = token_stream
            .step(next_token)
            .map_err(anyhow::Error::msg)
            .context("failed to decode generated token")?
        {
            generated_text_output.push_fragment(&fragment)?;
        }
    }
    generated_text_output.finish()?;

    let token_count = tokens.len() - prompt_token_count;
    let elapsed_time = generation_started.elapsed();
    let tokens_per_second = token_count as f64 / elapsed_time.as_secs_f64();
    info!(
        "Generated {token_count} tokens in {elapsed_time:.2?} \
         ({tokens_per_second:.2} tokens/s)"
    );

    Ok(())
}

fn format_chat_prompt(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn log_inference_config(settings: &InferenceSettings, device: &Device, dtype: DType) {
    info!(
        "Inference configuration:\n\
         Model: {}\n\
         Revision: {}\n\
         Device: {device:?}\n\
         Data type: {dtype:?}\n\
         Prompt: {}\n\
         Maximum new tokens: {}\n\
         Decoding: greedy\n\
         Repetition penalty: {} (last {} tokens)\n\
         Stream output: {}",
        settings.model_id,
        settings.model_revision,
        settings.prompt,
        settings.max_new_tokens,
        settings.repeat_penalty,
        settings.repeat_last_n,
        settings.stream_output
    );
}

fn get_inference_device() -> Result<Device> {
    #[cfg(feature = "metal")]
    {
        Device::new_metal(0).context("failed to initialize the Metal device")
    }

    #[cfg(not(feature = "metal"))]
    {
        log::warn!("Metal support is disabled; CPU inference will be slower");
        Ok(Device::Cpu)
    }
}
