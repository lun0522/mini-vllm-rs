mod utils;

use crate::utils::generated_text_output::create_generated_text_output;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::Config;
use candle_transformers::models::qwen2::ModelForCausalLM;
use candle_transformers::utils::apply_repeat_penalty;
use env_logger::Env;
use hf_hub::api::sync::Api;
use hf_hub::Repo;
use hf_hub::RepoType;
use log::error;
use log::info;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

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

struct ModelFiles {
    tokenizer: PathBuf,
    config: PathBuf,
    weights: PathBuf,
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

    if let Err(err) = run(&settings) {
        error!("{err:#}");
        std::process::exit(1);
    }
}

fn run(settings: &InferenceSettings) -> Result<()> {
    let device = get_inference_device()?;
    let dtype = get_inference_dtype(&device);
    log_inference_config(settings, &device, dtype);

    let files = download_model_files(settings)?;
    let tokenizer = load_tokenizer(&files.tokenizer)?;
    let config = load_model_config(&files.config)?;
    let mut model = load_model(&config, &files.weights, dtype, &device)?;
    generate_text(settings, &mut model, &tokenizer, &device)
}

fn initialize_logging() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();
}

fn download_model_files(settings: &InferenceSettings) -> Result<ModelFiles> {
    // hf-hub stores downloads in its standard local cache, so subsequent runs
    // reuse these files rather than downloading the model again.
    let api = Api::new().context("failed to create the Hugging Face Hub client")?;
    let repo = api.repo(Repo::with_revision(
        settings.model_id.clone(),
        RepoType::Model,
        settings.model_revision.clone(),
    ));

    Ok(ModelFiles {
        tokenizer: repo
            .get("tokenizer.json")
            .context("failed to download tokenizer.json")?,
        config: repo
            .get("config.json")
            .context("failed to download config.json")?,
        weights: repo
            .get("model.safetensors")
            .context("failed to download model.safetensors")?,
    })
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(anyhow::Error::msg)
        .context("failed to load the tokenizer")
}

fn load_model_config(path: &Path) -> Result<Config> {
    let config_bytes = std::fs::read(path).context("failed to read the model config")?;
    serde_json::from_slice(&config_bytes).context("failed to parse the model config")
}

fn load_model(
    config: &Config,
    weights_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<ModelForCausalLM> {
    // SAFETY: The weights are immutable files in the Hugging Face cache, and this
    // process never modifies or truncates them while the memory-mapped model exists.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
            .context("failed to memory-map the model weights")?
    };
    ModelForCausalLM::new(config, vb).context("failed to load the model")
}

fn generate_text(
    settings: &InferenceSettings,
    model: &mut ModelForCausalLM,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<()> {
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

fn get_inference_dtype(device: &Device) -> DType {
    // The checkpoint stores BF16 tensors. Candle converts them to F32 on CPU,
    // while Metal can run them as BF16 to save memory and improve throughput.
    if device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
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
