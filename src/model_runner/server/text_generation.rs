use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::CausalLanguageModel;
use crate::proto::GenerateText;
use crate::proto::TextGenerationStats;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use std::time::Duration;
use std::time::Instant;
use tokenizers::Tokenizer;

struct GenerationParameters {
    max_new_tokens: usize,
    repeat_last_n: usize,
}

impl GenerationParameters {
    fn from_command(command: &GenerateText) -> Result<Self> {
        Ok(Self {
            max_new_tokens: usize::try_from(command.max_new_tokens)
                .context("max_new_tokens does not fit in usize")?,
            repeat_last_n: usize::try_from(command.repeat_last_n)
                .context("repeat_last_n does not fit in usize")?,
        })
    }
}

pub(super) fn generate_text(
    loaded_model: &mut LoadedModel,
    tokenizer: &Tokenizer,
    command: &GenerateText,
    mut push_fragment: impl FnMut(&str) -> Result<()>,
    mut is_cancelled: impl FnMut() -> bool,
) -> Result<TextGenerationStats> {
    let parameters = GenerationParameters::from_command(command)?;
    let model_prompt = loaded_model.format_chat_prompt(&command.prompt);
    let end_of_sequence_tokens = loaded_model.end_of_sequence_tokens();
    let (model, device) = loaded_model.start_inference();
    let mut tokens = tokenize_prompt(tokenizer, model_prompt)?;
    let prompt_token_count = tokens.len();
    let eos_tokens = resolve_end_of_sequence_tokens(tokenizer, end_of_sequence_tokens);
    let mut logits_processor = LogitsProcessor::new(0, None, None);
    let mut token_stream = tokenizer.decode_stream(true);

    let generation_started = Instant::now();
    let mut prefill_finished = None;
    for step in 0..parameters.max_new_tokens {
        if is_cancelled() {
            anyhow::bail!("generation request was cancelled");
        }
        let next_token = sample_next_token(
            model,
            device,
            &tokens,
            parameters.repeat_last_n,
            command.repeat_penalty,
            step,
            &mut logits_processor,
        )?;

        if eos_tokens.contains(&next_token) {
            if step == 0 {
                prefill_finished = Some(Instant::now());
            }
            break;
        }
        tokens.push(next_token);
        if let Some(fragment) = token_stream
            .step(next_token)
            .map_err(anyhow::Error::msg)
            .context("failed to decode generated token")?
        {
            push_fragment(&fragment)?;
        }
        if step == 0 {
            prefill_finished = Some(Instant::now());
        }
    }

    let decode_finished = Instant::now();
    create_generation_stats(
        prompt_token_count,
        tokens.len(),
        generation_started,
        prefill_finished,
        decode_finished,
    )
}

fn tokenize_prompt(tokenizer: &Tokenizer, model_prompt: String) -> Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(model_prompt, true)
        .map_err(anyhow::Error::msg)
        .context("failed to tokenize the prompt")?;
    Ok(encoding.get_ids().to_vec())
}

fn resolve_end_of_sequence_tokens(
    tokenizer: &Tokenizer,
    end_of_sequence_tokens: &[&str],
) -> Vec<u32> {
    end_of_sequence_tokens
        .iter()
        .filter_map(|token| tokenizer.token_to_id(token))
        .collect()
}

fn sample_next_token(
    model: &mut dyn CausalLanguageModel,
    device: &Device,
    tokens: &[u32],
    repeat_last_n: usize,
    repeat_penalty: f32,
    step: usize,
    logits_processor: &mut LogitsProcessor,
) -> Result<u32> {
    // Feed the full prompt once, then one token at a time. The model backend
    // retains the KV cache between these calls.
    let context_size = if step == 0 { tokens.len() } else { 1 };
    let start_pos = tokens.len() - context_size;
    let input = Tensor::new(&tokens[start_pos..], device)?.unsqueeze(0)?;
    let logits = model.forward(&input, start_pos)?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let repeat_start = tokens.len().saturating_sub(repeat_last_n);
    let logits = apply_repeat_penalty(&logits, repeat_penalty, &tokens[repeat_start..])?;
    logits_processor.sample(&logits).map_err(Into::into)
}

fn create_generation_stats(
    prompt_token_count: usize,
    total_token_count: usize,
    generation_started: Instant,
    prefill_finished: Option<Instant>,
    decode_finished: Instant,
) -> Result<TextGenerationStats> {
    let prefill_finished = prefill_finished.unwrap_or(decode_finished);
    let output_token_count = total_token_count - prompt_token_count;
    Ok(TextGenerationStats {
        input_token_count: u64::try_from(prompt_token_count)
            .context("input token count does not fit in u64")?,
        output_token_count: u64::try_from(output_token_count)
            .context("output token count does not fit in u64")?,
        prefill_duration_milliseconds: duration_milliseconds(
            prefill_finished.duration_since(generation_started),
        ),
        decode_duration_milliseconds: duration_milliseconds(
            decode_finished.duration_since(prefill_finished),
        ),
    })
}

fn duration_milliseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or_default()
}
