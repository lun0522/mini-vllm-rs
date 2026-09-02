use crate::model_loaders::CausalLanguageModel;
use crate::model_loaders::KvCache;
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

use super::ModelAndKvCache;

struct GenerationParameters {
    max_new_tokens: usize,
    repeat_last_n: usize,
    repeat_penalty: f32,
}

impl GenerationParameters {
    fn from_command(command: &GenerateText) -> Result<Self> {
        Ok(Self {
            max_new_tokens: usize::try_from(command.max_new_tokens)
                .context("max_new_tokens does not fit in usize")?,
            repeat_last_n: usize::try_from(command.repeat_last_n)
                .context("repeat_last_n does not fit in usize")?,
            repeat_penalty: command.repeat_penalty,
        })
    }
}

pub(super) fn generate_text(
    tokenizer: &Tokenizer,
    target: &mut ModelAndKvCache,
    draft: Option<&mut ModelAndKvCache>,
    draft_token_count: usize,
    command: &GenerateText,
    mut push_fragment: impl FnMut(&str) -> Result<()>,
    mut is_cancelled: impl FnMut() -> bool,
) -> Result<TextGenerationStats> {
    let parameters = GenerationParameters::from_command(command)?;
    let model_prompt = target.model.format_chat_prompt(&command.prompt);
    let end_of_sequence_tokens = target.model.end_of_sequence_tokens();
    let device = target.model.device().clone();
    let mut tokens = tokenize_prompt(tokenizer, model_prompt)?;
    let prompt_token_count = tokens.len();
    let eos_tokens = resolve_end_of_sequence_tokens(tokenizer, end_of_sequence_tokens);
    let mut logits_processor = LogitsProcessor::new(0, None, None);
    let mut token_stream = tokenizer.decode_stream(true);

    let generation_started = Instant::now();
    let mut prefill_finished = None;
    if let Some(draft) = draft {
        let _draft_tokens = generate_draft_tokens(
            draft,
            &tokens,
            draft_token_count.min(parameters.max_new_tokens),
            &parameters,
            &eos_tokens,
            &mut is_cancelled,
        )?;
    }
    let model = target.model.model();
    for step in 0..parameters.max_new_tokens {
        if is_cancelled() {
            anyhow::bail!("generation request was cancelled");
        }
        let next_token = sample_next_token(
            model,
            &device,
            target.kv_cache.as_mut(),
            &tokens,
            &parameters,
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

fn generate_draft_tokens(
    draft: &mut ModelAndKvCache,
    tokens: &[u32],
    draft_token_count: usize,
    parameters: &GenerationParameters,
    eos_tokens: &[u32],
    is_cancelled: &mut impl FnMut() -> bool,
) -> Result<Vec<u32>> {
    let device = draft.model.device().clone();
    let model = draft.model.model();
    let mut draft_context = tokens.to_vec();
    let mut draft_tokens = Vec::with_capacity(draft_token_count);
    let mut logits_processor = LogitsProcessor::new(0, None, None);
    for step in 0..draft_token_count {
        if is_cancelled() {
            anyhow::bail!("generation request was cancelled");
        }
        let next_token = sample_next_token(
            model,
            &device,
            draft.kv_cache.as_mut(),
            &draft_context,
            parameters,
            step,
            &mut logits_processor,
        )?;
        draft_tokens.push(next_token);
        if eos_tokens.contains(&next_token) {
            break;
        }
        draft_context.push(next_token);
    }
    Ok(draft_tokens)
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
    kv_cache: &mut dyn KvCache,
    tokens: &[u32],
    parameters: &GenerationParameters,
    step: usize,
    logits_processor: &mut LogitsProcessor,
) -> Result<u32> {
    // Feed the full prompt once, then one token at a time. The engine-owned KV
    // cache retains attention state between these calls.
    let context_size = if step == 0 { tokens.len() } else { 1 };
    let start_pos = tokens.len() - context_size;
    let input = Tensor::new(&tokens[start_pos..], device)?.unsqueeze(0)?;
    let logits = model.forward(&input, start_pos, kv_cache)?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let repeat_start = tokens.len().saturating_sub(parameters.repeat_last_n);
    let logits = apply_repeat_penalty(&logits, parameters.repeat_penalty, &tokens[repeat_start..])?;
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
