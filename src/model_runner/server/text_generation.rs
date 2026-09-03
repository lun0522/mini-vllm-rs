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
    _draft_token_count: usize,
    command: &GenerateText,
    mut push_fragment: impl FnMut(&str) -> Result<()>,
    mut is_cancelled: impl FnMut() -> bool,
) -> Result<TextGenerationStats> {
    let parameters = GenerationParameters::from_command(command)?;
    let model_prompt = target.model.format_chat_prompt(&command.prompt);
    let device = target.model.device().clone();
    let mut tokens = tokenize_prompt(tokenizer, model_prompt)?;
    let prompt_token_count = tokens.len();
    let eos_tokens =
        resolve_end_of_sequence_tokens(tokenizer, target.model.end_of_sequence_tokens());
    let mut logits_processor = LogitsProcessor::new(0, None, None);

    let mut token_stream = tokenizer.decode_stream(true);
    let mut push_token = |next_token| {
        if let Some(fragment) = token_stream
            .step(next_token)
            .map_err(anyhow::Error::msg)
            .context("failed to decode generated token")?
        {
            push_fragment(&fragment)?;
        }
        Ok(())
    };

    let generation_started = Instant::now();
    let mut prefill_finished = None;
    if parameters.max_new_tokens > 0 {
        let should_decode = run_prefill_phase(
            target,
            draft,
            &device,
            &mut tokens,
            &parameters,
            &eos_tokens,
            &mut logits_processor,
            &mut push_token,
            &mut is_cancelled,
        )?;
        prefill_finished = Some(Instant::now());
        if should_decode {
            run_decode_phase(
                target,
                &device,
                &mut tokens,
                parameters.max_new_tokens - 1,
                &parameters,
                &eos_tokens,
                &mut logits_processor,
                &mut push_token,
                &mut is_cancelled,
            )?;
        }
    }

    let decode_finished = Instant::now();
    let prefill_finished = prefill_finished.unwrap_or(decode_finished);
    create_generation_stats(
        prompt_token_count,
        tokens.len(),
        generation_started,
        prefill_finished,
        decode_finished,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_prefill_phase(
    target: &mut ModelAndKvCache,
    draft: Option<&mut ModelAndKvCache>,
    device: &Device,
    tokens: &mut Vec<u32>,
    parameters: &GenerationParameters,
    eos_tokens: &[u32],
    logits_processor: &mut LogitsProcessor,
    push_token: &mut impl FnMut(u32) -> Result<()>,
    is_cancelled: &mut impl FnMut() -> bool,
) -> Result<bool> {
    if is_cancelled() {
        anyhow::bail!("generation request was cancelled");
    }
    if let Some(draft) = draft {
        let input = Tensor::new(&tokens[..], draft.model.device())?.unsqueeze(0)?;
        draft
            .model
            .model()
            .forward(&input, 0, draft.kv_cache.as_mut())?;
    }
    let next_token = sample_next_token(
        target,
        device,
        tokens,
        /* start_position */ 0,
        parameters,
        logits_processor,
    )?;
    commit_next_token(next_token, eos_tokens, tokens, push_token)
}

#[allow(clippy::too_many_arguments)]
fn run_decode_phase(
    target: &mut ModelAndKvCache,
    device: &Device,
    tokens: &mut Vec<u32>,
    max_decode_token_count: usize,
    parameters: &GenerationParameters,
    eos_tokens: &[u32],
    logits_processor: &mut LogitsProcessor,
    push_token: &mut impl FnMut(u32) -> Result<()>,
    is_cancelled: &mut impl FnMut() -> bool,
) -> Result<()> {
    for _ in 0..max_decode_token_count {
        if is_cancelled() {
            anyhow::bail!("generation request was cancelled");
        }
        let start_position = tokens.len() - 1;
        let next_token = sample_next_token(
            target,
            device,
            tokens,
            start_position,
            parameters,
            logits_processor,
        )?;
        if !commit_next_token(next_token, eos_tokens, tokens, push_token)? {
            break;
        }
    }
    Ok(())
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
    target: &mut ModelAndKvCache,
    device: &Device,
    tokens: &[u32],
    start_position: usize,
    parameters: &GenerationParameters,
    logits_processor: &mut LogitsProcessor,
) -> Result<u32> {
    let input = Tensor::new(&tokens[start_position..], device)?.unsqueeze(0)?;
    let logits = target
        .model
        .model()
        .forward(&input, start_position, target.kv_cache.as_mut())?;
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    let repeat_start = tokens.len().saturating_sub(parameters.repeat_last_n);
    let logits = apply_repeat_penalty(&logits, parameters.repeat_penalty, &tokens[repeat_start..])?;
    logits_processor.sample(&logits).map_err(Into::into)
}

fn commit_next_token(
    next_token: u32,
    eos_tokens: &[u32],
    tokens: &mut Vec<u32>,
    push_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<bool> {
    if eos_tokens.contains(&next_token) {
        return Ok(false);
    }
    tokens.push(next_token);
    push_token(next_token)?;
    Ok(true)
}

fn create_generation_stats(
    prompt_token_count: usize,
    total_token_count: usize,
    generation_started: Instant,
    prefill_finished: Instant,
    decode_finished: Instant,
) -> Result<TextGenerationStats> {
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
