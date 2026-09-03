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

struct TextGenerator<PushToken, IsCancelled> {
    device: Device,
    tokens: Vec<u32>,
    max_new_token_count: usize,
    repeat_last_n: usize,
    repeat_penalty: f32,
    eos_tokens: Vec<u32>,
    logits_processor: LogitsProcessor,
    push_token: PushToken,
    is_cancelled: IsCancelled,
}

pub(super) fn generate_text(
    tokenizer: &Tokenizer,
    target: &ModelAndKvCache,
    draft: Option<&ModelAndKvCache>,
    _draft_token_count: usize,
    command: &GenerateText,
    mut push_fragment: impl FnMut(&str) -> Result<()>,
    is_cancelled: impl FnMut() -> bool,
) -> Result<TextGenerationStats> {
    let target_model = target.model.borrow();
    let model_prompt = target_model.format_chat_prompt(&command.prompt);
    let device = target_model.device().clone();
    let eos_tokens =
        resolve_end_of_sequence_tokens(tokenizer, target_model.end_of_sequence_tokens());
    drop(target_model);

    let tokens = tokenize_prompt(tokenizer, model_prompt)?;
    let mut token_stream = tokenizer.decode_stream(true);
    let push_token = |next_token| {
        if let Some(fragment) = token_stream
            .step(next_token)
            .map_err(anyhow::Error::msg)
            .context("failed to decode generated token")?
        {
            push_fragment(&fragment)?;
        }
        Ok(())
    };

    TextGenerator {
        device,
        tokens,
        max_new_token_count: usize::try_from(command.max_new_tokens)
            .context("max_new_tokens does not fit in usize")?,
        repeat_last_n: usize::try_from(command.repeat_last_n)
            .context("repeat_last_n does not fit in usize")?,
        repeat_penalty: command.repeat_penalty,
        eos_tokens,
        logits_processor: LogitsProcessor::new(0, None, None),
        push_token,
        is_cancelled,
    }
    .run(target, draft)
}

impl<PushToken, IsCancelled> TextGenerator<PushToken, IsCancelled>
where
    PushToken: FnMut(u32) -> Result<()>,
    IsCancelled: FnMut() -> bool,
{
    fn run(
        mut self,
        target: &ModelAndKvCache,
        draft: Option<&ModelAndKvCache>,
    ) -> Result<TextGenerationStats> {
        let prompt_token_count = self.tokens.len();
        let generation_started = Instant::now();
        let mut prefill_finished = None;
        if self.max_new_token_count > 0 {
            let should_decode = self.run_prefill_phase(target, draft)?;
            prefill_finished = Some(Instant::now());
            if should_decode {
                self.run_decode_phase(target)?;
            }
        }
        let decode_finished = Instant::now();
        let prefill_finished = prefill_finished.unwrap_or(decode_finished);
        create_generation_stats(
            prompt_token_count,
            self.tokens.len(),
            generation_started,
            prefill_finished,
            decode_finished,
        )
    }

    fn run_prefill_phase(
        &mut self,
        target: &ModelAndKvCache,
        draft: Option<&ModelAndKvCache>,
    ) -> Result<bool> {
        if (self.is_cancelled)() {
            anyhow::bail!("generation request was cancelled");
        }
        if let Some(draft) = draft {
            let input =
                Tensor::new(&self.tokens[..], draft.model.borrow().device())?.unsqueeze(0)?;
            draft.forward(&input, 0)?;
        }
        let next_token = self.sample_next_token(target, /* start_position */ 0)?;
        self.commit_next_token(next_token)
    }

    fn run_decode_phase(&mut self, target: &ModelAndKvCache) -> Result<()> {
        for _ in 1..self.max_new_token_count {
            if (self.is_cancelled)() {
                anyhow::bail!("generation request was cancelled");
            }
            let start_position = self.tokens.len() - 1;
            let next_token = self.sample_next_token(target, start_position)?;
            if !self.commit_next_token(next_token)? {
                break;
            }
        }
        Ok(())
    }

    fn sample_next_token(
        &mut self,
        target: &ModelAndKvCache,
        start_position: usize,
    ) -> Result<u32> {
        let input = Tensor::new(&self.tokens[start_position..], &self.device)?.unsqueeze(0)?;
        let logits = target.forward(&input, start_position)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let repeat_start = self.tokens.len().saturating_sub(self.repeat_last_n);
        let logits =
            apply_repeat_penalty(&logits, self.repeat_penalty, &self.tokens[repeat_start..])?;
        self.logits_processor.sample(&logits).map_err(Into::into)
    }

    fn commit_next_token(&mut self, next_token: u32) -> Result<bool> {
        if self.eos_tokens.contains(&next_token) {
            return Ok(false);
        }
        self.tokens.push(next_token);
        (self.push_token)(next_token)?;
        Ok(true)
    }
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
