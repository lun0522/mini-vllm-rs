use crate::proto::GenerateText;
use crate::proto::TextGenerationStats;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::IndexOp;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use std::cell::RefCell;
use std::time::Duration;
use std::time::Instant;
use tokenizers::Tokenizer;

use super::ModelAndKvCache;

struct TextGenerator<PushToken, IsCancelled> {
    tokens: Vec<u32>,
    draft_token_count: usize,
    max_new_token_count: usize,
    repeat_last_n: usize,
    repeat_penalty: f32,
    eos_tokens: Vec<u32>,
    target_logits_processor: RefCell<LogitsProcessor>,
    draft_logits_processor: RefCell<LogitsProcessor>,
    accepted_draft_token_count: usize,
    proposed_draft_token_count: usize,
    push_token: PushToken,
    is_cancelled: IsCancelled,
}

struct DraftVerificationResult {
    accepted_token_count: usize,
    maybe_replacement_token: Option<u32>,
}

struct DecodeIterationResult {
    committed_token_count: usize,
    should_continue: bool,
}

pub(super) fn generate_text(
    tokenizer: &Tokenizer,
    target: &ModelAndKvCache,
    draft: Option<&ModelAndKvCache>,
    draft_token_count: usize,
    command: &GenerateText,
    mut push_fragment: impl FnMut(&str) -> Result<()>,
    is_cancelled: impl FnMut() -> bool,
) -> Result<TextGenerationStats> {
    let target_model = target.model.borrow();
    let model_prompt = target_model.format_chat_prompt(&command.prompt);
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
        tokens,
        draft_token_count,
        max_new_token_count: usize::try_from(command.max_new_tokens)
            .context("max_new_tokens does not fit in usize")?,
        repeat_last_n: usize::try_from(command.repeat_last_n)
            .context("repeat_last_n does not fit in usize")?,
        repeat_penalty: command.repeat_penalty,
        eos_tokens,
        target_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
        draft_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
        accepted_draft_token_count: 0,
        proposed_draft_token_count: 0,
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
                self.run_decode_phase(target, draft)?;
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
            self.compute_draft_token_acceptance_rate(),
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
            // Prefill the draft KV cache only. Draft-token proposals are generated during
            // decoding, so the prompt logits are not sampled here.
            draft.forward(&input, 0)?;
        }
        let next_token = self.sample_next_token(
            target,
            &self.tokens,
            /* start_position */ 0,
            /* appended_tokens */ &[],
            &mut self.target_logits_processor.borrow_mut(),
        )?;
        self.commit_next_token(next_token)
    }

    fn run_decode_phase(
        &mut self,
        target: &ModelAndKvCache,
        draft: Option<&ModelAndKvCache>,
    ) -> Result<()> {
        let mut generated_token_count = 1;
        while generated_token_count < self.max_new_token_count {
            if (self.is_cancelled)() {
                anyhow::bail!("generation request was cancelled");
            }
            let should_continue = match draft {
                Some(draft) => {
                    let DecodeIterationResult {
                        committed_token_count,
                        should_continue,
                    } = self.run_speculative_iteration(
                        target,
                        draft,
                        self.max_new_token_count - generated_token_count,
                    )?;
                    generated_token_count += committed_token_count;
                    should_continue
                }
                None => {
                    let start_position = self.tokens.len() - 1;
                    let next_token = self.sample_next_token(
                        target,
                        &self.tokens[start_position..],
                        start_position,
                        /* appended_tokens */ &[],
                        &mut self.target_logits_processor.borrow_mut(),
                    )?;
                    generated_token_count += 1;
                    self.commit_next_token(next_token)?
                }
            };
            if !should_continue {
                return Ok(());
            }
        }
        Ok(())
    }

    fn run_speculative_iteration(
        &mut self,
        target: &ModelAndKvCache,
        draft: &ModelAndKvCache,
        remaining_max_token_count: usize,
    ) -> Result<DecodeIterationResult> {
        // Generate draft proposals autoregressively, starting with the pending token that has
        // not yet been written to either model's KV cache.
        let original_cached_token_count = self.tokens.len() - 1;
        let draft_tokens = self.generate_draft_tokens(
            draft,
            original_cached_token_count,
            remaining_max_token_count,
        )?;

        // Run the target model once over the pending token and proposed prefix, producing one
        // set of verification logits for every draft token.
        let verification_logits = self.compute_draft_verification_logits(
            target,
            &draft_tokens,
            original_cached_token_count,
        )?;

        // Accept the longest prefix on which target and draft sampling agree. At the first
        // mismatch, retain the target token as the replacement output.
        let DraftVerificationResult {
            accepted_token_count,
            maybe_replacement_token,
        } = self.verify_draft_tokens(&draft_tokens, &verification_logits)?;
        self.accepted_draft_token_count += accepted_token_count;
        self.proposed_draft_token_count += draft_tokens.len();

        // Discard cache entries derived from rejected proposals and retain exactly the verified
        // input prefix needed for the next decode iteration.
        let replacement_token_count = usize::from(maybe_replacement_token.is_some());
        let retained_cached_token_count =
            original_cached_token_count + accepted_token_count + replacement_token_count;
        target.truncate(retained_cached_token_count)?;
        draft.truncate(retained_cached_token_count)?;

        // Publish the accepted draft prefix, followed by the target replacement when the models
        // disagreed. EOS is observed but never added to the generated output.
        self.commit_speculative_tokens(
            &draft_tokens[..accepted_token_count],
            maybe_replacement_token,
        )
    }

    fn generate_draft_tokens(
        &mut self,
        draft: &ModelAndKvCache,
        original_cached_token_count: usize,
        remaining_max_token_count: usize,
    ) -> Result<Vec<u32>> {
        // Limit the proposal batch to the request's remaining output budget.
        let draft_token_count = self.draft_token_count.min(remaining_max_token_count);
        let mut draft_tokens = Vec::with_capacity(draft_token_count);
        for _ in 0..draft_token_count {
            if (self.is_cancelled)() {
                anyhow::bail!("generation request was cancelled");
            }
            let start_position = original_cached_token_count + draft_tokens.len();
            let input_token = draft_tokens
                .last()
                .copied()
                .or_else(|| self.tokens.last().copied())
                .context("generation context is empty")?;
            let next_token = self.sample_next_token(
                draft,
                &[input_token],
                start_position,
                &draft_tokens,
                &mut self.draft_logits_processor.borrow_mut(),
            )?;
            draft_tokens.push(next_token);
            if self.eos_tokens.contains(&next_token) {
                break;
            }
        }
        Ok(draft_tokens)
    }

    fn compute_draft_verification_logits(
        &self,
        target: &ModelAndKvCache,
        draft_tokens: &[u32],
        start_position: usize,
    ) -> Result<Tensor> {
        let mut verification_tokens = Vec::with_capacity(draft_tokens.len());
        verification_tokens.push(*self.tokens.last().context("generation context is empty")?);
        verification_tokens.extend_from_slice(&draft_tokens[..draft_tokens.len() - 1]);
        let input =
            Tensor::new(&verification_tokens[..], target.model.borrow().device())?.unsqueeze(0)?;
        Ok(target
            .forward_for_speculative_verification(&input, start_position)?
            .squeeze(0)?
            .to_dtype(DType::F32)?)
    }

    fn verify_draft_tokens(
        &mut self,
        draft_tokens: &[u32],
        verification_logits: &Tensor,
    ) -> Result<DraftVerificationResult> {
        let mut accepted_token_count = 0;
        for (position, &draft_token) in draft_tokens.iter().enumerate() {
            let target_token = self.sample_logits(
                &verification_logits.i(position)?,
                &draft_tokens[..position],
                &mut self.target_logits_processor.borrow_mut(),
            )?;
            if target_token != draft_token {
                return Ok(DraftVerificationResult {
                    accepted_token_count,
                    maybe_replacement_token: Some(target_token),
                });
            }
            accepted_token_count += 1;
            if self.eos_tokens.contains(&draft_token) {
                break;
            }
        }
        Ok(DraftVerificationResult {
            accepted_token_count,
            maybe_replacement_token: None,
        })
    }

    fn commit_speculative_tokens(
        &mut self,
        accepted_draft_tokens: &[u32],
        maybe_replacement_token: Option<u32>,
    ) -> Result<DecodeIterationResult> {
        let mut committed_token_count = 0;
        for &draft_token in accepted_draft_tokens {
            if !self.commit_next_token(draft_token)? {
                return Ok(DecodeIterationResult {
                    committed_token_count,
                    should_continue: false,
                });
            }
            committed_token_count += 1;
        }
        if let Some(replacement_token) = maybe_replacement_token {
            if !self.commit_next_token(replacement_token)? {
                return Ok(DecodeIterationResult {
                    committed_token_count,
                    should_continue: false,
                });
            }
            committed_token_count += 1;
        }
        Ok(DecodeIterationResult {
            committed_token_count,
            should_continue: true,
        })
    }

    fn compute_draft_token_acceptance_rate(&self) -> Option<f32> {
        if self.proposed_draft_token_count == 0 {
            return None;
        }
        Some(self.accepted_draft_token_count as f32 / self.proposed_draft_token_count as f32)
    }

    fn sample_next_token(
        &self,
        model: &ModelAndKvCache,
        input_tokens: &[u32],
        start_position: usize,
        appended_tokens: &[u32],
        logits_processor: &mut LogitsProcessor,
    ) -> Result<u32> {
        let input = Tensor::new(input_tokens, model.model.borrow().device())?.unsqueeze(0)?;
        let logits = model.forward(&input, start_position)?;
        let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        self.sample_logits(&logits, appended_tokens, logits_processor)
    }

    fn sample_logits(
        &self,
        logits: &Tensor,
        appended_tokens: &[u32],
        logits_processor: &mut LogitsProcessor,
    ) -> Result<u32> {
        // Collect the repetition window from committed and speculative tokens.
        let total_token_count = self.tokens.len() + appended_tokens.len();
        let repeat_start = total_token_count.saturating_sub(self.repeat_last_n);
        let mut repeat_tokens = Vec::with_capacity(total_token_count - repeat_start);
        if repeat_start < self.tokens.len() {
            repeat_tokens.extend_from_slice(&self.tokens[repeat_start..]);
        }
        let appended_start = repeat_start.saturating_sub(self.tokens.len());
        repeat_tokens.extend_from_slice(&appended_tokens[appended_start..]);

        // Apply the repetition penalty before sampling the next token.
        let logits = apply_repeat_penalty(logits, self.repeat_penalty, &repeat_tokens)?;
        logits_processor.sample(&logits).map_err(Into::into)
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
    draft_token_acceptance_rate: Option<f32>,
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
        draft_token_acceptance_rate,
    })
}

fn duration_milliseconds(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    type TestGenerator = TextGenerator<fn(u32) -> Result<()>, fn() -> bool>;

    fn create_test_generator(tokens: Vec<u32>, eos_tokens: Vec<u32>) -> TestGenerator {
        fn ignore_token(_: u32) -> Result<()> {
            Ok(())
        }

        fn never_cancelled() -> bool {
            false
        }

        TextGenerator {
            tokens,
            draft_token_count: 4,
            max_new_token_count: 16,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            eos_tokens,
            target_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
            draft_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
            accepted_draft_token_count: 0,
            proposed_draft_token_count: 0,
            push_token: ignore_token,
            is_cancelled: never_cancelled,
        }
    }

    #[test]
    fn verifies_the_accepted_draft_prefix_and_target_replacement() -> Result<()> {
        let mut generator = create_test_generator(vec![0], vec![]);
        let draft_tokens = [1, 2, 3];
        let verification_logits = Tensor::new(
            &[
                [0.0f32, 10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0, 10.0, 0.0],
            ],
            &Device::Cpu,
        )?;

        let result = generator.verify_draft_tokens(&draft_tokens, &verification_logits)?;

        assert_eq!(result.accepted_token_count, 1);
        assert_eq!(result.maybe_replacement_token, Some(4));
        Ok(())
    }

    #[test]
    fn verifies_when_all_draft_tokens_are_accepted() -> Result<()> {
        let mut generator = create_test_generator(vec![0], vec![]);
        let draft_tokens = [1, 2, 3];
        let verification_logits = Tensor::new(
            &[
                [0.0f32, 10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 10.0],
            ],
            &Device::Cpu,
        )?;

        let result = generator.verify_draft_tokens(&draft_tokens, &verification_logits)?;

        assert_eq!(result.accepted_token_count, draft_tokens.len());
        assert_eq!(result.maybe_replacement_token, None);
        Ok(())
    }

    #[test]
    fn commits_speculative_tokens_and_stops_before_eos() -> Result<()> {
        let mut generator = create_test_generator(vec![1], vec![3]);
        let result = generator.commit_speculative_tokens(&[2, 3], Some(4))?;

        assert_eq!(result.committed_token_count, 1);
        assert!(!result.should_continue);
        assert_eq!(generator.tokens, vec![1, 2]);
        Ok(())
    }

    #[test]
    fn computes_acceptance_rate_only_when_draft_tokens_were_proposed() {
        let mut generator = create_test_generator(vec![1], vec![]);
        assert_eq!(generator.compute_draft_token_acceptance_rate(), None);

        generator.accepted_draft_token_count = 3;
        generator.proposed_draft_token_count = 4;
        assert_eq!(generator.compute_draft_token_acceptance_rate(), Some(0.75));
    }
}
