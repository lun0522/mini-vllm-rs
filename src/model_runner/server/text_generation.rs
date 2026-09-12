use crate::proto::model_runner::GenerateTextRequest;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::IndexOp;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use std::cell::RefCell;
use std::error::Error;
use std::fmt;
use std::time::Duration;
use std::time::Instant;

use super::model_instance::ModelInstance;

#[derive(Debug)]
pub(super) struct GenerationCancelled;

impl fmt::Display for GenerationCancelled {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("generation request was cancelled")
    }
}

impl Error for GenerationCancelled {}

#[derive(Clone, Copy)]
pub(super) enum GenerationPhase {
    Prefill { remaining_token_count: usize },
    Decode { generated_token_count: usize },
    Finished,
}

impl fmt::Display for GenerationPhase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prefill {
                remaining_token_count,
            } => write!(
                formatter,
                "prefill(remaining_tokens={remaining_token_count})"
            ),
            Self::Decode {
                generated_token_count,
            } => write!(
                formatter,
                "decode(generated_tokens={generated_token_count})"
            ),
            Self::Finished => formatter.write_str("finished"),
        }
    }
}

pub(super) struct GenerationStep {
    pub(super) output_token_ids: Vec<u32>,
    pub(super) phase: GenerationPhase,
}

struct DraftVerificationResult {
    accepted_token_count: usize,
    maybe_replacement_token: Option<u32>,
}

struct DecodeIterationResult {
    committed_token_count: usize,
    should_continue: bool,
}

#[derive(Clone, Copy)]
/// Records how many input tokens each model restored from its prefix cache, so prefill starts at
/// the first uncached token instead of position zero.
pub(super) struct PrefillStartPositions {
    pub(super) target: usize,
    pub(super) draft: Option<usize>,
}

pub(super) struct CompletedGeneration {
    pub(super) cached_sequence_token_ids: Vec<u32>,
    pub(super) stats: TextGenerationStats,
}

pub(super) struct TextGenerationStats {
    pub(super) input_token_count: u64,
    pub(super) output_token_count: u64,
    pub(super) target_cached_token_count: usize,
    pub(super) draft_stats: Option<DraftGenerationStats>,
    pub(super) prefill_duration: Duration,
    pub(super) decode_duration: Duration,
}

pub(super) struct DraftGenerationStats {
    pub(super) cached_token_count: usize,
    pub(super) accepted_token_count: usize,
    pub(super) proposed_token_count: usize,
}

/// Owns the generation progress and sampling state for one resumable request.
pub(super) struct RequestExecutionState {
    request_id: u64,
    tokens: Vec<u32>,
    input_token_count: usize,
    draft_token_count: usize,
    max_new_token_count: usize,
    repeat_last_n: usize,
    repeat_penalty: f32,
    eos_tokens: Vec<u32>,
    target_logits_processor: RefCell<LogitsProcessor>,
    draft_logits_processor: RefCell<LogitsProcessor>,
    prefill_initial_positions: PrefillStartPositions,
    prefill_current_positions: PrefillStartPositions,
    phase: GenerationPhase,
    pending_output_token_ids: Vec<u32>,
    prefill_duration: Duration,
    decode_duration: Duration,
    accepted_draft_token_count: usize,
    proposed_draft_token_count: usize,
}

impl RequestExecutionState {
    pub(super) fn request_id(&self) -> u64 {
        self.request_id
    }

    pub(super) fn phase(&self) -> GenerationPhase {
        self.phase
    }

    pub(super) fn new(
        request: GenerateTextRequest,
        draft_token_count: usize,
        prefill_initial_positions: PrefillStartPositions,
    ) -> Result<Self> {
        if request.input_token_ids.is_empty() {
            anyhow::bail!("input token IDs must not be empty");
        }
        let input_token_count = request.input_token_ids.len();
        let phase = determine_initial_phase(input_token_count, prefill_initial_positions);
        let max_new_token_count = usize::try_from(request.max_new_tokens)
            .context("max_new_tokens does not fit in usize")?;
        Ok(Self {
            request_id: request.request_id,
            tokens: request.input_token_ids,
            input_token_count,
            draft_token_count,
            max_new_token_count,
            repeat_last_n: usize::try_from(request.repeat_last_n)
                .context("repeat_last_n does not fit in usize")?,
            repeat_penalty: request.repeat_penalty,
            eos_tokens: request.end_of_sequence_token_ids,
            target_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
            draft_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
            prefill_initial_positions,
            prefill_current_positions: prefill_initial_positions,
            phase,
            pending_output_token_ids: Vec::new(),
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            accepted_draft_token_count: 0,
            proposed_draft_token_count: 0,
        })
    }

    /// Advances one prefill phase or decode iteration and then yields back to the caller.
    pub(super) fn run_one_step(
        &mut self,
        target: &mut ModelInstance,
        draft: Option<&mut ModelInstance>,
        token_budget: usize,
    ) -> Result<GenerationStep> {
        self.pending_output_token_ids.clear();
        self.phase = match self.phase {
            GenerationPhase::Prefill { .. } => {
                let started = Instant::now();
                let phase = self.run_prefill_phase(target, draft, token_budget)?;
                self.prefill_duration += started.elapsed();
                phase
            }
            GenerationPhase::Decode {
                generated_token_count,
            } => {
                let started = Instant::now();
                let phase = self.run_decode_iteration(target, draft, generated_token_count)?;
                self.decode_duration += started.elapsed();
                phase
            }
            GenerationPhase::Finished => GenerationPhase::Finished,
        };
        Ok(GenerationStep {
            output_token_ids: std::mem::take(&mut self.pending_output_token_ids),
            phase: self.phase,
        })
    }

    pub(super) fn into_completed_generation(self) -> Result<CompletedGeneration> {
        if !matches!(self.phase, GenerationPhase::Finished) {
            anyhow::bail!("generation request is not finished");
        }
        let output_token_count = self.tokens.len() - self.input_token_count;
        let stats = TextGenerationStats {
            input_token_count: u64::try_from(self.input_token_count)
                .context("input token count does not fit in u64")?,
            output_token_count: u64::try_from(output_token_count)
                .context("output token count does not fit in u64")?,
            target_cached_token_count: self.prefill_initial_positions.target,
            draft_stats: self
                .prefill_initial_positions
                .draft
                .map(|cached_token_count| DraftGenerationStats {
                    cached_token_count,
                    accepted_token_count: self.accepted_draft_token_count,
                    proposed_token_count: self.proposed_draft_token_count,
                }),
            prefill_duration: self.prefill_duration,
            decode_duration: self.decode_duration,
        };
        Ok(CompletedGeneration {
            cached_sequence_token_ids: self.tokens,
            stats,
        })
    }

    // TODO: simplify the logic.
    fn run_prefill_phase(
        &mut self,
        target: &mut ModelInstance,
        draft: Option<&mut ModelInstance>,
        token_budget: usize,
    ) -> Result<GenerationPhase> {
        if token_budget == 0 {
            anyhow::bail!("prefill token budget must be greater than zero");
        }
        if let Some(draft) = draft {
            let draft_position = self
                .prefill_current_positions
                .draft
                .context("draft prefill position is missing")?;
            let prefill_end_position = self.input_token_count - 1;
            let chunk_end_position = self
                .prefill_current_positions
                .target
                .min(draft_position)
                .saturating_add(token_budget)
                .min(prefill_end_position);
            if self.prefill_current_positions.target < chunk_end_position {
                self.forward_input_chunk(
                    target,
                    self.prefill_current_positions.target,
                    chunk_end_position,
                )?;
                self.prefill_current_positions.target = chunk_end_position;
            }
            if draft_position < chunk_end_position {
                self.forward_input_chunk(draft, draft_position, chunk_end_position)?;
                self.prefill_current_positions.draft = Some(chunk_end_position);
            }
            let remaining_token_count = prefill_end_position
                - self.prefill_current_positions.target.min(
                    self.prefill_current_positions
                        .draft
                        .unwrap_or(prefill_end_position),
                );
            return Ok(if remaining_token_count == 0 {
                GenerationPhase::Decode {
                    generated_token_count: 0,
                }
            } else {
                GenerationPhase::Prefill {
                    remaining_token_count,
                }
            });
        }

        let start_position = self.prefill_current_positions.target;
        let chunk_end_position = start_position
            .saturating_add(token_budget)
            .min(self.input_token_count);
        if chunk_end_position < self.input_token_count {
            self.forward_input_chunk(target, start_position, chunk_end_position)?;
            self.prefill_current_positions.target = chunk_end_position;
            return Ok(GenerationPhase::Prefill {
                remaining_token_count: self.input_token_count - chunk_end_position,
            });
        }
        let next_token = self.sample_next_token(
            target,
            &self.tokens[start_position..chunk_end_position],
            start_position,
            /* appended_tokens */ &[],
            &mut self.target_logits_processor.borrow_mut(),
        )?;
        self.prefill_current_positions.target = chunk_end_position;
        let should_decode = self.commit_next_token(next_token);
        let generated_token_count = usize::from(should_decode);
        Ok(
            if should_decode && generated_token_count < self.max_new_token_count {
                GenerationPhase::Decode {
                    generated_token_count,
                }
            } else {
                GenerationPhase::Finished
            },
        )
    }

    fn forward_input_chunk(
        &self,
        model: &mut ModelInstance,
        start_position: usize,
        end_position: usize,
    ) -> Result<()> {
        let input = model.create_input_tensor(&self.tokens[start_position..end_position])?;
        model.forward(self.request_id, &input, start_position)?;
        Ok(())
    }

    fn run_decode_iteration(
        &mut self,
        target: &mut ModelInstance,
        draft: Option<&mut ModelInstance>,
        generated_token_count: usize,
    ) -> Result<GenerationPhase> {
        let DecodeIterationResult {
            committed_token_count,
            should_continue,
        } = match draft {
            Some(draft) => self.run_speculative_iteration(
                target,
                draft,
                self.max_new_token_count - generated_token_count,
            )?,
            None => {
                let start_position = self.tokens.len() - 1;
                let next_token = self.sample_next_token(
                    target,
                    &self.tokens[start_position..],
                    start_position,
                    /* appended_tokens */ &[],
                    &mut self.target_logits_processor.borrow_mut(),
                )?;
                DecodeIterationResult {
                    committed_token_count: 1,
                    should_continue: self.commit_next_token(next_token),
                }
            }
        };
        let generated_token_count = generated_token_count + committed_token_count;
        Ok(
            if should_continue && generated_token_count < self.max_new_token_count {
                GenerationPhase::Decode {
                    generated_token_count,
                }
            } else {
                GenerationPhase::Finished
            },
        )
    }

    fn run_speculative_iteration(
        &mut self,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
        remaining_max_token_count: usize,
    ) -> Result<DecodeIterationResult> {
        // Generate draft proposals autoregressively, starting with the pending input or output
        // token that has not yet been written to either model's KV cache.
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
        target.truncate_cache(self.request_id, retained_cached_token_count)?;
        draft.truncate_cache(self.request_id, retained_cached_token_count)?;

        // Publish the accepted draft prefix, followed by the target replacement when the models
        // disagreed. EOS is observed but never added to the generated output.
        Ok(self.commit_speculative_tokens(
            &draft_tokens[..accepted_token_count],
            maybe_replacement_token,
        ))
    }

    fn generate_draft_tokens(
        &mut self,
        draft: &mut ModelInstance,
        original_cached_token_count: usize,
        remaining_max_token_count: usize,
    ) -> Result<Vec<u32>> {
        // Limit the proposal batch to the request's remaining output budget.
        let draft_token_count = self.draft_token_count.min(remaining_max_token_count);
        let mut draft_tokens = Vec::with_capacity(draft_token_count);
        for _ in 0..draft_token_count {
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
        target: &mut ModelInstance,
        draft_tokens: &[u32],
        start_position: usize,
    ) -> Result<Tensor> {
        let mut verification_tokens = Vec::with_capacity(draft_tokens.len());
        verification_tokens.push(*self.tokens.last().context("generation context is empty")?);
        verification_tokens.extend_from_slice(&draft_tokens[..draft_tokens.len() - 1]);
        let input = target.create_input_tensor(&verification_tokens)?;
        Ok(target
            .forward_for_speculative_verification(self.request_id, &input, start_position)?
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
    ) -> DecodeIterationResult {
        let mut committed_token_count = 0;
        for &draft_token in accepted_draft_tokens {
            if !self.commit_next_token(draft_token) {
                return DecodeIterationResult {
                    committed_token_count,
                    should_continue: false,
                };
            }
            committed_token_count += 1;
        }
        if let Some(replacement_token) = maybe_replacement_token {
            if !self.commit_next_token(replacement_token) {
                return DecodeIterationResult {
                    committed_token_count,
                    should_continue: false,
                };
            }
            committed_token_count += 1;
        }
        DecodeIterationResult {
            committed_token_count,
            should_continue: true,
        }
    }

    fn sample_next_token(
        &self,
        model: &mut ModelInstance,
        input_tokens: &[u32],
        start_position: usize,
        appended_tokens: &[u32],
        logits_processor: &mut LogitsProcessor,
    ) -> Result<u32> {
        let input = model.create_input_tensor(input_tokens)?;
        let logits = model.forward(self.request_id, &input, start_position)?;
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

    fn commit_next_token(&mut self, next_token: u32) -> bool {
        if self.eos_tokens.contains(&next_token) {
            return false;
        }
        self.tokens.push(next_token);
        self.pending_output_token_ids.push(next_token);
        true
    }
}

fn determine_initial_phase(
    input_token_count: usize,
    prefill_initial_positions: PrefillStartPositions,
) -> GenerationPhase {
    // Speculative decoding cannot begin until both model caches reach the final input token.
    let remaining_token_count = match prefill_initial_positions.draft {
        Some(draft_initial_position) => {
            input_token_count - 1 - prefill_initial_positions.target.min(draft_initial_position)
        }
        None => input_token_count - prefill_initial_positions.target,
    };
    if remaining_token_count == 0 {
        GenerationPhase::Decode {
            generated_token_count: 0,
        }
    } else {
        GenerationPhase::Prefill {
            remaining_token_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn create_test_execution_state(
        tokens: Vec<u32>,
        eos_tokens: Vec<u32>,
    ) -> RequestExecutionState {
        let input_token_count = tokens.len();
        RequestExecutionState {
            request_id: 1,
            tokens,
            input_token_count,
            draft_token_count: 4,
            max_new_token_count: 16,
            repeat_last_n: 64,
            repeat_penalty: 1.0,
            eos_tokens,
            target_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
            draft_logits_processor: RefCell::new(LogitsProcessor::new(0, None, None)),
            prefill_initial_positions: PrefillStartPositions {
                target: 0,
                draft: None,
            },
            prefill_current_positions: PrefillStartPositions {
                target: 0,
                draft: None,
            },
            phase: GenerationPhase::Decode {
                generated_token_count: 0,
            },
            pending_output_token_ids: Vec::new(),
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            accepted_draft_token_count: 0,
            proposed_draft_token_count: 0,
        }
    }

    #[test]
    fn initializes_target_prefill_with_uncached_input_count() -> Result<()> {
        let execution_state = RequestExecutionState::new(
            GenerateTextRequest {
                request_id: 1,
                input_token_ids: vec![1, 2, 3, 4, 5],
                max_new_tokens: 1,
                ..Default::default()
            },
            4,
            PrefillStartPositions {
                target: 2,
                draft: None,
            },
        )?;

        assert!(matches!(
            execution_state.phase(),
            GenerationPhase::Prefill {
                remaining_token_count: 3
            }
        ));
        Ok(())
    }

    #[test]
    fn initializes_speculative_prefill_from_the_shorter_cached_prefix() -> Result<()> {
        let execution_state = RequestExecutionState::new(
            GenerateTextRequest {
                request_id: 1,
                input_token_ids: vec![1, 2, 3, 4, 5, 6],
                max_new_tokens: 1,
                ..Default::default()
            },
            4,
            PrefillStartPositions {
                target: 3,
                draft: Some(1),
            },
        )?;

        assert!(matches!(
            execution_state.phase(),
            GenerationPhase::Prefill {
                remaining_token_count: 4
            }
        ));
        Ok(())
    }

    #[test]
    fn verifies_the_accepted_draft_prefix_and_target_replacement() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![0], vec![]);
        let draft_tokens = [1, 2, 3];
        let verification_logits = Tensor::new(
            &[
                [0.0f32, 10.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 10.0],
                [0.0, 0.0, 0.0, 10.0, 0.0],
            ],
            &Device::Cpu,
        )?;

        let result = execution_state.verify_draft_tokens(&draft_tokens, &verification_logits)?;

        assert_eq!(result.accepted_token_count, 1);
        assert_eq!(result.maybe_replacement_token, Some(4));
        Ok(())
    }

    #[test]
    fn verifies_when_all_draft_tokens_are_accepted() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![0], vec![]);
        let draft_tokens = [1, 2, 3];
        let verification_logits = Tensor::new(
            &[
                [0.0f32, 10.0, 0.0, 0.0],
                [0.0, 0.0, 10.0, 0.0],
                [0.0, 0.0, 0.0, 10.0],
            ],
            &Device::Cpu,
        )?;

        let result = execution_state.verify_draft_tokens(&draft_tokens, &verification_logits)?;

        assert_eq!(result.accepted_token_count, draft_tokens.len());
        assert_eq!(result.maybe_replacement_token, None);
        Ok(())
    }

    #[test]
    fn commits_speculative_tokens_and_stops_before_eos() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1], vec![3]);
        let result = execution_state.commit_speculative_tokens(&[2, 3], Some(4));

        assert_eq!(result.committed_token_count, 1);
        assert!(!result.should_continue);
        assert_eq!(execution_state.tokens, vec![1, 2]);
        assert_eq!(execution_state.pending_output_token_ids, vec![2]);
        Ok(())
    }

    #[test]
    fn rejects_result_before_generation_finishes() {
        let execution_state = create_test_execution_state(vec![1], vec![]);

        let error = execution_state
            .into_completed_generation()
            .err()
            .expect("unfinished generation should not produce a result")
            .to_string();

        assert_eq!(error, "generation request is not finished");
    }
}
