use crate::models::BatchedForwardInput;
use crate::models::ModelRole;
use crate::proto::model_runner::GenerateTextRequest;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::IndexOp;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
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
    pub(super) generation_phase: GenerationPhase,
}

struct PreparedModelForward {
    // TODO: Avoid copying the vector.
    input_token_ids: Vec<u32>,
    start_position: usize,
}

struct PreparedSpeculativePrefill {
    target: Option<PreparedModelForward>,
    draft: Option<PreparedModelForward>,
    chunk_end_position: usize,
    started_at: Instant,
}

struct PreparedSpeculativeDecode {
    generated_token_count: usize,
    original_cached_token_count: usize,
    maximum_draft_token_count: usize,
    started_at: Instant,
}

enum PreparedForwardPhase {
    Prefill {
        end_position: usize,
        started_at: Instant,
    },
    Decode {
        generated_token_count: usize,
        started_at: Instant,
    },
}

enum PreparedForward {
    Target {
        model_forward: PreparedModelForward,
        phase: PreparedForwardPhase,
    },
    SpeculativePrefill(PreparedSpeculativePrefill),
    SpeculativeDecode(PreparedSpeculativeDecode),
}

struct BatchedModelInputs {
    target_request_indices: Vec<usize>,
    target_inputs: Vec<BatchedForwardInput>,
    draft_inputs: Vec<BatchedForwardInput>,
    draft_proposals: Vec<BatchedDraftProposal>,
}

struct BatchedDraftProposal {
    request_index: usize,
    original_cached_token_count: usize,
    maximum_token_count: usize,
    token_ids: Vec<u32>,
}

struct BatchedTargetVerificationInput {
    request_index: usize,
    input: BatchedForwardInput,
}

struct BatchedTargetLogits {
    generation: Vec<Option<Tensor>>,
    verification: Vec<Option<Tensor>>,
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
    target_logits_processor: LogitsProcessor,
    draft_logits_processor: LogitsProcessor,
    prefill_initial_positions: PrefillStartPositions,
    prefill_current_positions: PrefillStartPositions,
    generation_phase: GenerationPhase,
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
        self.generation_phase
    }

    pub(super) fn new(
        request: GenerateTextRequest,
        draft_token_count: usize,
        prefill_initial_positions: PrefillStartPositions,
    ) -> Result<Self> {
        ensure!(
            !request.input_token_ids.is_empty(),
            "input token IDs must not be empty"
        );
        let input_token_count = request.input_token_ids.len();
        validate_prefill_start_positions(input_token_count, prefill_initial_positions)?;
        let generation_phase =
            determine_initial_phase(input_token_count, prefill_initial_positions);
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
            target_logits_processor: LogitsProcessor::new(0, None, None),
            draft_logits_processor: LogitsProcessor::new(0, None, None),
            prefill_initial_positions,
            prefill_current_positions: prefill_initial_positions,
            generation_phase,
            pending_output_token_ids: Vec::new(),
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
            accepted_draft_token_count: 0,
            proposed_draft_token_count: 0,
        })
    }

    fn prepare_forward(
        &mut self,
        token_budget: usize,
        use_speculative_decoding: bool,
    ) -> Result<PreparedForward> {
        ensure!(token_budget > 0, "token budget must be greater than zero");
        self.pending_output_token_ids.clear();
        if !use_speculative_decoding {
            let (model_forward, phase) = self.prepare_target_forward(token_budget)?;
            return Ok(PreparedForward::Target {
                model_forward,
                phase,
            });
        }

        match self.generation_phase {
            GenerationPhase::Prefill { .. } => Ok(PreparedForward::SpeculativePrefill(
                self.prepare_speculative_prefill(token_budget)?,
            )),
            GenerationPhase::Decode {
                generated_token_count,
            } => Ok(PreparedForward::SpeculativeDecode(
                self.prepare_speculative_decode(generated_token_count),
            )),
            GenerationPhase::Finished => {
                bail!("cannot prepare a finished request for batched execution")
            }
        }
    }

    /// Selects the next input chunk and records how its logits must advance this request.
    fn prepare_target_forward(
        &mut self,
        token_budget: usize,
    ) -> Result<(PreparedModelForward, PreparedForwardPhase)> {
        let started_at = Instant::now();
        let (input_token_ids, start_position, prepared_phase) = match self.generation_phase {
            GenerationPhase::Prefill { .. } => {
                let start_position = self.prefill_current_positions.target;
                let end_position = start_position
                    .saturating_add(token_budget)
                    .min(self.input_token_count);
                (
                    self.tokens[start_position..end_position].to_vec(),
                    start_position,
                    PreparedForwardPhase::Prefill {
                        end_position,
                        started_at,
                    },
                )
            }
            GenerationPhase::Decode {
                generated_token_count,
            } => {
                let start_position = self.tokens.len() - 1;
                (
                    self.tokens[start_position..].to_vec(),
                    start_position,
                    PreparedForwardPhase::Decode {
                        generated_token_count,
                        started_at,
                    },
                )
            }
            GenerationPhase::Finished => {
                bail!("cannot prepare a finished request for batched execution")
            }
        };
        Ok((
            PreparedModelForward {
                input_token_ids,
                start_position,
            },
            prepared_phase,
        ))
    }

    fn prepare_speculative_prefill(
        &self,
        token_budget: usize,
    ) -> Result<PreparedSpeculativePrefill> {
        ensure!(
            token_budget > 0,
            "prefill token budget must be greater than zero"
        );
        let target_position = self.prefill_current_positions.target;
        let draft_position = self
            .prefill_current_positions
            .draft
            .context("draft prefill position is missing")?;
        let prefill_end_position = self.input_token_count - 1;
        let chunk_end_position = target_position
            .min(draft_position)
            .saturating_add(token_budget)
            .min(prefill_end_position);
        let prepare_model_forward = |start_position| PreparedModelForward {
            input_token_ids: self.tokens[start_position..chunk_end_position].to_vec(),
            start_position,
        };
        Ok(PreparedSpeculativePrefill {
            target: (target_position < chunk_end_position)
                .then(|| prepare_model_forward(target_position)),
            draft: (draft_position < chunk_end_position)
                .then(|| prepare_model_forward(draft_position)),
            chunk_end_position,
            started_at: Instant::now(),
        })
    }

    fn prepare_speculative_decode(
        &self,
        generated_token_count: usize,
    ) -> PreparedSpeculativeDecode {
        let remaining_max_token_count = self.max_new_token_count - generated_token_count;
        PreparedSpeculativeDecode {
            generated_token_count,
            original_cached_token_count: self.tokens.len() - 1,
            maximum_draft_token_count: self.draft_token_count.min(remaining_max_token_count),
            started_at: Instant::now(),
        }
    }

    /// Applies logits from either a sequential or batched model forward.
    fn complete_forward(
        &mut self,
        prepared_phase: PreparedForwardPhase,
        logits: &Tensor,
    ) -> Result<GenerationStep> {
        self.generation_phase = match prepared_phase {
            PreparedForwardPhase::Prefill {
                end_position,
                started_at,
            } => {
                self.prefill_duration += started_at.elapsed();
                self.prefill_current_positions.target = end_position;
                if end_position < self.input_token_count {
                    determine_prefill_phase(self.input_token_count - end_position)
                } else {
                    let should_decode = self.sample_and_commit_target_token(logits)?;
                    self.determine_decode_phase(usize::from(should_decode), should_decode)
                }
            }
            PreparedForwardPhase::Decode {
                generated_token_count,
                started_at,
            } => {
                self.decode_duration += started_at.elapsed();
                let should_continue = self.sample_and_commit_target_token(logits)?;
                self.determine_decode_phase(generated_token_count + 1, should_continue)
            }
        };
        Ok(GenerationStep {
            output_token_ids: std::mem::take(&mut self.pending_output_token_ids),
            generation_phase: self.generation_phase,
        })
    }

    fn complete_speculative_prefill(
        &mut self,
        prepared: PreparedSpeculativePrefill,
    ) -> Result<GenerationStep> {
        ensure!(
            prepared.target.is_some() || prepared.draft.is_some(),
            "speculative prefill must advance the target or draft model"
        );
        self.prefill_duration += prepared.started_at.elapsed();
        if prepared.target.is_some() {
            self.prefill_current_positions.target = prepared.chunk_end_position;
        }
        if prepared.draft.is_some() {
            self.prefill_current_positions.draft = Some(prepared.chunk_end_position);
        }

        let prefill_end_position = self.input_token_count - 1;
        let current_position = self.prefill_current_positions.target.min(
            self.prefill_current_positions
                .draft
                .unwrap_or(prefill_end_position),
        );
        self.generation_phase = determine_prefill_phase(
            /* remaining_token_count */ prefill_end_position - current_position,
        );
        Ok(GenerationStep {
            output_token_ids: std::mem::take(&mut self.pending_output_token_ids),
            generation_phase: self.generation_phase,
        })
    }

    fn complete_batched_speculative_decode(
        &mut self,
        prepared: PreparedSpeculativeDecode,
        draft_tokens: &[u32],
        verification_logits: &Tensor,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
    ) -> Result<GenerationStep> {
        let DecodeIterationResult {
            committed_token_count,
            should_continue,
        } = self.complete_speculative_iteration(
            target,
            draft,
            prepared.original_cached_token_count,
            draft_tokens,
            verification_logits,
        )?;
        self.decode_duration += prepared.started_at.elapsed();
        let generated_token_count = prepared.generated_token_count + committed_token_count;
        self.generation_phase = self.determine_decode_phase(generated_token_count, should_continue);
        Ok(GenerationStep {
            output_token_ids: std::mem::take(&mut self.pending_output_token_ids),
            generation_phase: self.generation_phase,
        })
    }

    /// Advances one prefill phase or decode iteration and then yields back to the caller.
    pub(super) fn run_one_step(
        &mut self,
        target: &mut ModelInstance,
        draft: Option<&mut ModelInstance>,
        token_budget: usize,
    ) -> Result<GenerationStep> {
        let Some(draft) = draft else {
            return self.run_target_iteration(target, token_budget);
        };

        self.pending_output_token_ids.clear();
        self.generation_phase = match self.generation_phase {
            GenerationPhase::Prefill { .. } => {
                self.run_speculative_prefill_chunk(target, draft, token_budget)?
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
            generation_phase: self.generation_phase,
        })
    }

    fn run_target_iteration(
        &mut self,
        target: &mut ModelInstance,
        token_budget: usize,
    ) -> Result<GenerationStep> {
        ensure!(token_budget > 0, "token budget must be greater than zero");
        self.pending_output_token_ids.clear();
        let (model_forward, prepared_phase) = self.prepare_target_forward(token_budget)?;
        let input = target.create_input_tensor(&model_forward.input_token_ids)?;
        let logits = target.forward(self.request_id, &input, model_forward.start_position)?;
        self.complete_forward(prepared_phase, &logits)
    }

    pub(super) fn into_completed_generation(self) -> Result<CompletedGeneration> {
        ensure!(
            matches!(self.generation_phase, GenerationPhase::Finished),
            "generation request is not finished"
        );
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

    fn run_speculative_prefill_chunk(
        &mut self,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
        token_budget: usize,
    ) -> Result<GenerationPhase> {
        let prepared = self.prepare_speculative_prefill(token_budget)?;
        if let Some(target_forward) = &prepared.target {
            self.forward_prepared_input(target, target_forward)?;
        }
        if let Some(draft_forward) = &prepared.draft {
            self.forward_prepared_input(draft, draft_forward)?;
        }
        Ok(self
            .complete_speculative_prefill(prepared)?
            .generation_phase)
    }

    fn forward_prepared_input(
        &self,
        model: &mut ModelInstance,
        prepared: &PreparedModelForward,
    ) -> Result<()> {
        let input = model.create_input_tensor(&prepared.input_token_ids)?;
        model.forward(self.request_id, &input, prepared.start_position)?;
        Ok(())
    }

    fn run_decode_iteration(
        &mut self,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
        generated_token_count: usize,
    ) -> Result<GenerationPhase> {
        let DecodeIterationResult {
            committed_token_count,
            should_continue,
        } = self.run_speculative_iteration(
            target,
            draft,
            self.max_new_token_count - generated_token_count,
        )?;
        let generated_token_count = generated_token_count + committed_token_count;
        Ok(self.determine_decode_phase(generated_token_count, should_continue))
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

        self.complete_speculative_iteration(
            target,
            draft,
            original_cached_token_count,
            &draft_tokens,
            &verification_logits,
        )
    }

    fn complete_speculative_iteration(
        &mut self,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
        original_cached_token_count: usize,
        draft_tokens: &[u32],
        verification_logits: &Tensor,
    ) -> Result<DecodeIterationResult> {
        // Accept the longest prefix on which target and draft sampling agree. At the first
        // mismatch, retain the target token as the replacement output.
        let DraftVerificationResult {
            accepted_token_count,
            maybe_replacement_token,
        } = self.verify_draft_tokens(draft_tokens, verification_logits)?;
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
            let next_token =
                self.sample_next_draft_token(draft, &[input_token], start_position, &draft_tokens)?;
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
                ModelRole::Target,
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

    fn sample_next_draft_token(
        &mut self,
        draft: &mut ModelInstance,
        input_tokens: &[u32],
        start_position: usize,
        appended_tokens: &[u32],
    ) -> Result<u32> {
        let input = draft.create_input_tensor(input_tokens)?;
        let logits = draft.forward(self.request_id, &input, start_position)?;
        let logits = logits.to_dtype(DType::F32)?;
        self.sample_logits(&logits, appended_tokens, ModelRole::Draft)
    }

    fn sample_and_commit_target_token(&mut self, logits: &Tensor) -> Result<bool> {
        let logits = logits.to_dtype(DType::F32)?;
        let next_token =
            self.sample_logits(&logits, /* appended_tokens */ &[], ModelRole::Target)?;
        Ok(self.commit_next_token(next_token))
    }

    fn sample_logits(
        &mut self,
        logits: &Tensor,
        appended_tokens: &[u32],
        model_role: ModelRole,
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
        let logits_processor = match model_role {
            ModelRole::Target => &mut self.target_logits_processor,
            ModelRole::Draft => &mut self.draft_logits_processor,
        };
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

    fn determine_decode_phase(
        &self,
        generated_token_count: usize,
        should_continue: bool,
    ) -> GenerationPhase {
        if should_continue && generated_token_count < self.max_new_token_count {
            GenerationPhase::Decode {
                generated_token_count,
            }
        } else {
            GenerationPhase::Finished
        }
    }
}

struct BatchedRequestExecution {
    execution_state: Box<RequestExecutionState>,
    token_budget: usize,
}

/// Owns the request states selected for one batched model execution.
pub(super) struct RequestExecutionBatch {
    requests: Vec<BatchedRequestExecution>,
}

impl RequestExecutionBatch {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            requests: Vec::with_capacity(capacity),
        }
    }

    pub(super) fn push(
        &mut self,
        execution_state: Box<RequestExecutionState>,
        token_budget: usize,
    ) {
        self.requests.push(BatchedRequestExecution {
            execution_state,
            token_budget,
        });
    }

    pub(super) fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub(super) fn len(&self) -> usize {
        self.requests.len()
    }

    pub(super) fn into_execution_states(self) -> impl Iterator<Item = Box<RequestExecutionState>> {
        self.requests
            .into_iter()
            .map(|request| request.execution_state)
    }

    pub(super) fn run_batched_steps(
        &mut self,
        target: &mut ModelInstance,
        draft: Option<&mut ModelInstance>,
    ) -> Result<Vec<GenerationStep>> {
        match draft {
            Some(draft) => self.run_batched_speculative_decoding_steps(target, draft),
            None => self.run_batched_target_steps(target),
        }
    }

    fn run_batched_target_steps(
        &mut self,
        target: &mut ModelInstance,
    ) -> Result<Vec<GenerationStep>> {
        let prepared_forwards = self
            .requests
            .iter_mut()
            .map(|request| {
                request.execution_state.prepare_forward(
                    request.token_budget,
                    /* use_speculative_decoding */ false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let BatchedModelInputs {
            target_request_indices,
            target_inputs,
            draft_inputs: _,
            draft_proposals: _,
        } = self.create_initial_batched_model_inputs(&prepared_forwards, target, None)?;

        let mut target_logits = vec![None; self.requests.len()];
        let logits = target.forward_batched(&target_inputs)?;
        ensure!(
            logits.len() == target_inputs.len(),
            "batched target model forward returned {} outputs for {} requests",
            logits.len(),
            target_inputs.len()
        );
        for (request_index, logits) in target_request_indices.into_iter().zip(logits) {
            target_logits[request_index] = Some(logits);
        }

        self.requests
            .iter_mut()
            .zip(prepared_forwards)
            .zip(target_logits)
            .map(|((request, prepared_forward), target_logits)| {
                let PreparedForward::Target { phase, .. } = prepared_forward else {
                    bail!("non-speculative batch contains speculative work")
                };
                request.execution_state.complete_forward(
                    phase,
                    &target_logits.context("target model logits are missing")?,
                )
            })
            .collect()
    }

    fn run_batched_speculative_decoding_steps(
        &mut self,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
    ) -> Result<Vec<GenerationStep>> {
        // 1. Collect initial draft and target work, but defer target execution until draft work
        // has produced every speculative proposal.
        let prepared_forwards = self
            .requests
            .iter_mut()
            .map(|request| {
                request.execution_state.prepare_forward(
                    request.token_budget,
                    /* use_speculative_decoding */ true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let BatchedModelInputs {
            target_request_indices,
            target_inputs: generation_inputs,
            draft_inputs,
            mut draft_proposals,
        } = self.create_initial_batched_model_inputs(&prepared_forwards, target, Some(draft))?;

        // 2. Run all draft-prefill work, then generate every request's remaining proposals.
        if !draft_inputs.is_empty() {
            let logits = draft.forward_batched(&draft_inputs)?;
            ensure!(
                logits.len() == draft_inputs.len(),
                "batched draft model forward returned {} outputs for {} requests",
                logits.len(),
                draft_inputs.len()
            );
        }
        self.generate_batched_draft_proposals(draft, &mut draft_proposals)?;

        // 3. Build target-verification inputs from the completed draft proposals.
        let verification_inputs =
            self.create_target_verification_inputs(target, &draft_proposals)?;

        // 4. Run one combined target-model pass containing ordinary prefill, ordinary decode,
        // speculative target prefill, and speculative verification.
        let target_logits = self.run_combined_batched_target_forward(
            target,
            target_request_indices,
            generation_inputs,
            verification_inputs,
        )?;
        self.complete_batched_speculative_steps(
            prepared_forwards,
            draft_proposals,
            target_logits,
            target,
            draft,
        )
    }

    fn create_initial_batched_model_inputs(
        &self,
        prepared_forwards: &[PreparedForward],
        target: &ModelInstance,
        draft: Option<&ModelInstance>,
    ) -> Result<BatchedModelInputs> {
        let mut target_request_indices = Vec::new();
        let mut target_inputs = Vec::new();
        let mut draft_inputs = Vec::new();
        let mut draft_proposals = Vec::new();
        for (request_index, (request, prepared_forward)) in
            self.requests.iter().zip(prepared_forwards).enumerate()
        {
            let request_id = request.execution_state.request_id();
            match prepared_forward {
                PreparedForward::Target { model_forward, .. } => {
                    target_request_indices.push(request_index);
                    target_inputs.push(BatchedForwardInput {
                        request_id,
                        input: target.create_input_tensor(&model_forward.input_token_ids)?,
                        start_position: model_forward.start_position,
                    });
                }
                PreparedForward::SpeculativePrefill(prepared) => {
                    if let Some(target_forward) = &prepared.target {
                        target_request_indices.push(request_index);
                        target_inputs.push(BatchedForwardInput {
                            request_id,
                            input: target.create_input_tensor(&target_forward.input_token_ids)?,
                            start_position: target_forward.start_position,
                        });
                    }
                    if let Some(draft_forward) = &prepared.draft {
                        let draft = draft.context("draft model is missing")?;
                        draft_inputs.push(BatchedForwardInput {
                            request_id,
                            input: draft.create_input_tensor(&draft_forward.input_token_ids)?,
                            start_position: draft_forward.start_position,
                        });
                    }
                }
                PreparedForward::SpeculativeDecode(prepared) => {
                    draft_proposals.push(BatchedDraftProposal {
                        request_index,
                        original_cached_token_count: prepared.original_cached_token_count,
                        maximum_token_count: prepared.maximum_draft_token_count,
                        token_ids: Vec::new(),
                    });
                }
            }
        }
        Ok(BatchedModelInputs {
            target_request_indices,
            target_inputs,
            draft_inputs,
            draft_proposals,
        })
    }

    fn generate_batched_draft_proposals(
        &mut self,
        draft: &mut ModelInstance,
        proposals: &mut [BatchedDraftProposal],
    ) -> Result<()> {
        loop {
            let proposal_indices = self.extract_active_draft_proposal_indices(proposals);
            if proposal_indices.is_empty() {
                return Ok(());
            }
            let proposal_inputs =
                self.create_draft_proposal_inputs(draft, proposals, &proposal_indices)?;
            let logits = draft.forward_batched(&proposal_inputs)?;
            self.sample_and_append_draft_proposals(proposals, &proposal_indices, logits)?;
        }
    }

    fn extract_active_draft_proposal_indices(
        &self,
        proposals: &[BatchedDraftProposal],
    ) -> Vec<usize> {
        let mut active_proposal_indices = Vec::new();
        for (proposal_index, proposal) in proposals.iter().enumerate() {
            let reached_token_limit = proposal.token_ids.len() >= proposal.maximum_token_count;
            let reached_end_of_sequence = proposal.token_ids.last().is_some_and(|token_id| {
                self.requests[proposal.request_index]
                    .execution_state
                    .eos_tokens
                    .contains(token_id)
            });
            if !reached_token_limit && !reached_end_of_sequence {
                active_proposal_indices.push(proposal_index);
            }
        }
        active_proposal_indices
    }

    fn create_draft_proposal_inputs(
        &self,
        draft: &ModelInstance,
        proposals: &[BatchedDraftProposal],
        active_proposal_indices: &[usize],
    ) -> Result<Vec<BatchedForwardInput>> {
        active_proposal_indices
            .iter()
            .map(|&proposal_index| {
                let proposal = &proposals[proposal_index];
                let execution_state = &self.requests[proposal.request_index].execution_state;
                let input_token_id = proposal
                    .token_ids
                    .last()
                    .copied()
                    .or_else(|| execution_state.tokens.last().copied())
                    .context("generation context is empty")?;
                Ok(BatchedForwardInput {
                    request_id: execution_state.request_id,
                    input: draft.create_input_tensor(&[input_token_id])?,
                    start_position: proposal.original_cached_token_count + proposal.token_ids.len(),
                })
            })
            .collect()
    }

    fn sample_and_append_draft_proposals(
        &mut self,
        proposals: &mut [BatchedDraftProposal],
        active_proposal_indices: &[usize],
        logits: Vec<Tensor>,
    ) -> Result<()> {
        ensure!(
            logits.len() == active_proposal_indices.len(),
            "batched draft model forward returned {} outputs for {} requests",
            logits.len(),
            active_proposal_indices.len()
        );
        for (proposal_index, logits) in active_proposal_indices.iter().copied().zip(logits) {
            let proposal = &mut proposals[proposal_index];
            let execution_state = &mut self.requests[proposal.request_index].execution_state;
            let logits = logits.to_dtype(DType::F32)?;
            let next_token =
                execution_state.sample_logits(&logits, &proposal.token_ids, ModelRole::Draft)?;
            proposal.token_ids.push(next_token);
        }
        Ok(())
    }

    fn create_target_verification_inputs(
        &self,
        target: &ModelInstance,
        proposals: &[BatchedDraftProposal],
    ) -> Result<Vec<BatchedTargetVerificationInput>> {
        let mut verification_inputs = Vec::with_capacity(proposals.len());
        for proposal in proposals {
            ensure!(
                !proposal.token_ids.is_empty(),
                "target verification requires at least one draft token"
            );
            let execution_state = &self.requests[proposal.request_index].execution_state;
            // Verify the last known token followed by every speculative token except the last;
            // each resulting logit predicts the speculative token at the same position.
            let mut input_token_ids = Vec::with_capacity(proposal.token_ids.len());
            input_token_ids.push(
                *execution_state
                    .tokens
                    .last()
                    .context("generation context is empty")?,
            );
            input_token_ids.extend_from_slice(&proposal.token_ids[..proposal.token_ids.len() - 1]);
            verification_inputs.push(BatchedTargetVerificationInput {
                request_index: proposal.request_index,
                input: BatchedForwardInput {
                    request_id: execution_state.request_id,
                    input: target.create_input_tensor(&input_token_ids)?,
                    start_position: proposal.original_cached_token_count,
                },
            });
        }
        Ok(verification_inputs)
    }

    fn run_combined_batched_target_forward(
        &self,
        target: &mut ModelInstance,
        generation_request_indices: Vec<usize>,
        generation_inputs: Vec<BatchedForwardInput>,
        verification_inputs: Vec<BatchedTargetVerificationInput>,
    ) -> Result<BatchedTargetLogits> {
        let mut generation = vec![None; self.requests.len()];
        let mut verification = vec![None; self.requests.len()];
        let verification_request_indices = verification_inputs
            .iter()
            .map(|verification_input| verification_input.request_index)
            .collect::<Vec<_>>();
        let verification_inputs = verification_inputs
            .into_iter()
            .map(|verification_input| verification_input.input)
            .collect::<Vec<_>>();

        if !generation_inputs.is_empty() || !verification_inputs.is_empty() {
            let output = target.forward_batched_with_speculative_verification(
                &generation_inputs,
                &verification_inputs,
            )?;
            ensure!(
                output.generation_logits.len() == generation_inputs.len(),
                "batched target model forward returned {} outputs for {} requests",
                output.generation_logits.len(),
                generation_inputs.len()
            );
            ensure!(
                output.verification_logits.len() == verification_inputs.len(),
                "batched target verification returned {} outputs for {} requests",
                output.verification_logits.len(),
                verification_inputs.len()
            );
            for (request_index, logits) in generation_request_indices
                .into_iter()
                .zip(output.generation_logits)
            {
                generation[request_index] = Some(logits);
            }
            for (request_index, logits) in verification_request_indices
                .into_iter()
                .zip(output.verification_logits)
            {
                verification[request_index] = Some(logits);
            }
        }

        Ok(BatchedTargetLogits {
            generation,
            verification,
        })
    }

    fn complete_batched_speculative_steps(
        &mut self,
        prepared_forwards: Vec<PreparedForward>,
        draft_proposals: Vec<BatchedDraftProposal>,
        mut target_logits: BatchedTargetLogits,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
    ) -> Result<Vec<GenerationStep>> {
        let mut draft_tokens = std::iter::repeat_with(|| None)
            .take(self.requests.len())
            .collect::<Vec<_>>();
        for proposal in draft_proposals {
            draft_tokens[proposal.request_index] = Some(proposal.token_ids);
        }

        let mut generation_steps = Vec::with_capacity(self.requests.len());
        for (request_index, (request, prepared_forward)) in
            self.requests.iter_mut().zip(prepared_forwards).enumerate()
        {
            let generation_step = match prepared_forward {
                PreparedForward::Target { phase, .. } => request.execution_state.complete_forward(
                    phase,
                    &target_logits.generation[request_index]
                        .take()
                        .context("target model logits are missing")?,
                ),
                PreparedForward::SpeculativePrefill(prepared) => request
                    .execution_state
                    .complete_speculative_prefill(prepared),
                PreparedForward::SpeculativeDecode(prepared) => {
                    request.execution_state.complete_batched_speculative_decode(
                        prepared,
                        &draft_tokens[request_index]
                            .take()
                            .context("draft proposal tokens are missing")?,
                        &target_logits.verification[request_index]
                            .take()
                            .context("target verification logits are missing")?,
                        target,
                        draft,
                    )
                }
            }?;
            generation_steps.push(generation_step);
        }
        Ok(generation_steps)
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
    determine_prefill_phase(remaining_token_count)
}

fn determine_prefill_phase(remaining_token_count: usize) -> GenerationPhase {
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

fn validate_prefill_start_positions(
    input_token_count: usize,
    positions: PrefillStartPositions,
) -> Result<()> {
    let maximum_position = input_token_count - 1;
    ensure!(
        positions.target <= maximum_position,
        "target prefill position {} exceeds the maximum position {maximum_position}",
        positions.target
    );
    if let Some(draft_position) = positions.draft {
        ensure!(
            draft_position <= maximum_position,
            "draft prefill position {draft_position} exceeds the maximum position \
             {maximum_position}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::server::kv_cache::create_kv_cache;
    use crate::model_runner::KvCacheType;
    use crate::models::loaded_model::LoadedModel;
    use crate::models::BatchedForwardInput;
    use crate::models::BatchedForwardOutput;
    use crate::models::BatchedKvCache;
    use crate::models::CausalLanguageModel;
    use crate::models::ForwardContext;
    use crate::models::KvCache;
    use crate::models::ModelInfo;
    use crate::models::ModelRole;
    use candle_core::Device;
    use std::sync::Arc;
    use std::sync::Mutex;

    #[derive(Debug, PartialEq)]
    struct ForwardCall {
        start_position: usize,
        token_ids: Vec<u32>,
    }

    struct TestModel {
        info: ModelInfo,
        next_token: u32,
        forward_calls: Arc<Mutex<Vec<ForwardCall>>>,
    }

    impl CausalLanguageModel for TestModel {
        fn info(&self) -> &ModelInfo {
            &self.info
        }

        fn forward(
            &mut self,
            input: &Tensor,
            context: &ForwardContext,
            _kv_cache: &mut dyn KvCache,
        ) -> Result<Tensor> {
            let token_ids = input.to_vec2::<u32>()?.into_iter().flatten().collect();
            self.forward_calls.lock().unwrap().push(ForwardCall {
                start_position: context.start_position,
                token_ids,
            });
            let mut logits = vec![0.0f32; 8];
            logits[self.next_token as usize] = 1.0;
            Ok(Tensor::new(logits.as_slice(), &Device::Cpu)?)
        }

        fn forward_batched_with_speculative_verification(
            &mut self,
            generation_inputs: &[BatchedForwardInput],
            verification_inputs: &[BatchedForwardInput],
            _kv_cache: &mut dyn BatchedKvCache,
        ) -> Result<BatchedForwardOutput> {
            let generation_logits = generation_inputs
                .iter()
                .map(|input| {
                    let token_ids = input
                        .input
                        .to_vec2::<u32>()?
                        .into_iter()
                        .flatten()
                        .collect();
                    self.forward_calls.lock().unwrap().push(ForwardCall {
                        start_position: input.start_position,
                        token_ids,
                    });
                    let mut logits = vec![0.0f32; 8];
                    logits[self.next_token as usize] = 1.0;
                    Ok(Tensor::new(logits, &Device::Cpu)?)
                })
                .collect::<Result<Vec<_>>>()?;
            let verification_logits = verification_inputs
                .iter()
                .map(|input| {
                    let token_ids = input
                        .input
                        .to_vec2::<u32>()?
                        .into_iter()
                        .flatten()
                        .collect();
                    self.forward_calls.lock().unwrap().push(ForwardCall {
                        start_position: input.start_position,
                        token_ids,
                    });
                    let query_len = input.input.dim(1)?;
                    let mut logits = vec![0.0f32; query_len * 8];
                    for position in 0..query_len {
                        logits[position * 8 + self.next_token as usize] = 1.0;
                    }
                    Ok(Tensor::new(logits, &Device::Cpu)?.reshape((query_len, 8))?)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(BatchedForwardOutput {
                generation_logits,
                verification_logits,
            })
        }

        fn forward_for_speculative_verification(
            &mut self,
            input: &Tensor,
            context: &ForwardContext,
            kv_cache: &mut dyn KvCache,
        ) -> Result<Tensor> {
            Ok(self.forward(input, context, kv_cache)?.reshape((1, 1, 8))?)
        }
    }

    fn create_test_model(next_token: u32) -> Result<(ModelInstance, Arc<Mutex<Vec<ForwardCall>>>)> {
        let forward_calls = Arc::new(Mutex::new(Vec::new()));
        let model = LoadedModel::for_test(Box::new(TestModel {
            info: ModelInfo {
                layer_count: 1,
                num_kv_heads: 1,
                head_dim: 1,
                activation_dtype: DType::F32,
            },
            next_token,
            forward_calls: Arc::clone(&forward_calls),
        }));
        let kv_cache = create_kv_cache(
            KvCacheType::Contiguous,
            &model,
            ModelRole::Target,
            /* total_size_bytes */ 128,
        )?;
        Ok((ModelInstance::new(model, kv_cache), forward_calls))
    }

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
            target_logits_processor: LogitsProcessor::new(0, None, None),
            draft_logits_processor: LogitsProcessor::new(0, None, None),
            prefill_initial_positions: PrefillStartPositions {
                target: 0,
                draft: None,
            },
            prefill_current_positions: PrefillStartPositions {
                target: 0,
                draft: None,
            },
            generation_phase: GenerationPhase::Decode {
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
    fn runs_a_partial_target_prefill_chunk() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4], vec![]);
        execution_state.prefill_current_positions.target = 0;
        execution_state.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 4,
        };
        let (mut target, forward_calls) = create_test_model(5)?;

        let step = execution_state.run_one_step(&mut target, None, 2)?;

        assert!(matches!(
            step.generation_phase,
            GenerationPhase::Prefill {
                remaining_token_count: 2
            }
        ));
        assert_eq!(execution_state.prefill_current_positions.target, 2);
        assert_eq!(
            *forward_calls.lock().unwrap(),
            vec![ForwardCall {
                start_position: 0,
                token_ids: vec![1, 2],
            }]
        );
        Ok(())
    }

    #[test]
    fn prepares_and_completes_a_partial_prefill() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4], vec![]);
        execution_state.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 4,
        };

        let (model_forward, prepared_phase) = execution_state.prepare_target_forward(2)?;
        assert_eq!(model_forward.start_position, 0);
        assert_eq!(model_forward.input_token_ids, [1, 2]);

        let logits = Tensor::zeros(8, DType::F32, &Device::Cpu)?;
        let step = execution_state.complete_forward(prepared_phase, &logits)?;
        assert!(step.output_token_ids.is_empty());
        assert!(matches!(
            step.generation_phase,
            GenerationPhase::Prefill {
                remaining_token_count: 2
            }
        ));
        assert_eq!(execution_state.prefill_current_positions.target, 2);
        Ok(())
    }

    #[test]
    fn completes_final_prefill_with_the_first_output_token() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4], vec![]);
        execution_state.prefill_current_positions.target = 2;
        execution_state.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 2,
        };

        let (model_forward, prepared_phase) = execution_state.prepare_target_forward(8)?;
        assert_eq!(model_forward.start_position, 2);
        assert_eq!(model_forward.input_token_ids, [3, 4]);

        let mut logits = vec![0.0f32; 8];
        logits[5] = 1.0;
        let logits = Tensor::new(logits, &Device::Cpu)?;
        let step = execution_state.complete_forward(prepared_phase, &logits)?;
        assert_eq!(step.output_token_ids, [5]);
        assert!(matches!(
            step.generation_phase,
            GenerationPhase::Decode {
                generated_token_count: 1
            }
        ));
        assert_eq!(execution_state.tokens, [1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn prepares_and_completes_a_decode_iteration() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4], vec![]);

        let (model_forward, prepared_phase) = execution_state.prepare_target_forward(1)?;
        assert_eq!(model_forward.start_position, 3);
        assert_eq!(model_forward.input_token_ids, [4]);

        let mut logits = vec![0.0f32; 8];
        logits[5] = 1.0;
        let logits = Tensor::new(logits, &Device::Cpu)?;
        let step = execution_state.complete_forward(prepared_phase, &logits)?;
        assert_eq!(step.output_token_ids, [5]);
        assert!(matches!(
            step.generation_phase,
            GenerationPhase::Decode {
                generated_token_count: 1
            }
        ));
        assert_eq!(execution_state.tokens, [1, 2, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn runs_multiple_request_states_in_one_model_batch() -> Result<()> {
        let mut first = create_test_execution_state(vec![1, 2], vec![]);
        first.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 2,
        };
        let mut second = create_test_execution_state(vec![3, 4, 5], vec![]);
        second.request_id = 2;
        second.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 3,
        };
        let (mut target, forward_calls) = create_test_model(6)?;
        let mut batch = RequestExecutionBatch::with_capacity(2);
        batch.push(Box::new(first), 2);
        batch.push(Box::new(second), 3);

        let steps = batch.run_batched_steps(&mut target, None)?;
        let mut execution_states = batch.into_execution_states();
        let first = execution_states
            .next()
            .context("first execution is missing")?;
        let second = execution_states
            .next()
            .context("second execution is missing")?;

        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].output_token_ids, [6]);
        assert_eq!(steps[1].output_token_ids, [6]);
        assert_eq!(first.tokens, [1, 2, 6]);
        assert_eq!(second.tokens, [3, 4, 5, 6]);
        assert_eq!(
            *forward_calls.lock().unwrap(),
            [
                ForwardCall {
                    start_position: 0,
                    token_ids: vec![1, 2],
                },
                ForwardCall {
                    start_position: 0,
                    token_ids: vec![3, 4, 5],
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn batches_speculative_prefill_for_models_with_different_cached_positions() -> Result<()> {
        let mut first = create_test_execution_state(vec![1, 2, 3, 4, 5, 6], vec![]);
        first.prefill_current_positions = PrefillStartPositions {
            target: 0,
            draft: Some(2),
        };
        first.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 5,
        };
        let mut second = create_test_execution_state(vec![7, 8, 9, 10, 11, 12], vec![]);
        second.request_id = 2;
        second.prefill_current_positions = PrefillStartPositions {
            target: 3,
            draft: Some(1),
        };
        second.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 4,
        };
        let (mut target, target_forward_calls) = create_test_model(6)?;
        let (mut draft, draft_forward_calls) = create_test_model(6)?;
        let mut batch = RequestExecutionBatch::with_capacity(2);
        batch.push(Box::new(first), 2);
        batch.push(Box::new(second), 2);

        let steps = batch.run_batched_steps(&mut target, Some(&mut draft))?;
        let mut execution_states = batch.into_execution_states();
        let first = execution_states
            .next()
            .context("first execution is missing")?;
        let second = execution_states
            .next()
            .context("second execution is missing")?;

        assert!(matches!(
            steps[0].generation_phase,
            GenerationPhase::Prefill {
                remaining_token_count: 3
            }
        ));
        assert!(matches!(
            steps[1].generation_phase,
            GenerationPhase::Prefill {
                remaining_token_count: 2
            }
        ));
        assert_eq!(first.prefill_current_positions.target, 2);
        assert_eq!(first.prefill_current_positions.draft, Some(2));
        assert_eq!(second.prefill_current_positions.target, 3);
        assert_eq!(second.prefill_current_positions.draft, Some(3));
        assert_eq!(
            *target_forward_calls.lock().unwrap(),
            [ForwardCall {
                start_position: 0,
                token_ids: vec![1, 2],
            }]
        );
        assert_eq!(
            *draft_forward_calls.lock().unwrap(),
            [ForwardCall {
                start_position: 1,
                token_ids: vec![8, 9],
            }]
        );
        Ok(())
    }

    #[test]
    fn generates_batched_draft_proposals_and_target_verification_inputs() -> Result<()> {
        let execution_state = create_test_execution_state(vec![1, 2], vec![]);
        let (mut draft, draft_forward_calls) = create_test_model(6)?;
        let (target, _) = create_test_model(6)?;
        let mut batch = RequestExecutionBatch::with_capacity(1);
        batch.push(Box::new(execution_state), 1);
        let mut proposals = vec![BatchedDraftProposal {
            request_index: 0,
            original_cached_token_count: 1,
            maximum_token_count: 3,
            token_ids: Vec::new(),
        }];

        batch.generate_batched_draft_proposals(&mut draft, &mut proposals)?;
        let verification_inputs = batch.create_target_verification_inputs(&target, &proposals)?;

        assert_eq!(proposals[0].token_ids, [6, 6, 6]);
        assert_eq!(verification_inputs.len(), 1);
        assert_eq!(verification_inputs[0].request_index, 0);
        assert_eq!(verification_inputs[0].input.start_position, 1);
        assert_eq!(
            verification_inputs[0].input.input.to_vec2::<u32>()?,
            [vec![2, 6, 6]]
        );
        assert_eq!(
            *draft_forward_calls.lock().unwrap(),
            [
                ForwardCall {
                    start_position: 1,
                    token_ids: vec![2],
                },
                ForwardCall {
                    start_position: 2,
                    token_ids: vec![6],
                },
                ForwardCall {
                    start_position: 3,
                    token_ids: vec![6],
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn final_target_prefill_chunk_generates_the_first_output_token() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4], vec![]);
        execution_state.prefill_current_positions.target = 2;
        execution_state.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 2,
        };
        let (mut target, forward_calls) = create_test_model(5)?;

        let step = execution_state.run_one_step(&mut target, None, 8)?;

        assert!(matches!(
            step.generation_phase,
            GenerationPhase::Decode {
                generated_token_count: 1
            }
        ));
        assert_eq!(execution_state.prefill_current_positions.target, 4);
        assert_eq!(execution_state.tokens, vec![1, 2, 3, 4, 5]);
        assert_eq!(step.output_token_ids, vec![5]);
        assert_eq!(
            *forward_calls.lock().unwrap(),
            vec![ForwardCall {
                start_position: 2,
                token_ids: vec![3, 4],
            }]
        );
        Ok(())
    }

    #[test]
    fn speculative_prefill_advances_the_lagging_model_then_both_models() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4, 5, 6], vec![]);
        execution_state.prefill_current_positions = PrefillStartPositions {
            target: 3,
            draft: Some(1),
        };
        let (mut target, target_calls) = create_test_model(6)?;
        let (mut draft, draft_calls) = create_test_model(6)?;

        let phase = execution_state.run_speculative_prefill_chunk(&mut target, &mut draft, 2)?;

        assert!(matches!(
            phase,
            GenerationPhase::Prefill {
                remaining_token_count: 2
            }
        ));
        assert_eq!(execution_state.prefill_current_positions.target, 3);
        assert_eq!(execution_state.prefill_current_positions.draft, Some(3));
        assert!(target_calls.lock().unwrap().is_empty());
        assert_eq!(
            *draft_calls.lock().unwrap(),
            vec![ForwardCall {
                start_position: 1,
                token_ids: vec![2, 3],
            }]
        );

        let phase = execution_state.run_speculative_prefill_chunk(&mut target, &mut draft, 8)?;

        assert!(matches!(
            phase,
            GenerationPhase::Decode {
                generated_token_count: 0
            }
        ));
        assert_eq!(execution_state.prefill_current_positions.target, 5);
        assert_eq!(execution_state.prefill_current_positions.draft, Some(5));
        assert_eq!(
            *target_calls.lock().unwrap(),
            vec![ForwardCall {
                start_position: 3,
                token_ids: vec![4, 5],
            }]
        );
        assert_eq!(
            *draft_calls.lock().unwrap(),
            vec![
                ForwardCall {
                    start_position: 1,
                    token_ids: vec![2, 3],
                },
                ForwardCall {
                    start_position: 3,
                    token_ids: vec![4, 5],
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn speculative_prefill_advances_the_target_when_the_draft_is_ahead() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4, 5, 6], vec![]);
        execution_state.prefill_current_positions = PrefillStartPositions {
            target: 1,
            draft: Some(3),
        };
        let (mut target, target_calls) = create_test_model(6)?;
        let (mut draft, draft_calls) = create_test_model(6)?;

        let phase = execution_state.run_speculative_prefill_chunk(&mut target, &mut draft, 2)?;

        assert!(matches!(
            phase,
            GenerationPhase::Prefill {
                remaining_token_count: 2
            }
        ));
        assert_eq!(execution_state.prefill_current_positions.target, 3);
        assert_eq!(execution_state.prefill_current_positions.draft, Some(3));
        assert_eq!(
            *target_calls.lock().unwrap(),
            vec![ForwardCall {
                start_position: 1,
                token_ids: vec![2, 3],
            }]
        );
        assert!(draft_calls.lock().unwrap().is_empty());
        Ok(())
    }

    #[test]
    fn determines_whether_decode_should_continue() {
        let mut execution_state = create_test_execution_state(vec![1], vec![]);
        execution_state.max_new_token_count = 2;

        assert!(matches!(
            execution_state.determine_decode_phase(1, true),
            GenerationPhase::Decode {
                generated_token_count: 1
            }
        ));
        assert!(matches!(
            execution_state.determine_decode_phase(2, true),
            GenerationPhase::Finished
        ));
        assert!(matches!(
            execution_state.determine_decode_phase(1, false),
            GenerationPhase::Finished
        ));
    }

    #[test]
    fn rejects_prefill_positions_past_the_restorable_input_prefix() {
        let error = RequestExecutionState::new(
            GenerateTextRequest {
                input_token_ids: vec![1, 2, 3],
                max_new_tokens: 1,
                ..Default::default()
            },
            4,
            PrefillStartPositions {
                target: 3,
                draft: None,
            },
        )
        .err()
        .expect("invalid prefill position should be rejected")
        .to_string();

        assert_eq!(
            error,
            "target prefill position 3 exceeds the maximum position 2"
        );
    }

    #[test]
    fn rejects_draft_prefill_positions_past_the_restorable_input_prefix() {
        let error = RequestExecutionState::new(
            GenerateTextRequest {
                input_token_ids: vec![1, 2, 3],
                max_new_tokens: 1,
                ..Default::default()
            },
            4,
            PrefillStartPositions {
                target: 0,
                draft: Some(3),
            },
        )
        .err()
        .expect("invalid draft prefill position should be rejected")
        .to_string();

        assert_eq!(
            error,
            "draft prefill position 3 exceeds the maximum position 2"
        );
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
