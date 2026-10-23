use crate::models::ForwardInput;
use crate::proto::model_runner::GenerateTextRequest;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::IndexOp;
use candle_core::Tensor;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::ops::Range;
use std::time::Duration;
use std::time::Instant;

use super::super::model_instance::ModelInstance;
use super::draft_token_count::DraftTokenCountController;
use super::generation_stats::DraftGenerationStats;
use super::generation_stats::GenerationExecutionStats;
use super::generation_stats::GenerationFinishReason;
use super::generation_stats::PrefixTokenCounts;
use super::sampler::SampleWithProbabilities;
use super::sampler::Sampler;

#[derive(Debug)]
pub struct GenerationCancelled;

impl fmt::Display for GenerationCancelled {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("generation request was cancelled")
    }
}

impl Error for GenerationCancelled {}

#[derive(Clone, Copy)]
pub enum GenerationPhase {
    Prefill {
        remaining_token_count: usize,
    },
    Decode {
        generated_token_count: usize,
    },
    Finished {
        finish_reason: GenerationFinishReason,
    },
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
            Self::Finished { finish_reason } => {
                write!(formatter, "finished(reason={finish_reason})")
            }
        }
    }
}

pub struct GenerationStep {
    pub output_token_ids: Vec<u32>,
    pub generation_phase: GenerationPhase,
}

struct TokenRange(Range<usize>);

impl TokenRange {
    fn token_ids<'a>(&self, tokens: &'a [u32]) -> &'a [u32] {
        &tokens[self.0.clone()]
    }

    fn start_position(&self) -> usize {
        self.0.start
    }
}

struct PreparedSpeculativePrefill {
    target: Option<TokenRange>,
    draft: Option<TokenRange>,
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
        token_range: TokenRange,
        phase: PreparedForwardPhase,
    },
    SpeculativePrefill(PreparedSpeculativePrefill),
    SpeculativeDecode(PreparedSpeculativeDecode),
}

struct ModelInputs {
    target_request_indices: Vec<usize>,
    target_inputs: Vec<ForwardInput>,
    draft_inputs: Vec<ForwardInput>,
    draft_proposals: Vec<DraftProposal>,
}

struct DraftProposal {
    request_index: usize,
    original_cached_token_count: usize,
    maximum_token_count: usize,
    token_ids: Vec<u32>,
    // One full q distribution per token for stochastic decoding; empty for greedy decoding.
    probabilities: Vec<Vec<f32>>,
}

struct TargetVerificationInput {
    request_index: usize,
    input: ForwardInput,
}

struct TargetLogits {
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
pub struct PrefillStartPositions {
    pub target: usize,
    pub draft: Option<usize>,
}

pub struct GenerationOutput {
    pub committed_token_ids: Vec<u32>,
    pub execution_stats: GenerationExecutionStats,
    pub restored_prefix_token_counts: PrefixTokenCounts,
}

struct DraftTokenCountState {
    controller: DraftTokenCountController,
    accepted_token_count: usize,
    proposed_token_count: usize,
    selected_token_count_histogram: BTreeMap<usize, usize>,
}

impl DraftTokenCountState {
    fn new(controller: DraftTokenCountController) -> Self {
        Self {
            controller,
            accepted_token_count: 0,
            proposed_token_count: 0,
            selected_token_count_histogram: BTreeMap::new(),
        }
    }

    fn record_verification(
        &mut self,
        accepted_token_count: usize,
        proposed_token_count: usize,
    ) -> Result<()> {
        let selected_token_count = self.controller.draft_token_count();
        *self
            .selected_token_count_histogram
            .entry(selected_token_count)
            .or_default() += 1;
        self.controller
            .update(accepted_token_count, proposed_token_count)?;
        self.accepted_token_count += accepted_token_count;
        self.proposed_token_count += proposed_token_count;
        Ok(())
    }
}

/// Owns the generation progress and sampling state for one resumable request.
pub struct RequestExecutionState {
    request_id: u64,
    tokens: Vec<u32>,
    input_token_count: usize,
    draft_token_count_state: Option<DraftTokenCountState>,
    max_new_token_count: usize,
    eos_tokens: Vec<u32>,
    target_sampler: Sampler,
    draft_sampler: Sampler,
    prefill_initial_positions: PrefillStartPositions,
    prefill_current_positions: PrefillStartPositions,
    generation_phase: GenerationPhase,
    pending_output_token_ids: Vec<u32>,
    prefill_duration: Duration,
    decode_duration: Duration,
}

impl RequestExecutionState {
    pub fn new(
        request: GenerateTextRequest,
        draft_token_count_controller: Option<DraftTokenCountController>,
        prefill_initial_positions: PrefillStartPositions,
        tokenizer_vocabulary_size: usize,
    ) -> Result<Self> {
        let input_token_count = request.input_token_ids.len();
        validate_prefill_start_positions(input_token_count, prefill_initial_positions)?;
        ensure!(
            draft_token_count_controller.is_some() == prefill_initial_positions.draft.is_some(),
            "draft token-count controller and draft prefill position must both be present or absent"
        );

        let generation_phase =
            determine_initial_phase(input_token_count, prefill_initial_positions);
        let max_new_token_count = usize::try_from(request.max_new_tokens)
            .context("max_new_tokens does not fit in usize")?;
        let (target_sampler, draft_sampler) = create_samplers(&request, tokenizer_vocabulary_size)?;
        Ok(Self {
            request_id: request.request_id,
            tokens: request.input_token_ids,
            input_token_count,
            draft_token_count_state: draft_token_count_controller.map(DraftTokenCountState::new),
            max_new_token_count,
            eos_tokens: request.end_of_sequence_token_ids,
            target_sampler,
            draft_sampler,
            prefill_initial_positions,
            prefill_current_positions: prefill_initial_positions,
            generation_phase,
            pending_output_token_ids: Vec::new(),
            prefill_duration: Duration::ZERO,
            decode_duration: Duration::ZERO,
        })
    }

    pub fn request_id(&self) -> u64 {
        self.request_id
    }

    /// Returns the committed sequence. A model's KV cache may contain only a prefix of it because
    /// the final sampled token has not necessarily passed through that model yet.
    pub fn committed_token_ids(&self) -> &[u32] {
        &self.tokens
    }

    pub fn phase(&self) -> GenerationPhase {
        self.generation_phase
    }

    fn prepare_forward(
        &mut self,
        token_budget: usize,
        use_speculative_decoding: bool,
    ) -> Result<PreparedForward> {
        ensure!(token_budget > 0, "token budget must be greater than zero");
        self.pending_output_token_ids.clear();
        if !use_speculative_decoding {
            let (token_range, phase) = self.prepare_target_only_forward(token_budget)?;
            return Ok(PreparedForward::Target { token_range, phase });
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
            GenerationPhase::Finished { .. } => {
                bail!("cannot prepare a finished request (speculative decoding)")
            }
        }
    }

    /// Selects the next input chunk and records how its logits must advance this request.
    fn prepare_target_only_forward(
        &mut self,
        token_budget: usize,
    ) -> Result<(TokenRange, PreparedForwardPhase)> {
        let started_at = Instant::now();
        let (token_range, prepared_phase) = match self.generation_phase {
            GenerationPhase::Prefill { .. } => {
                let start_position = self.prefill_current_positions.target;
                let end_position = start_position
                    .saturating_add(token_budget)
                    .min(self.input_token_count);
                (
                    start_position..end_position,
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
                    start_position..self.tokens.len(),
                    PreparedForwardPhase::Decode {
                        generated_token_count,
                        started_at,
                    },
                )
            }
            GenerationPhase::Finished { .. } => {
                bail!("cannot prepare a finished request (target-only)")
            }
        };
        Ok((TokenRange(token_range), prepared_phase))
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
        let token_range = |start_position| TokenRange(start_position..chunk_end_position);
        Ok(PreparedSpeculativePrefill {
            target: (target_position < chunk_end_position).then(|| token_range(target_position)),
            draft: (draft_position < chunk_end_position).then(|| token_range(draft_position)),
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
            maximum_draft_token_count: self
                .draft_token_count_state
                .as_ref()
                .expect("speculative decoding requires draft token-count state")
                .controller
                .draft_token_count()
                .min(remaining_max_token_count),
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

    fn complete_speculative_decode(
        &mut self,
        prepared_decode: PreparedSpeculativeDecode,
        draft_proposal: &DraftProposal,
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
            prepared_decode.original_cached_token_count,
            draft_proposal,
            verification_logits,
        )?;
        self.decode_duration += prepared_decode.started_at.elapsed();
        let generated_token_count = prepared_decode.generated_token_count + committed_token_count;
        self.generation_phase = self.determine_decode_phase(generated_token_count, should_continue);
        Ok(GenerationStep {
            output_token_ids: std::mem::take(&mut self.pending_output_token_ids),
            generation_phase: self.generation_phase,
        })
    }

    pub fn into_generation_output(self) -> Result<GenerationOutput> {
        let GenerationPhase::Finished { finish_reason } = self.generation_phase else {
            bail!("generation request is not finished")
        };
        let output_token_count = self.tokens.len() - self.input_token_count;
        let execution_stats = GenerationExecutionStats {
            finish_reason,
            input_token_count: u64::try_from(self.input_token_count)
                .context("input token count does not fit in u64")?,
            output_token_count: u64::try_from(output_token_count)
                .context("output token count does not fit in u64")?,
            draft_stats: self
                .draft_token_count_state
                .map(|state| DraftGenerationStats {
                    accepted_token_count: state.accepted_token_count,
                    proposed_token_count: state.proposed_token_count,
                    selected_token_count_histogram: state.selected_token_count_histogram,
                }),
            prefill_duration: self.prefill_duration,
            decode_duration: self.decode_duration,
        };
        Ok(GenerationOutput {
            committed_token_ids: self.tokens,
            execution_stats,
            restored_prefix_token_counts: PrefixTokenCounts {
                target: self.prefill_initial_positions.target,
                draft: self.prefill_initial_positions.draft,
            },
        })
    }

    fn complete_speculative_iteration(
        &mut self,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
        original_cached_token_count: usize,
        draft_proposal: &DraftProposal,
        verification_logits: &Tensor,
    ) -> Result<DecodeIterationResult> {
        let draft_tokens = &draft_proposal.token_ids;
        let DraftVerificationResult {
            accepted_token_count,
            maybe_replacement_token,
        } = self.verify_draft_tokens(draft_proposal, verification_logits)?;
        self.draft_token_count_state
            .as_mut()
            .expect("speculative decoding requires draft token-count state")
            .record_verification(accepted_token_count, draft_tokens.len())?;

        // Discard cache entries derived from rejected proposals and retain exactly the verified
        // input prefix needed for the next decode iteration.
        let replacement_token_count = usize::from(maybe_replacement_token.is_some());
        let retained_cached_token_count =
            original_cached_token_count + accepted_token_count + replacement_token_count;
        target.truncate_cache(self.request_id, retained_cached_token_count)?;
        draft.truncate_cache(self.request_id, retained_cached_token_count)?;

        // Publish the accepted draft prefix, then the replacement after a rejection.
        // EOS is observed but never added to the generated output.
        Ok(self.commit_speculative_tokens(
            &draft_tokens[..accepted_token_count],
            maybe_replacement_token,
        ))
    }

    fn verify_draft_tokens(
        &mut self,
        draft_proposal: &DraftProposal,
        verification_logits: &Tensor,
    ) -> Result<DraftVerificationResult> {
        let draft_tokens = &draft_proposal.token_ids;
        let mut accepted_token_count = 0;
        for (position, &draft_token) in draft_tokens.iter().enumerate() {
            let logits = verification_logits.i(position)?;
            let appended_tokens = &draft_tokens[..position];
            let maybe_replacement_token = if self.target_sampler.is_greedy() {
                self.target_sampler.verify_greedy_draft_token(
                    draft_token,
                    &logits,
                    &self.tokens,
                    appended_tokens,
                )?
            } else {
                self.target_sampler.verify_stochastic_draft_token(
                    draft_token,
                    &logits,
                    &self.tokens,
                    appended_tokens,
                    &draft_proposal.probabilities[position],
                )?
            };
            if let Some(replacement_token) = maybe_replacement_token {
                return Ok(DraftVerificationResult {
                    accepted_token_count,
                    maybe_replacement_token: Some(replacement_token),
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

    fn sample_and_commit_target_token(&mut self, logits: &Tensor) -> Result<bool> {
        let logits = logits.to_dtype(DType::F32)?;
        let next_token =
            self.target_sampler
                .sample(&logits, &self.tokens, /* appended_tokens */ &[])?;
        Ok(self.commit_next_token(next_token))
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
        if !should_continue {
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::EndOfSequenceToken,
            }
        } else if generated_token_count < self.max_new_token_count {
            GenerationPhase::Decode {
                generated_token_count,
            }
        } else {
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::MaxNewTokensReached,
            }
        }
    }
}

struct RequestExecution {
    execution_state: Box<RequestExecutionState>,
    token_budget: usize,
}

/// Owns the request states selected for one batched model execution.
pub struct RequestExecutionBatch {
    requests: Vec<RequestExecution>,
}

impl RequestExecutionBatch {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            requests: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, execution_state: Box<RequestExecutionState>, token_budget: usize) {
        self.requests.push(RequestExecution {
            execution_state,
            token_budget,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    pub fn len(&self) -> usize {
        self.requests.len()
    }

    pub fn into_execution_states(self) -> impl Iterator<Item = Box<RequestExecutionState>> {
        self.requests
            .into_iter()
            .map(|request| request.execution_state)
    }

    pub fn execution_states(&self) -> impl Iterator<Item = &RequestExecutionState> {
        self.requests
            .iter()
            .map(|request| request.execution_state.as_ref())
    }

    pub fn run_steps(
        &mut self,
        target: &mut ModelInstance,
        draft: Option<&mut ModelInstance>,
    ) -> Result<Vec<GenerationStep>> {
        match draft {
            Some(draft) => self.run_speculative_decoding_steps(target, draft),
            None => self.run_target_only_steps(target),
        }
    }

    fn run_target_only_steps(&mut self, target: &mut ModelInstance) -> Result<Vec<GenerationStep>> {
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
        let ModelInputs {
            target_request_indices,
            target_inputs,
            draft_inputs: _,
            draft_proposals: _,
        } = self.create_initial_model_inputs(&prepared_forwards, target, None)?;

        let mut target_logits = vec![None; self.requests.len()];
        let logits = target.forward(&target_inputs)?;
        ensure!(
            logits.len() == target_inputs.len(),
            "target model forward returned {} outputs for {} requests",
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

    fn run_speculative_decoding_steps(
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
        let ModelInputs {
            target_request_indices,
            target_inputs: generation_inputs,
            draft_inputs,
            mut draft_proposals,
        } = self.create_initial_model_inputs(&prepared_forwards, target, Some(draft))?;

        // 2. Run all draft-prefill work, then generate every request's remaining proposals.
        if !draft_inputs.is_empty() {
            let logits = draft.forward(&draft_inputs)?;
            ensure!(
                logits.len() == draft_inputs.len(),
                "draft model forward returned {} outputs for {} requests",
                logits.len(),
                draft_inputs.len()
            );
        }
        self.generate_draft_proposals(draft, &mut draft_proposals)?;

        // 3. Build target-verification inputs from the completed draft proposals.
        let verification_inputs =
            self.create_target_verification_inputs(target, &draft_proposals)?;

        // 4. Run one combined target-model pass containing ordinary prefill, ordinary decode,
        // speculative target prefill, and speculative verification.
        let target_logits = self.run_combined_target_forward(
            target,
            target_request_indices,
            generation_inputs,
            verification_inputs,
        )?;
        self.complete_speculative_steps(
            prepared_forwards,
            draft_proposals,
            target_logits,
            target,
            draft,
        )
    }

    fn create_initial_model_inputs(
        &self,
        prepared_forwards: &[PreparedForward],
        target: &ModelInstance,
        draft: Option<&ModelInstance>,
    ) -> Result<ModelInputs> {
        let mut target_request_indices = Vec::new();
        let mut target_inputs = Vec::new();
        let mut draft_inputs = Vec::new();
        let mut draft_proposals = Vec::new();
        for (request_index, (request, prepared_forward)) in
            self.requests.iter().zip(prepared_forwards).enumerate()
        {
            let request_id = request.execution_state.request_id();
            match prepared_forward {
                PreparedForward::Target { token_range, .. } => {
                    target_request_indices.push(request_index);
                    target_inputs.push(ForwardInput {
                        request_id,
                        input: target.create_input_tensor(
                            token_range.token_ids(&request.execution_state.tokens),
                        )?,
                        start_position: token_range.start_position(),
                    });
                }
                PreparedForward::SpeculativePrefill(prepared) => {
                    if let Some(target_token_range) = &prepared.target {
                        target_request_indices.push(request_index);
                        target_inputs.push(ForwardInput {
                            request_id,
                            input: target.create_input_tensor(
                                target_token_range.token_ids(&request.execution_state.tokens),
                            )?,
                            start_position: target_token_range.start_position(),
                        });
                    }
                    if let Some(draft_token_range) = &prepared.draft {
                        let draft = draft.context("draft model is missing")?;
                        draft_inputs.push(ForwardInput {
                            request_id,
                            input: draft.create_input_tensor(
                                draft_token_range.token_ids(&request.execution_state.tokens),
                            )?,
                            start_position: draft_token_range.start_position(),
                        });
                    }
                }
                PreparedForward::SpeculativeDecode(prepared) => {
                    draft_proposals.push(DraftProposal {
                        request_index,
                        original_cached_token_count: prepared.original_cached_token_count,
                        maximum_token_count: prepared.maximum_draft_token_count,
                        token_ids: Vec::new(),
                        probabilities: Vec::new(),
                    });
                }
            }
        }
        Ok(ModelInputs {
            target_request_indices,
            target_inputs,
            draft_inputs,
            draft_proposals,
        })
    }

    fn generate_draft_proposals(
        &mut self,
        draft: &mut ModelInstance,
        proposals: &mut [DraftProposal],
    ) -> Result<()> {
        loop {
            let proposal_indices = self.extract_active_draft_proposal_indices(proposals);
            if proposal_indices.is_empty() {
                return Ok(());
            }
            let proposal_inputs =
                self.create_draft_proposal_inputs(draft, proposals, &proposal_indices)?;
            let logits = draft.forward(&proposal_inputs)?;
            self.sample_and_append_draft_proposals(proposals, &proposal_indices, logits)?;
        }
    }

    fn extract_active_draft_proposal_indices(&self, proposals: &[DraftProposal]) -> Vec<usize> {
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
        proposals: &[DraftProposal],
        active_proposal_indices: &[usize],
    ) -> Result<Vec<ForwardInput>> {
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
                Ok(ForwardInput {
                    request_id: execution_state.request_id,
                    input: draft.create_input_tensor(&[input_token_id])?,
                    start_position: proposal.original_cached_token_count + proposal.token_ids.len(),
                })
            })
            .collect()
    }

    fn sample_and_append_draft_proposals(
        &mut self,
        proposals: &mut [DraftProposal],
        active_proposal_indices: &[usize],
        logits: Vec<Tensor>,
    ) -> Result<()> {
        ensure!(
            logits.len() == active_proposal_indices.len(),
            "draft model forward returned {} outputs for {} requests",
            logits.len(),
            active_proposal_indices.len()
        );
        for (proposal_index, logits) in active_proposal_indices.iter().copied().zip(logits) {
            let proposal = &mut proposals[proposal_index];
            let execution_state = &mut self.requests[proposal.request_index].execution_state;
            let logits = logits.to_dtype(DType::F32)?;
            let next_token = if execution_state.draft_sampler.is_greedy() {
                execution_state.draft_sampler.sample(
                    &logits,
                    &execution_state.tokens,
                    &proposal.token_ids,
                )?
            } else {
                let SampleWithProbabilities {
                    token_id,
                    probabilities,
                } = execution_state.draft_sampler.sample_with_probabilities(
                    &logits,
                    &execution_state.tokens,
                    &proposal.token_ids,
                )?;
                proposal.probabilities.push(probabilities);
                token_id
            };
            proposal.token_ids.push(next_token);
        }
        Ok(())
    }

    fn create_target_verification_inputs(
        &self,
        target: &ModelInstance,
        proposals: &[DraftProposal],
    ) -> Result<Vec<TargetVerificationInput>> {
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
            verification_inputs.push(TargetVerificationInput {
                request_index: proposal.request_index,
                input: ForwardInput {
                    request_id: execution_state.request_id,
                    input: target.create_input_tensor(&input_token_ids)?,
                    start_position: proposal.original_cached_token_count,
                },
            });
        }
        Ok(verification_inputs)
    }

    fn run_combined_target_forward(
        &self,
        target: &mut ModelInstance,
        generation_request_indices: Vec<usize>,
        generation_inputs: Vec<ForwardInput>,
        verification_inputs: Vec<TargetVerificationInput>,
    ) -> Result<TargetLogits> {
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
            let output = target
                .forward_with_speculative_verification(&generation_inputs, &verification_inputs)?;
            ensure!(
                output.generation_logits.len() == generation_inputs.len(),
                "target model forward returned {} outputs for {} requests",
                output.generation_logits.len(),
                generation_inputs.len()
            );
            ensure!(
                output.verification_logits.len() == verification_inputs.len(),
                "target verification returned {} outputs for {} requests",
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

        Ok(TargetLogits {
            generation,
            verification,
        })
    }

    fn complete_speculative_steps(
        &mut self,
        prepared_forwards: Vec<PreparedForward>,
        draft_proposals: Vec<DraftProposal>,
        mut target_logits: TargetLogits,
        target: &mut ModelInstance,
        draft: &mut ModelInstance,
    ) -> Result<Vec<GenerationStep>> {
        let mut proposals = std::iter::repeat_with(|| None)
            .take(self.requests.len())
            .collect::<Vec<_>>();
        for proposal in draft_proposals {
            let request_index = proposal.request_index;
            proposals[request_index] = Some(proposal);
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
                    request.execution_state.complete_speculative_decode(
                        prepared,
                        &proposals[request_index]
                            .take()
                            .context("draft proposal is missing")?,
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

fn create_samplers(
    request: &GenerateTextRequest,
    tokenizer_vocabulary_size: usize,
) -> Result<(Sampler, Sampler)> {
    let seed = request.seed.unwrap_or_else(rand::random);
    let repeat_last_n =
        usize::try_from(request.repeat_last_n).context("repeat_last_n does not fit in usize")?;
    // Keep the target's existing seed sequence and give the draft its own RNG stream.
    Ok((
        Sampler::new(
            tokenizer_vocabulary_size,
            request.temperature,
            seed,
            request.repeat_penalty,
            repeat_last_n,
        ),
        Sampler::new(
            tokenizer_vocabulary_size,
            request.temperature,
            seed.wrapping_add(1),
            request.repeat_penalty,
            repeat_last_n,
        ),
    ))
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
    use crate::model_runner::server::generation::draft_token_count::DraftTokenCountPolicy;
    use crate::model_runner::server::kv_cache::create_kv_cache;
    use crate::model_runner::server::BackendId;
    use crate::model_runner::KvCacheType;
    use crate::models::loaded_model::LoadedModel;
    use crate::models::CausalLanguageModel;
    use crate::models::ForwardInput;
    use crate::models::ForwardOutput;
    use crate::models::KvCache;
    use crate::models::ModelInfo;
    use crate::models::ModelRole;
    use candle_core::Device;
    use std::sync::Arc;
    use std::sync::Mutex;

    #[test]
    fn configures_both_samplers_with_temperature_and_independent_seeded_streams() -> Result<()> {
        let request = GenerateTextRequest {
            temperature: 0.7,
            repeat_penalty: 1.0,
            seed: Some(0),
            ..Default::default()
        };
        let (mut target, mut draft) = create_samplers(&request, 3)?;
        let mut reference_target = Sampler::new(3, 0.7, 0, 1.0, 0);
        let mut reference_draft = Sampler::new(3, 0.7, 1, 1.0, 0);
        let logits = Tensor::new(&[0.0f32, 0.5, 1.0], &Device::Cpu)?;
        for _ in 0..32 {
            assert_eq!(
                draft.sample(&logits, &[], &[])?,
                reference_draft.sample(&logits, &[], &[])?
            );
            assert_eq!(
                target.sample(&logits, &[], &[])?,
                reference_target.sample(&logits, &[], &[])?
            );
        }
        Ok(())
    }

    fn acceptance_rate_controller() -> DraftTokenCountController {
        DraftTokenCountController::new(DraftTokenCountPolicy::AcceptanceRate {
            initial_draft_token_count: 4,
            decrease_threshold: 0.4,
            increase_threshold: 0.8,
            minimum_draft_token_count: 1,
            maximum_draft_token_count: 8,
        })
    }

    #[test]
    fn records_policy_selected_counts_before_output_caps() -> Result<()> {
        let mut state = DraftTokenCountState::new(DraftTokenCountController::new(
            DraftTokenCountPolicy::Fixed {
                draft_token_count: 4,
            },
        ));

        state.record_verification(2, 2)?;

        assert_eq!(state.accepted_token_count, 2);
        assert_eq!(state.proposed_token_count, 2);
        assert_eq!(state.selected_token_count_histogram.len(), 1);
        assert_eq!(state.selected_token_count_histogram.get(&4), Some(&1));
        Ok(())
    }

    #[test]
    fn records_sorted_adaptive_policy_histogram() -> Result<()> {
        let mut state = DraftTokenCountState::new(acceptance_rate_controller());

        state.record_verification(1, 4)?;
        state.record_verification(3, 3)?;
        state.record_verification(4, 4)?;

        assert_eq!(state.controller.draft_token_count(), 5);
        assert_eq!(state.accepted_token_count, 8);
        assert_eq!(state.proposed_token_count, 11);
        assert_eq!(
            state
                .selected_token_count_histogram
                .iter()
                .map(|(&count, &uses)| (count, uses))
                .collect::<Vec<_>>(),
            vec![(3, 1), (4, 2)]
        );
        Ok(())
    }

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

        fn forward_with_speculative_verification(
            &mut self,
            generation_inputs: &[ForwardInput],
            verification_inputs: &[ForwardInput],
            _kv_cache: &mut dyn KvCache,
        ) -> Result<ForwardOutput> {
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
            Ok(ForwardOutput {
                generation_logits,
                verification_logits,
            })
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
                context_length: 128,
            },
            next_token,
            forward_calls: Arc::clone(&forward_calls),
        }));
        let kv_cache = create_kv_cache(
            BackendId::from_index(0),
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
            draft_token_count_state: Some(DraftTokenCountState::new(
                DraftTokenCountController::new(DraftTokenCountPolicy::Fixed {
                    draft_token_count: 4,
                }),
            )),
            max_new_token_count: 16,
            eos_tokens,
            target_sampler: Sampler::new(3, 0.0, 0, 1.0, 64),
            draft_sampler: Sampler::new(3, 0.0, 1, 1.0, 64),
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
        }
    }

    struct CachedForwardCall {
        request_id: u64,
        start_position: usize,
        cached_tokens: Vec<u32>,
    }

    /// Uses the previous token to choose logits, and stores actual input tokens in the KV cache.
    struct SamplingTestModel {
        info: ModelInfo,
        logits: Vec<Vec<f32>>,
        calls: Arc<Mutex<Vec<CachedForwardCall>>>,
    }

    impl CausalLanguageModel for SamplingTestModel {
        fn info(&self) -> &ModelInfo {
            &self.info
        }

        fn forward_with_speculative_verification(
            &mut self,
            generation_inputs: &[ForwardInput],
            verification_inputs: &[ForwardInput],
            kv_cache: &mut dyn KvCache,
        ) -> Result<ForwardOutput> {
            let mut forward = |input: &ForwardInput, verification: bool| -> Result<Tensor> {
                let cached_tokens = if input.start_position == 0 {
                    Vec::new()
                } else {
                    kv_cache
                        .get_contiguous_cache_tensors(input.request_id, 0)?
                        .key
                        .flatten_all()?
                        .to_vec1::<u32>()?
                };
                self.calls.lock().unwrap().push(CachedForwardCall {
                    request_id: input.request_id,
                    start_position: input.start_position,
                    cached_tokens,
                });
                let tokens = input.input.flatten_all()?.to_vec1::<u32>()?;
                let values = input.input.reshape((1, 1, tokens.len(), 1))?;
                kv_cache.append_new_key_value(
                    input.request_id,
                    0,
                    input.start_position,
                    &values,
                    &values,
                )?;
                let logits: Vec<Vec<f32>> = tokens
                    .iter()
                    .map(|&token| self.logits[token as usize].clone())
                    .collect();
                if verification {
                    Ok(Tensor::new(logits, &Device::Cpu)?)
                } else {
                    Ok(Tensor::new(
                        logits.last().unwrap().as_slice(),
                        &Device::Cpu,
                    )?)
                }
            };
            Ok(ForwardOutput {
                generation_logits: generation_inputs
                    .iter()
                    .map(|input| forward(input, false))
                    .collect::<Result<_>>()?,
                verification_logits: verification_inputs
                    .iter()
                    .map(|input| forward(input, true))
                    .collect::<Result<_>>()?,
            })
        }
    }

    fn create_sampling_test_model(
        logits: Vec<Vec<f32>>,
        cache_type: KvCacheType,
    ) -> Result<(ModelInstance, Arc<Mutex<Vec<CachedForwardCall>>>)> {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let model = LoadedModel::for_test(Box::new(SamplingTestModel {
            info: ModelInfo {
                layer_count: 1,
                num_kv_heads: 1,
                head_dim: 1,
                activation_dtype: DType::U32,
                context_length: 128,
            },
            logits,
            calls: Arc::clone(&calls),
        }));
        let cache = create_kv_cache(
            BackendId::from_index(0),
            cache_type,
            &model,
            ModelRole::Target,
            1024,
        )?;
        Ok((ModelInstance::new(model, cache), calls))
    }

    fn forced_logits(next_tokens: &[usize]) -> Vec<Vec<f32>> {
        next_tokens
            .iter()
            .map(|&next_token| {
                let mut logits = vec![f32::NEG_INFINITY; next_tokens.len()];
                logits[next_token] = 0.0;
                logits
            })
            .collect()
    }

    #[test]
    fn speculative_rejection_rolls_back_both_cache_types_and_resumes() -> Result<()> {
        for cache_type in [
            KvCacheType::Contiguous,
            KvCacheType::Paged {
                per_page_token_count: 2,
                enable_prefix_caching: true,
            },
        ] {
            for temperature in [0.0, 0.7] {
                // Accept token 1, then reject draft token 3 in favor of target token 2.
                let (mut target, target_calls) =
                    create_sampling_test_model(forced_logits(&[1, 2, 1, 1]), cache_type)?;
                let (mut draft, draft_calls) =
                    create_sampling_test_model(forced_logits(&[1, 3, 1, 1]), cache_type)?;
                target.start_request(1)?;
                draft.start_request(1)?;
                let mut state = create_test_execution_state(vec![0], vec![]);
                state.target_sampler = Sampler::new(4, temperature, 42, 1.0, 64);
                state.draft_sampler = Sampler::new(4, temperature, 43, 1.0, 64);
                state.max_new_token_count = 5;
                let mut batch = RequestExecutionBatch::with_capacity(1);
                batch.push(Box::new(state), 1);
                for expected in [vec![1, 2], vec![1, 2], vec![1]] {
                    let step = batch.run_steps(&mut target, Some(&mut draft))?.remove(0);
                    assert_eq!(step.output_token_ids, expected);
                }
                let state = batch.into_execution_states().next().unwrap();
                assert_eq!(state.tokens, [0, 1, 2, 1, 2, 1]);
                assert!(matches!(
                    state.phase(),
                    GenerationPhase::Finished {
                        finish_reason: GenerationFinishReason::MaxNewTokensReached
                    }
                ));
                let output = state.into_generation_output()?;
                let stats = output.execution_stats.draft_stats.unwrap();
                assert_eq!(stats.accepted_token_count, 3);
                assert_eq!(stats.proposed_token_count, 8); // 4 + 3 + 1, capped by remaining output.
                for calls in [target_calls, draft_calls] {
                    let calls = calls.lock().unwrap();
                    let resumed = calls.iter().find(|call| call.start_position == 2).unwrap();
                    assert_eq!(resumed.cached_tokens, [0, 1]);
                    let resumed = calls.iter().find(|call| call.start_position == 4).unwrap();
                    assert_eq!(resumed.cached_tokens, [0, 1, 2, 1]);
                    assert_eq!(resumed.request_id, 1);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn stochastic_full_acceptance_uses_matching_penalties_and_seeded_draft_samples() -> Result<()> {
        let logits = vec![
            vec![0.2f32, 1.2, -0.3],
            vec![0.3, -0.2, 1.4],
            vec![1.1, 0.4, -0.1],
        ];
        for temperature in [0.00001, 0.7, 2.0] {
            for cache_type in [
                KvCacheType::Contiguous,
                KvCacheType::Paged {
                    per_page_token_count: 2,
                    enable_prefix_caching: true,
                },
            ] {
                let (mut target, target_calls) =
                    create_sampling_test_model(logits.clone(), cache_type)?;
                let (mut draft, _) = create_sampling_test_model(logits.clone(), cache_type)?;
                target.start_request(1)?;
                draft.start_request(1)?;
                let request = GenerateTextRequest {
                    request_id: 1,
                    input_token_ids: vec![0, 1],
                    max_new_tokens: 5,
                    temperature,
                    seed: Some(42),
                    repeat_penalty: 1.7,
                    repeat_last_n: 3,
                    ..Default::default()
                };
                let state = RequestExecutionState::new(
                    request,
                    Some(DraftTokenCountController::new(
                        DraftTokenCountPolicy::Fixed {
                            draft_token_count: 4,
                        },
                    )),
                    PrefillStartPositions {
                        target: 0,
                        draft: Some(0),
                    },
                    3,
                )?;
                let mut reference = Sampler::new(3, temperature, 43, 1.7, 3);
                let mut expected_tokens = vec![0, 1];
                for _ in 0..5 {
                    let next = reference.sample(
                        &Tensor::new(
                            logits[*expected_tokens.last().unwrap() as usize].as_slice(),
                            &Device::Cpu,
                        )?,
                        &expected_tokens,
                        &[],
                    )?;
                    expected_tokens.push(next);
                }
                let mut batch = RequestExecutionBatch::with_capacity(1);
                batch.push(Box::new(state), 1);
                let prefill = batch.run_steps(&mut target, Some(&mut draft))?.remove(0);
                assert!(prefill.output_token_ids.is_empty());
                let first = batch.run_steps(&mut target, Some(&mut draft))?.remove(0);
                assert_eq!(first.output_token_ids, expected_tokens[2..6]);
                let second = batch.run_steps(&mut target, Some(&mut draft))?.remove(0);
                assert_eq!(second.output_token_ids, expected_tokens[6..]);
                let output = batch
                    .into_execution_states()
                    .next()
                    .unwrap()
                    .into_generation_output()?;
                assert_eq!(output.committed_token_ids, expected_tokens);
                let stats = output.execution_stats.draft_stats.unwrap();
                assert_eq!(stats.accepted_token_count, 5);
                assert_eq!(stats.proposed_token_count, 5);
                // The final accepted token is pending, and both models consume it next iteration.
                let calls = target_calls.lock().unwrap();
                assert_eq!(calls.last().unwrap().start_position, 5);
                assert_eq!(calls.last().unwrap().cached_tokens, expected_tokens[..5]);
            }
        }
        Ok(())
    }

    #[test]
    fn stochastic_speculation_stops_at_accepted_and_replacement_eos() -> Result<()> {
        for accepted_eos in [false, true] {
            let cache_type = KvCacheType::Paged {
                per_page_token_count: 2,
                enable_prefix_caching: true,
            };
            let (mut target, _) =
                create_sampling_test_model(forced_logits(&[1, 2, 1, 1]), cache_type)?;
            let (mut draft, draft_calls) = create_sampling_test_model(
                forced_logits(&[1, if accepted_eos { 2 } else { 3 }, 1, 1]),
                cache_type,
            )?;
            target.start_request(1)?;
            draft.start_request(1)?;
            let mut state = create_test_execution_state(vec![0], vec![2]);
            state.target_sampler = Sampler::new(4, 0.7, 42, 1.0, 64);
            state.draft_sampler = Sampler::new(4, 0.7, 43, 1.0, 64);
            let mut batch = RequestExecutionBatch::with_capacity(1);
            batch.push(Box::new(state), 1);
            let step = batch.run_steps(&mut target, Some(&mut draft))?.remove(0);
            assert_eq!(step.output_token_ids, [1]);
            assert!(matches!(
                step.generation_phase,
                GenerationPhase::Finished {
                    finish_reason: GenerationFinishReason::EndOfSequenceToken
                }
            ));
            let output = batch
                .into_execution_states()
                .next()
                .unwrap()
                .into_generation_output()?;
            assert_eq!(output.committed_token_ids, [0, 1]);
            let stats = output.execution_stats.draft_stats.unwrap();
            assert_eq!(stats.accepted_token_count, if accepted_eos { 2 } else { 1 });
            assert_eq!(stats.proposed_token_count, if accepted_eos { 2 } else { 4 });
            assert_eq!(
                draft_calls.lock().unwrap().len(),
                if accepted_eos { 2 } else { 4 }
            );
            // Completing the request publishes only committed prefix pages, even with EOS.
            target.finish_request(1, &output.committed_token_ids)?;
            draft.finish_request(1, &output.committed_token_ids)?;
        }
        Ok(())
    }

    #[test]
    fn seeded_speculative_requests_are_independent_of_batch_composition() -> Result<()> {
        let cache_type = KvCacheType::Paged {
            per_page_token_count: 2,
            enable_prefix_caching: true,
        };
        let run = |request_ids: &[u64]| -> Result<BTreeMap<u64, Vec<u32>>> {
            let (mut target, _) = create_sampling_test_model(
                vec![
                    vec![0.3, 1.4, -0.2],
                    vec![1.1, -0.4, 0.5],
                    vec![0.2, -0.3, 1.2],
                ],
                cache_type,
            )?;
            let (mut draft, _) = create_sampling_test_model(
                vec![
                    vec![1.1, -0.2, 0.4],
                    vec![-0.1, 1.3, 0.2],
                    vec![0.9, 0.4, -0.3],
                ],
                cache_type,
            )?;
            let mut batch = RequestExecutionBatch::with_capacity(request_ids.len());
            for &id in request_ids {
                target.start_request(id)?;
                draft.start_request(id)?;
                let request = GenerateTextRequest {
                    request_id: id,
                    input_token_ids: vec![0, 1, 2][..id as usize].to_vec(),
                    max_new_tokens: 12,
                    temperature: if id == 1 { 0.0 } else { 0.7 },
                    seed: Some(42),
                    repeat_penalty: 1.5,
                    repeat_last_n: 3,
                    ..Default::default()
                };
                let state = RequestExecutionState::new(
                    request,
                    Some(DraftTokenCountController::new(
                        DraftTokenCountPolicy::Fixed {
                            draft_token_count: 4,
                        },
                    )),
                    PrefillStartPositions {
                        target: 0,
                        draft: Some(0),
                    },
                    3,
                )?;
                batch.push(Box::new(state), 1);
            }
            let mut outputs = BTreeMap::new();
            while !batch.is_empty() {
                batch.run_steps(&mut target, Some(&mut draft))?;
                let mut next_batch = RequestExecutionBatch::with_capacity(batch.len());
                for state in batch.into_execution_states() {
                    if matches!(state.phase(), GenerationPhase::Finished { .. }) {
                        let id = state.request_id();
                        let output = state.into_generation_output()?;
                        assert_eq!(output.execution_stats.output_token_count, 12);
                        let stats = output.execution_stats.draft_stats.unwrap();
                        if id != 1 {
                            assert!(stats.accepted_token_count > 0);
                        }
                        assert!(stats.accepted_token_count < stats.proposed_token_count);
                        target.finish_request(id, &output.committed_token_ids)?;
                        draft.finish_request(id, &output.committed_token_ids)?;
                        outputs.insert(id, output.committed_token_ids);
                    } else {
                        next_batch.push(state, 1);
                    }
                }
                batch = next_batch;
            }
            Ok(outputs)
        };
        let batched = run(&[1, 2, 3])?;
        assert_eq!(batched, run(&[3, 2, 1])?);
        for id in [1, 2, 3] {
            assert_eq!(batched[&id], run(&[id])?[&id]);
        }
        Ok(())
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
            None,
            PrefillStartPositions {
                target: 2,
                draft: None,
            },
            32,
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
            Some(DraftTokenCountController::new(
                DraftTokenCountPolicy::Fixed {
                    draft_token_count: 4,
                },
            )),
            PrefillStartPositions {
                target: 3,
                draft: Some(1),
            },
            32,
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
    fn prepares_and_completes_a_partial_prefill() -> Result<()> {
        let mut execution_state = create_test_execution_state(vec![1, 2, 3, 4], vec![]);
        execution_state.generation_phase = GenerationPhase::Prefill {
            remaining_token_count: 4,
        };

        let (token_range, prepared_phase) = execution_state.prepare_target_only_forward(2)?;
        assert_eq!(token_range.start_position(), 0);
        assert_eq!(token_range.token_ids(&execution_state.tokens), [1, 2]);

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

        let (token_range, prepared_phase) = execution_state.prepare_target_only_forward(8)?;
        assert_eq!(token_range.start_position(), 2);
        assert_eq!(token_range.token_ids(&execution_state.tokens), [3, 4]);

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

        let (token_range, prepared_phase) = execution_state.prepare_target_only_forward(1)?;
        assert_eq!(token_range.start_position(), 3);
        assert_eq!(token_range.token_ids(&execution_state.tokens), [4]);

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

        let steps = batch.run_steps(&mut target, None)?;
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

        let steps = batch.run_steps(&mut target, Some(&mut draft))?;
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
    fn generates_draft_proposals_and_target_verification_inputs() -> Result<()> {
        let execution_state = create_test_execution_state(vec![1, 2], vec![]);
        let (mut draft, draft_forward_calls) = create_test_model(6)?;
        let (target, _) = create_test_model(6)?;
        let mut batch = RequestExecutionBatch::with_capacity(1);
        batch.push(Box::new(execution_state), 1);
        let mut proposals = vec![DraftProposal {
            request_index: 0,
            original_cached_token_count: 1,
            maximum_token_count: 3,
            token_ids: Vec::new(),
            probabilities: Vec::new(),
        }];

        batch.generate_draft_proposals(&mut draft, &mut proposals)?;
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
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::MaxNewTokensReached
            }
        ));
        assert!(matches!(
            execution_state.determine_decode_phase(1, false),
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::EndOfSequenceToken
            }
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
            None,
            PrefillStartPositions {
                target: 3,
                draft: None,
            },
            32,
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
            Some(DraftTokenCountController::new(
                DraftTokenCountPolicy::Fixed {
                    draft_token_count: 4,
                },
            )),
            PrefillStartPositions {
                target: 0,
                draft: Some(3),
            },
            32,
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

        let proposal = DraftProposal {
            request_index: 0,
            original_cached_token_count: 0,
            maximum_token_count: draft_tokens.len(),
            token_ids: draft_tokens.to_vec(),
            probabilities: Vec::new(),
        };
        let result = execution_state.verify_draft_tokens(&proposal, &verification_logits)?;

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

        let proposal = DraftProposal {
            request_index: 0,
            original_cached_token_count: 0,
            maximum_token_count: draft_tokens.len(),
            token_ids: draft_tokens.to_vec(),
            probabilities: Vec::new(),
        };
        let result = execution_state.verify_draft_tokens(&proposal, &verification_logits)?;

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
        assert!(matches!(
            execution_state
                .determine_decode_phase(result.committed_token_count, result.should_continue),
            GenerationPhase::Finished {
                finish_reason: GenerationFinishReason::EndOfSequenceToken
            }
        ));
        assert_eq!(execution_state.tokens, vec![1, 2]);
        assert_eq!(execution_state.pending_output_token_ids, vec![2]);
        Ok(())
    }

    #[test]
    fn generation_output_records_the_finish_reason() {
        for finish_reason in [
            GenerationFinishReason::EndOfSequenceToken,
            GenerationFinishReason::MaxNewTokensReached,
        ] {
            let mut execution_state = create_test_execution_state(vec![1], vec![]);
            execution_state.draft_token_count_state = None;
            execution_state.generation_phase = GenerationPhase::Finished { finish_reason };

            let output = execution_state.into_generation_output().unwrap();

            assert_eq!(output.execution_stats.finish_reason, finish_reason);
        }
    }

    #[test]
    fn rejects_result_before_generation_finishes() {
        let execution_state = create_test_execution_state(vec![1], vec![]);

        let error = execution_state
            .into_generation_output()
            .err()
            .expect("unfinished generation should not produce a result")
            .to_string();

        assert_eq!(error, "generation request is not finished");
    }
}
