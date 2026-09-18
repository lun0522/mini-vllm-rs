mod backends;
pub(crate) mod loaded_model;
pub(crate) mod model_downloader;

use anyhow::Result;
use candle_core::DType;
use candle_core::Tensor;
use std::fmt;

/// Common inference operations implemented by each supported model architecture.
pub(crate) trait CausalLanguageModel: Send {
    fn info(&self) -> &ModelInfo;

    /// Returns one next-token logits tensor per request, each shaped
    /// `(vocabulary_size)`, in input order.
    fn forward_batched(
        &mut self,
        inputs: &[BatchedForwardInput],
        kv_cache: &mut dyn BatchedKvCache,
    ) -> Result<Vec<Tensor>> {
        Ok(self
            .forward_batched_with_speculative_verification(
                inputs,
                /* verification_inputs */ &[],
                kv_cache,
            )?
            .generation_logits)
    }

    /// Runs regular and speculative-verification requests in one model pass.
    ///
    /// Generation outputs contain one `(vocabulary_size)` tensor per `generation_inputs` entry.
    /// Verification outputs contain one `(sequence_length, vocabulary_size)` tensor per
    /// `verification_inputs` entry. Both output groups preserve their respective input order.
    fn forward_batched_with_speculative_verification(
        &mut self,
        generation_inputs: &[BatchedForwardInput],
        verification_inputs: &[BatchedForwardInput],
        kv_cache: &mut dyn BatchedKvCache,
    ) -> Result<BatchedForwardOutput>;
}

/// Provides one request's input and position to a batched model forward pass.
pub(crate) struct BatchedForwardInput {
    pub(crate) request_id: u64,
    pub(crate) input: Tensor,
    pub(crate) start_position: usize,
}

/// Separates outputs with different shapes from one combined batched model pass.
pub(crate) struct BatchedForwardOutput {
    pub(crate) generation_logits: Vec<Tensor>,
    pub(crate) verification_logits: Vec<Tensor>,
}

/// Provides request-specific key and value tensors during batched model execution.
pub(crate) trait BatchedKvCache: Send {
    /// Stores newly computed key/value tensors and returns the complete layer cache for attention.
    fn append(
        &mut self,
        request_id: u64,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue>;
}

pub(crate) struct CachedKeyValue {
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
}

pub(crate) struct ModelInfo {
    pub(crate) layer_count: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) activation_dtype: DType,
}

impl ModelInfo {
    pub(crate) fn kv_cache_bytes_per_token(&self) -> usize {
        // Use the projected KV width rather than the model-wide hidden dimension because GQA
        // stores fewer key/value heads than query heads.
        self.num_kv_heads * self.head_dim * self.activation_dtype.size_in_bytes()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ModelRole {
    Target,
    Draft,
}

impl fmt::Display for ModelRole {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target => formatter.write_str("target"),
            Self::Draft => formatter.write_str("draft"),
        }
    }
}
