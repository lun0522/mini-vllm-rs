pub(crate) mod loaded_model;
pub(crate) mod model_downloader;
mod models;

use candle_core::DType;
use candle_core::Tensor;
use std::fmt;

/// Common inference operations implemented by each supported model architecture.
pub(crate) trait CausalLanguageModel: Send {
    fn info(&self) -> &ModelInfo;

    /// Returns next-token logits shaped `(batch_size, 1, vocabulary_size)`.
    fn forward(
        &mut self,
        input: &Tensor,
        start_position: usize,
        kv_cache: &mut dyn KvCache,
    ) -> candle_core::Result<Tensor>;

    /// Returns logits for every input position, shaped
    /// `(batch_size, sequence_length, vocabulary_size)`. Speculative decoding uses these logits
    /// to verify multiple draft tokens with one target-model forward pass.
    fn forward_for_speculative_verification(
        &mut self,
        input: &Tensor,
        start_position: usize,
        kv_cache: &mut dyn KvCache,
    ) -> candle_core::Result<Tensor>;
}

/// Provides request-specific key and value tensors to model layers.
pub(crate) trait KvCache: Send {
    /// Stores newly computed key/value tensors and returns the complete layer cache for attention.
    fn append(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> anyhow::Result<CachedKeyValue>;
}

pub(crate) struct CachedKeyValue {
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
}

pub(crate) struct ModelInfo {
    pub(crate) layer_count: usize,
    pub(crate) kv_head_count: usize,
    pub(crate) head_dimension: usize,
    pub(crate) activation_dtype: DType,
}

impl ModelInfo {
    pub(crate) fn kv_cache_bytes_per_token(&self) -> usize {
        // Use the projected KV width rather than the model-wide hidden dimension because GQA
        // stores fewer key/value heads than query heads.
        self.kv_head_count * self.head_dimension * self.activation_dtype.size_in_bytes()
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
