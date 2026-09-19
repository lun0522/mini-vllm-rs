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
    fn forward(
        &mut self,
        inputs: &[ForwardInput],
        kv_cache: &mut dyn KvCache,
    ) -> Result<Vec<Tensor>> {
        Ok(self
            .forward_with_speculative_verification(
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
    fn forward_with_speculative_verification(
        &mut self,
        generation_inputs: &[ForwardInput],
        verification_inputs: &[ForwardInput],
        kv_cache: &mut dyn KvCache,
    ) -> Result<ForwardOutput>;
}

/// Provides one request's input and position to a model forward pass.
pub(crate) struct ForwardInput {
    pub(crate) request_id: u64,
    pub(crate) input: Tensor,
    pub(crate) start_position: usize,
}

/// Separates outputs with different shapes from one combined batched model pass.
pub(crate) struct ForwardOutput {
    pub(crate) generation_logits: Vec<Tensor>,
    pub(crate) verification_logits: Vec<Tensor>,
}

/// Provides request-specific key and value tensors during model execution.
///
/// Key and value tensors passed to or returned by this interface have shape
/// [1, num_kv_heads, token_count, head_dim]. The leading dimension is the model attention batch
/// axis. It is always 1 because requests share a model pass by being packed along the token axis,
/// while cache storage and attention are handled one request at a time. Keeping the singleton batch
/// axis preserves the model's rank-4 K/V layout.
///
/// Paged-cache pools have an additional leading physical-page axis and are shaped
/// [page_count, /* batch */ 1, num_kv_heads, per_page_token_count, head_dim].
pub(crate) trait KvCache: Send {
    fn append_new_key_value(
        &mut self,
        request_id: u64,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()>;

    fn get_contiguous_cache_tensors(
        &mut self,
        request_id: u64,
        layer_index: usize,
    ) -> Result<ContiguousCacheTensors>;

    fn get_paged_cache_layout(
        &mut self,
        request_id: u64,
        layer_index: usize,
    ) -> Result<Option<PagedCacheLayout>>;
}

pub(crate) struct ContiguousCacheTensors {
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
}

pub(crate) struct PagedCacheLayout {
    pub(crate) key_pool: Tensor,
    pub(crate) value_pool: Tensor,
    pub(crate) page_ids: Vec<usize>,
    pub(crate) per_page_token_count: usize,
    pub(crate) cached_token_count: usize,
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
