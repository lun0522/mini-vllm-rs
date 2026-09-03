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

    /// Discards cached tokens at and after `target_token_count` in every model layer.
    fn truncate(&mut self, target_token_count: usize) -> anyhow::Result<()>;

    fn clear(&mut self);
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

#[derive(Clone, Copy)]
pub(crate) enum ModelArchitecture {
    Llama,
    Qwen2,
}

impl ModelArchitecture {
    pub(crate) fn format_chat_prompt(self, prompt: &str) -> String {
        match self {
            Self::Llama => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
                 {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::Qwen2 => {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
        }
    }

    pub(crate) fn end_of_sequence_tokens(self) -> &'static [&'static str] {
        match self {
            Self::Llama => &["<|eot_id|>", "<|end_of_text|>"],
            Self::Qwen2 => &["<|im_end|>", "<|endoftext|>"],
        }
    }
}
