pub(crate) mod loaded_model;
pub(crate) mod model_downloader;
mod models;

use candle_core::Tensor;
use std::fmt;

/// Request-specific key and value tensors, indexed by transformer layer.
pub(crate) struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KvCache {
    pub(crate) fn new(layer_count: usize) -> Self {
        Self {
            layers: vec![None; layer_count],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.layers.fill(None);
    }

    pub(crate) fn layer(&mut self, layer_index: usize) -> &mut Option<(Tensor, Tensor)> {
        &mut self.layers[layer_index]
    }
}

/// Common inference operations implemented by each supported model architecture.
pub(crate) trait CausalLanguageModel: Send {
    fn layer_count(&self) -> usize;

    /// Returns next-token logits shaped `(batch_size, 1, vocabulary_size)`.
    fn forward(
        &mut self,
        input: &Tensor,
        start_position: usize,
        kv_cache: &mut KvCache,
    ) -> candle_core::Result<Tensor>;
}

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
