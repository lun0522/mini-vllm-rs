use crate::model_loaders::CausalLanguageModel;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::models::quantized_llama::ModelWeights;
use std::fs::File;

pub(super) struct LlamaBackend {
    model: ModelWeights,
}

impl LlamaBackend {
    pub(super) fn new(
        content: gguf_file::Content,
        gguf_file: &mut File,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            model: ModelWeights::from_gguf(content, gguf_file, device)?,
        })
    }
}

impl CausalLanguageModel for LlamaBackend {
    fn forward(&mut self, input: &Tensor, start_position: usize) -> candle_core::Result<Tensor> {
        self.model.forward(input, start_position)?.unsqueeze(1)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
