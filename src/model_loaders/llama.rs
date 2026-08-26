use crate::model_loaders::CausalLanguageModel;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::llama::Cache;
use candle_transformers::models::llama::Llama;
use candle_transformers::models::llama::LlamaConfig;
use std::path::Path;

pub(super) struct LlamaBackend {
    model: Llama,
    cache: Cache,
    empty_cache: Cache,
}

impl LlamaBackend {
    pub(super) fn new(
        config: &[u8],
        weights_path: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let config: LlamaConfig =
            serde_json::from_slice(config).context("failed to parse the Llama model config")?;
        let config = config.into_config(false);

        // SAFETY: The weights are immutable files in the Hugging Face cache, and this
        // process never modifies or truncates them while the memory-mapped model exists.
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .context("failed to memory-map the model weights")?
        };
        let model =
            Llama::load(var_builder, &config).context("failed to initialize the Llama model")?;
        let empty_cache = Cache::new(true, dtype, &config, device)
            .context("failed to initialize the Llama KV cache")?;
        let cache = empty_cache.clone();

        Ok(Self {
            model,
            cache,
            empty_cache,
        })
    }
}

impl CausalLanguageModel for LlamaBackend {
    fn forward(&mut self, input: &Tensor, start_position: usize) -> candle_core::Result<Tensor> {
        // Candle's Llama implementation omits the single-token sequence dimension.
        // Restore it so every backend follows the common trait's logits contract.
        self.model
            .forward(input, start_position, &mut self.cache)?
            .unsqueeze(1)
    }

    fn clear_kv_cache(&mut self) {
        self.cache = self.empty_cache.clone();
    }
}
