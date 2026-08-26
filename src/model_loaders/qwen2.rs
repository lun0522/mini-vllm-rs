use crate::model_loaders::CausalLanguageModel;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::Config;
use candle_transformers::models::qwen2::ModelForCausalLM;
use std::path::Path;

pub(super) struct Qwen2Backend {
    model: ModelForCausalLM,
}

impl Qwen2Backend {
    pub(super) fn new(
        config: &[u8],
        weights_path: &Path,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let config: Config =
            serde_json::from_slice(config).context("failed to parse the Qwen2 model config")?;

        // SAFETY: The weights are immutable files in the Hugging Face cache, and this
        // process never modifies or truncates them while the memory-mapped model exists.
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)
                .context("failed to memory-map the model weights")?
        };
        let model = ModelForCausalLM::new(&config, var_builder)
            .context("failed to initialize the Qwen2 model")?;

        Ok(Self { model })
    }
}

impl CausalLanguageModel for Qwen2Backend {
    fn forward(&mut self, input: &Tensor, start_position: usize) -> candle_core::Result<Tensor> {
        self.model.forward(input, start_position)
    }

    fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
