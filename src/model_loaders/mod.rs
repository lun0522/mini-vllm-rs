mod llama;
pub(crate) mod loaded_model;
mod qwen2;

use anyhow::bail;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use std::path::Path;

/// Common inference operations implemented by each supported model architecture.
pub(crate) trait CausalLanguageModel {
    /// Returns next-token logits shaped `(batch_size, 1, vocabulary_size)`.
    fn forward(&mut self, input: &Tensor, start_position: usize) -> candle_core::Result<Tensor>;

    /// Removes the KV-cache state left by the previous inference request.
    fn clear_kv_cache(&mut self);
}

pub(crate) fn load_model_backend(
    model_type: &str,
    config: &[u8],
    weights_path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<Box<dyn CausalLanguageModel>> {
    match model_type {
        "llama" => Ok(Box::new(llama::LlamaBackend::new(
            config,
            weights_path,
            dtype,
            device,
        )?)),
        "qwen2" => Ok(Box::new(qwen2::Qwen2Backend::new(
            config,
            weights_path,
            dtype,
            device,
        )?)),
        unsupported => bail!("unsupported model architecture: {unsupported}"),
    }
}
