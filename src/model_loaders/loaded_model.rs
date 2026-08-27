use crate::model_loaders::load_model_backend;
use crate::model_loaders::model_downloader::ModelFiles;
use crate::model_loaders::CausalLanguageModel;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

/// Owns a fully initialized model and the resources reused by each inference.
pub(crate) struct LoadedModel {
    model: Box<dyn CausalLanguageModel>,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl LoadedModel {
    /// Loads model artifacts from disk and initializes a supported model backend once.
    pub(crate) fn new(model_files: &ModelFiles, device: Device) -> Result<Self> {
        let tokenizer = load_tokenizer(&model_files.tokenizer)?;
        let model_config = load_model_config(&model_files.config)?;
        let model_type = get_model_type(&model_config)?;
        let dtype = get_inference_dtype(&device);
        let model = load_model_backend(
            &model_type,
            &model_config,
            &model_files.weights,
            dtype,
            &device,
        )?;

        Ok(Self {
            model,
            tokenizer,
            device,
            dtype,
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    /// Clears request-specific KV-cache state and returns reusable inference resources.
    pub(crate) fn start_inference(
        &mut self,
    ) -> (&mut dyn CausalLanguageModel, &Tokenizer, &Device) {
        self.model.clear_kv_cache();
        (&mut *self.model, &self.tokenizer, &self.device)
    }
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(anyhow::Error::msg)
        .context("failed to load the tokenizer")
}

fn load_model_config(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).context("failed to read the model config")
}

fn get_model_type(config: &[u8]) -> Result<String> {
    let config_metadata: Value =
        serde_json::from_slice(config).context("failed to parse the model config metadata")?;
    config_metadata["model_type"]
        .as_str()
        .map(str::to_owned)
        .context("model config does not contain a string model_type")
}

fn get_inference_dtype(device: &Device) -> DType {
    // The checkpoint stores BF16 tensors. Candle converts them to F32 on CPU,
    // while Metal can run them as BF16 to save memory and improve throughput.
    if device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}
