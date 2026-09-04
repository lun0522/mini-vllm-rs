use crate::model_loaders::models::quantized_llama::LlamaBackend;
use crate::model_loaders::models::quantized_qwen2::Qwen2Backend;
use crate::model_loaders::CausalLanguageModel;
use crate::model_loaders::ModelInfo;
use crate::proto::model_runner::ModelArchitecture;
use crate::proto::model_runner::ModelMetadata;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::fs::File;
use std::path::Path;

struct LoadedBackend {
    model: Box<dyn CausalLanguageModel>,
    architecture: ModelArchitecture,
    input_vocabulary_size: usize,
    output_vocabulary_size: usize,
}

/// Owns a fully initialized model and the resources reused by each inference.
pub(crate) struct LoadedModel {
    model: Box<dyn CausalLanguageModel>,
    device: Device,
    metadata: ModelMetadata,
}

impl LoadedModel {
    /// Loads model artifacts from disk and initializes a supported model backend once.
    pub(crate) fn new(gguf_path: &Path, device: Device) -> Result<Self> {
        let LoadedBackend {
            model,
            architecture,
            input_vocabulary_size,
            output_vocabulary_size,
        } = load_model_backend(gguf_path, &device)?;
        let metadata = ModelMetadata {
            architecture: architecture.into(),
            input_vocabulary_size: u64::try_from(input_vocabulary_size)
                .context("model input vocabulary size does not fit in u64")?,
            output_vocabulary_size: u64::try_from(output_vocabulary_size)
                .context("model output vocabulary size does not fit in u64")?,
        };
        Ok(Self {
            model,
            device,
            metadata,
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn info(&self) -> &ModelInfo {
        self.model.info()
    }

    pub(crate) fn metadata(&self) -> ModelMetadata {
        self.metadata
    }

    pub(crate) fn model(&mut self) -> &mut dyn CausalLanguageModel {
        &mut *self.model
    }
}

fn load_model_backend(gguf_path: &Path, device: &Device) -> Result<LoadedBackend> {
    let mut gguf_file = File::open(gguf_path).context("failed to open the GGUF model")?;
    let content =
        gguf_file::Content::read(&mut gguf_file).context("failed to read GGUF metadata")?;
    let input_vocabulary_size = content
        .tensor_infos
        .get("token_embd.weight")
        .context("GGUF model does not contain token_embd.weight")?
        .shape
        .dims()
        .first()
        .copied()
        .context("GGUF token embedding tensor has no dimensions")?;
    let output_vocabulary_size =
        content
            .tensor_infos
            .get("output.weight")
            .map_or(Ok(input_vocabulary_size), |tensor| {
                tensor
                    .shape
                    .dims()
                    .first()
                    .copied()
                    .context("GGUF output tensor has no dimensions")
            })?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .context("GGUF metadata does not contain general.architecture")?
        .to_string()
        .context("GGUF general.architecture is not a string")?
        .clone();
    match architecture.as_str() {
        "llama" => Ok(LoadedBackend {
            model: Box::new(LlamaBackend::new(content, &mut gguf_file, device)?),
            architecture: ModelArchitecture::Llama,
            input_vocabulary_size,
            output_vocabulary_size,
        }),
        "qwen2" => Ok(LoadedBackend {
            model: Box::new(Qwen2Backend::new(content, &mut gguf_file, device)?),
            architecture: ModelArchitecture::Qwen2,
            input_vocabulary_size,
            output_vocabulary_size,
        }),
        unsupported => bail!("unsupported model architecture: {unsupported}"),
    }
}
