use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::model_loaders::models::quantized_llama::LlamaBackend;
use crate::model_loaders::models::quantized_qwen2::Qwen2Backend;
use crate::model_loaders::CausalLanguageModel;
use crate::model_loaders::ModelArchitecture;
use crate::model_loaders::ModelRole;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;

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
    architecture: ModelArchitecture,
}

impl LoadedModel {
    /// Loads model artifacts from disk and initializes a supported model backend once.
    pub(crate) fn new(
        model_artifacts: &ModelArtifacts,
        device: Device,
        tokenizer: &Tokenizer,
        model_role: ModelRole,
    ) -> Result<Self> {
        let LoadedBackend {
            model,
            architecture,
            input_vocabulary_size,
            output_vocabulary_size,
        } = load_model_backend(&model_artifacts.gguf, &device)?;
        validate_vocabulary(
            input_vocabulary_size,
            output_vocabulary_size,
            tokenizer,
            model_role,
        )?;

        Ok(Self {
            model,
            device,
            architecture,
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn format_chat_prompt(&self, prompt: &str) -> String {
        self.architecture.format_chat_prompt(prompt)
    }

    pub(crate) fn end_of_sequence_tokens(&self) -> &'static [&'static str] {
        self.architecture.end_of_sequence_tokens()
    }

    /// Clears request-specific KV-cache state and returns reusable model resources.
    pub(crate) fn start_inference(&mut self) -> (&mut dyn CausalLanguageModel, &Device) {
        self.model.clear_kv_cache();
        (&mut *self.model, &self.device)
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

fn validate_vocabulary(
    input_vocabulary_size: usize,
    output_vocabulary_size: usize,
    tokenizer: &Tokenizer,
    model_role: ModelRole,
) -> Result<()> {
    // The model sizes describe the token-ID ranges accepted by its input
    // embeddings and produced by its output projection. The tokenizer's
    // required size is one past its highest assigned ID. A draft model must
    // cover that entire shared range because it consumes and proposes IDs
    // from the target model's tokenizer.
    let required_size = tokenizer
        .get_vocab(true)
        .values()
        .copied()
        .max()
        .map_or(0, |maximum_id| maximum_id as usize + 1);
    if input_vocabulary_size < required_size {
        anyhow::bail!(
            "{model_role} model input vocabulary has {input_vocabulary_size} entries but the shared \
             tokenizer requires token IDs through {}",
            required_size.saturating_sub(1),
        );
    }
    if output_vocabulary_size < required_size {
        anyhow::bail!(
            "{model_role} model output vocabulary has {output_vocabulary_size} entries but the shared \
             tokenizer requires token IDs through {}",
            required_size.saturating_sub(1),
        );
    }
    Ok(())
}
