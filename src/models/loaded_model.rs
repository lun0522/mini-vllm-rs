use crate::model_runner::ActivationDType;
use crate::models::backends::quantized_llama::LlamaBackend;
use crate::models::backends::quantized_qwen2::Qwen2Backend;
use crate::models::CausalLanguageModel;
use crate::models::ModelInfo;
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
    tokenizer_vocabulary_size: usize,
}

/// Owns a fully initialized model and the resources reused by each inference.
pub(crate) struct LoadedModel {
    model: Box<dyn CausalLanguageModel>,
    device: Device,
    metadata: ModelMetadata,
}

impl LoadedModel {
    /// Loads model artifacts from disk and initializes a supported model backend once.
    pub(crate) fn new(
        gguf_path: &Path,
        device: Device,
        activation_dtype: ActivationDType,
    ) -> Result<Self> {
        let LoadedBackend {
            model,
            architecture,
            input_vocabulary_size,
            output_vocabulary_size,
            tokenizer_vocabulary_size,
        } = load_model_backend(gguf_path, &device, activation_dtype)?;
        let metadata = ModelMetadata {
            context_length: u64::try_from(model.info().context_length)
                .context("model context length does not fit in u64")?,
            architecture: architecture.into(),
            input_vocabulary_size: u64::try_from(input_vocabulary_size)
                .context("model input vocabulary size does not fit in u64")?,
            output_vocabulary_size: u64::try_from(output_vocabulary_size)
                .context("model output vocabulary size does not fit in u64")?,
            tokenizer_vocabulary_size: u64::try_from(tokenizer_vocabulary_size)
                .context("model tokenizer vocabulary size does not fit in u64")?,
        };
        anyhow::ensure!(
            tokenizer_vocabulary_size <= input_vocabulary_size,
            "model input vocabulary has {input_vocabulary_size} entries but its tokenizer defines {tokenizer_vocabulary_size} token IDs"
        );
        anyhow::ensure!(
            tokenizer_vocabulary_size <= output_vocabulary_size,
            "model output vocabulary has {output_vocabulary_size} entries but its tokenizer defines {tokenizer_vocabulary_size} token IDs"
        );
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

    #[cfg(test)]
    pub(crate) fn for_test(model: Box<dyn CausalLanguageModel>) -> Self {
        Self {
            model,
            device: Device::Cpu,
            metadata: ModelMetadata::default(),
        }
    }
}

fn load_model_backend(
    gguf_path: &Path,
    device: &Device,
    activation_dtype: ActivationDType,
) -> Result<LoadedBackend> {
    let mut gguf_file = File::open(gguf_path).context("failed to open the GGUF model")?;
    let content =
        gguf_file::Content::read(&mut gguf_file).context("failed to read GGUF metadata")?;
    let architecture = content
        .metadata
        .get("general.architecture")
        .context("GGUF metadata does not contain general.architecture")?
        .to_string()
        .context("GGUF general.architecture is not a string")?
        .clone();
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
    let tokenizer_vocabulary_size = get_tokenizer_vocabulary_size(&content)?;
    match architecture.as_str() {
        "llama" => Ok(LoadedBackend {
            model: Box::new(LlamaBackend::new(
                content,
                &mut gguf_file,
                device,
                activation_dtype,
            )?),
            architecture: ModelArchitecture::Llama,
            input_vocabulary_size,
            output_vocabulary_size,
            tokenizer_vocabulary_size,
        }),
        "qwen2" => Ok(LoadedBackend {
            model: Box::new(Qwen2Backend::new(
                content,
                &mut gguf_file,
                device,
                activation_dtype,
            )?),
            architecture: ModelArchitecture::Qwen2,
            input_vocabulary_size,
            output_vocabulary_size,
            tokenizer_vocabulary_size,
        }),
        unsupported => bail!("unsupported model architecture: {unsupported}"),
    }
}

/// Returns the usable token-ID range, excluding trailing GGUF tokenizer entries marked `UNUSED`.
/// Older GGUF files without token-type metadata use the full tokenizer token array.
fn get_tokenizer_vocabulary_size(content: &gguf_file::Content) -> Result<usize> {
    let tokenizer_tokens = match content
        .metadata
        .get("tokenizer.ggml.tokens")
        .context("GGUF metadata does not contain tokenizer.ggml.tokens")?
    {
        gguf_file::Value::Array(tokens) => tokens,
        value => bail!(
            "GGUF tokenizer.ggml.tokens metadata is not an array: {:?}",
            value.value_type()
        ),
    };
    let tokenizer_vocabulary_size = content.metadata.get("tokenizer.ggml.token_type").map_or(
        Ok(tokenizer_tokens.len()),
        |value| {
            let gguf_file::Value::Array(token_types) = value else {
                bail!(
                    "GGUF tokenizer.ggml.token_type metadata is not an array: {:?}",
                    value.value_type()
                )
            };
            anyhow::ensure!(
                token_types.len() == tokenizer_tokens.len(),
                "GGUF tokenizer contains {} tokens but {} token types",
                tokenizer_tokens.len(),
                token_types.len()
            );
            anyhow::ensure!(
                token_types
                    .iter()
                    .all(|token_type| matches!(token_type, gguf_file::Value::I32(_))),
                "GGUF tokenizer.ggml.token_type metadata contains a non-i32 value"
            );
            const UNUSED_TOKEN_TYPE: i32 = 5;
            token_types
                .iter()
                .rposition(|token_type| {
                    !matches!(token_type, gguf_file::Value::I32(UNUSED_TOKEN_TYPE))
                })
                .map(|index| index + 1)
                .context("GGUF tokenizer marks every token as unused")
        },
    )?;
    anyhow::ensure!(
        tokenizer_vocabulary_size > 0,
        "GGUF tokenizer.ggml.tokens metadata is empty"
    );
    Ok(tokenizer_vocabulary_size)
}
