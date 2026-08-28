mod llama;
pub(crate) mod loaded_model;
pub(crate) mod model_downloader;
mod qwen2;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use candle_core::Tensor;
use std::fs::File;
use std::path::Path;

/// Common inference operations implemented by each supported model architecture.
pub(crate) trait CausalLanguageModel: Send {
    /// Returns next-token logits shaped `(batch_size, 1, vocabulary_size)`.
    fn forward(&mut self, input: &Tensor, start_position: usize) -> candle_core::Result<Tensor>;

    /// Removes the KV-cache state left by the previous inference request.
    fn clear_kv_cache(&mut self);
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

pub(crate) fn load_model_backend(
    gguf_path: &Path,
    device: &Device,
) -> Result<(Box<dyn CausalLanguageModel>, ModelArchitecture)> {
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
    match architecture.as_str() {
        "llama" => Ok((
            Box::new(llama::LlamaBackend::new(content, &mut gguf_file, device)?),
            ModelArchitecture::Llama,
        )),
        "qwen2" => Ok((
            Box::new(qwen2::Qwen2Backend::new(content, &mut gguf_file, device)?),
            ModelArchitecture::Qwen2,
        )),
        unsupported => bail!("unsupported model architecture: {unsupported}"),
    }
}
