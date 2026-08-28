use crate::model_loaders::load_model_backend;
use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::model_loaders::CausalLanguageModel;
use crate::model_loaders::ModelArchitecture;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use std::path::Path;
use tokenizers::Tokenizer;

/// Owns a fully initialized model and the resources reused by each inference.
pub(crate) struct LoadedModel {
    model: Box<dyn CausalLanguageModel>,
    tokenizer: Tokenizer,
    device: Device,
    architecture: ModelArchitecture,
}

impl LoadedModel {
    /// Loads model artifacts from disk and initializes a supported model backend once.
    pub(crate) fn new(model_artifacts: &ModelArtifacts, device: Device) -> Result<Self> {
        let tokenizer = load_tokenizer(&model_artifacts.tokenizer)?;
        let (model, architecture) = load_model_backend(&model_artifacts.gguf, &device)?;

        Ok(Self {
            model,
            tokenizer,
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

    /// Replaces this model's tokenizer with the tokenizer from another loaded model.
    /// This is only used for a speculative-decoding draft model, which must share
    /// the main model's tokenizer.
    pub(crate) fn substitute_tokenizer(&mut self, tokenizer_source: &LoadedModel) {
        self.tokenizer.clone_from(&tokenizer_source.tokenizer);
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
