use crate::model_loaders::load_model_backend;
use crate::model_loaders::CausalLanguageModel;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use hf_hub::api::sync::Api;
use hf_hub::api::sync::ApiRepo;
use hf_hub::Repo;
use hf_hub::RepoType;
use serde_json::Value;
use std::path::Path;
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Owns a fully initialized model and the resources reused by each inference.
pub(crate) struct LoadedModel {
    model: Box<dyn CausalLanguageModel>,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl LoadedModel {
    /// Downloads cached model files and initializes a supported model backend once.
    pub(crate) fn new(model_id: &str, revision: &str, device: Device) -> Result<Self> {
        let model_files = download_model_files(model_id, revision)?;
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

struct ModelFiles {
    tokenizer: PathBuf,
    config: PathBuf,
    weights: PathBuf,
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

fn download_model_files(model_id: &str, revision: &str) -> Result<ModelFiles> {
    // hf-hub stores downloads in its standard local cache, so subsequent runs
    // reuse these files rather than downloading the model again.
    let api = Api::new().context("failed to create the Hugging Face Hub client")?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_owned(),
        RepoType::Model,
        revision.to_owned(),
    ));
    let config = repo.get("config.json").map_err(|error| {
        anyhow::anyhow!(
            "could not access Hugging Face model '{model_id}' at revision '{revision}'. \
             Check that --model-id and the revision are correct. If the repository is \
             private or gated, authenticate by setting HF_TOKEN. Hugging Face response: \
             {error}"
        )
    })?;

    Ok(ModelFiles {
        tokenizer: download_required_model_file(&repo, model_id, "tokenizer.json")?,
        config,
        weights: download_required_model_file(&repo, model_id, "model.safetensors")?,
    })
}

fn download_required_model_file(repo: &ApiRepo, model_id: &str, filename: &str) -> Result<PathBuf> {
    repo.get(filename).map_err(|error| {
        anyhow::anyhow!(
            "model '{model_id}' does not provide the required file '{filename}', or the file \
             could not be downloaded. Hugging Face response: {error}"
        )
    })
}
