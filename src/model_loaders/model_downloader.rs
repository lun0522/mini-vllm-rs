use crate::model_loaders::ModelRole;
use crate::proto::model_config::ModelConfig;
use anyhow::Context;
use anyhow::Result;
use hf_hub::api::sync::Api;
use hf_hub::api::sync::ApiRepo;
use hf_hub::Repo;
use hf_hub::RepoType;
use log::info;
use std::path::PathBuf;
use std::time::Instant;

/// Downloads a quantized GGUF model and its tokenizer into the Hugging Face disk cache.
pub(crate) struct ModelDownloader {
    model: ModelConfig,
    role: ModelRole,
}

impl ModelDownloader {
    pub(crate) fn new(model: ModelConfig, role: ModelRole) -> Result<Self> {
        model.validate()?;
        Ok(Self { model, role })
    }

    /// Ensures the GGUF and tokenizer are present on disk and returns their paths.
    /// This is safe to call multiple times because files already in the Hugging Face
    /// cache are reused instead of downloaded again.
    pub(crate) fn download(&self) -> Result<ModelArtifacts> {
        let started = Instant::now();
        // hf-hub stores downloads in its standard local cache, so subsequent runs
        // reuse these files rather than downloading the model again.
        let api = Api::new().context("failed to create the Hugging Face Hub client")?;
        let repo = api.repo(Repo::with_revision(
            self.model.model_id.to_owned(),
            RepoType::Model,
            self.model.model_revision.to_owned(),
        ));
        let tokenizer_repo = api.repo(Repo::new(self.model.tokenizer_id.clone(), RepoType::Model));
        let model_artifacts = ModelArtifacts {
            gguf: self.download_gguf(&repo)?,
            tokenizer: self.download_tokenizer(&tokenizer_repo)?,
        };
        info!(
            "Prepared {} GGUF model artifacts (downloaded or reused from cache) in {:.2?}",
            self.role,
            started.elapsed()
        );
        Ok(model_artifacts)
    }

    fn download_gguf(&self, repo: &ApiRepo) -> Result<PathBuf> {
        repo.get(&self.model.model_filename).map_err(|error| {
            anyhow::anyhow!(
                "failed to download GGUF file '{}' from model '{}' at revision '{}'. Check that \
                 the repository, revision, and filename are correct. If the repository is \
                 private or gated, authenticate by setting HF_TOKEN. Hugging Face response: \
                 {error}",
                self.model.model_filename,
                self.model.model_id,
                self.model.model_revision,
            )
        })
    }

    fn download_tokenizer(&self, repo: &ApiRepo) -> Result<PathBuf> {
        repo.get("tokenizer.json").map_err(|error| {
            anyhow::anyhow!(
                "tokenizer model '{}' does not provide tokenizer.json, or the file could not be \
                 downloaded. Hugging Face response: {error}",
                self.model.tokenizer_id,
            )
        })
    }
}

/// Local artifacts required to initialize a GGUF model.
pub(crate) struct ModelArtifacts {
    pub gguf: PathBuf,
    pub tokenizer: PathBuf,
}
