use anyhow::Context;
use anyhow::Result;
use hf_hub::api::sync::Api;
use hf_hub::api::sync::ApiRepo;
use hf_hub::Repo;
use hf_hub::RepoType;
use log::info;
use std::path::PathBuf;
use std::time::Instant;

/// Downloads the files required to load a model into the Hugging Face disk cache.
pub(crate) struct ModelDownloader {
    model_id: String,
    revision: String,
}

impl ModelDownloader {
    pub(crate) fn new(model_id: String, revision: String) -> Self {
        Self { model_id, revision }
    }

    /// Ensures every required model file is present on disk and returns its path.
    /// This is safe to call multiple times because files already in the Hugging Face
    /// cache are reused instead of downloaded again.
    pub(crate) fn download(&self) -> Result<ModelFiles> {
        let started = Instant::now();
        // hf-hub stores downloads in its standard local cache, so subsequent runs
        // reuse these files rather than downloading the model again.
        let api = Api::new().context("failed to create the Hugging Face Hub client")?;
        let repo = api.repo(Repo::with_revision(
            self.model_id.to_owned(),
            RepoType::Model,
            self.revision.to_owned(),
        ));
        let config = repo.get("config.json").map_err(|error| {
            anyhow::anyhow!(
                "could not access Hugging Face model '{}' at revision '{}'. \
                 Check that --model-id and the revision are correct. If the repository is \
                 private or gated, authenticate by setting HF_TOKEN. Hugging Face response: \
                 {error}",
                self.model_id,
                self.revision,
            )
        })?;

        let model_files = ModelFiles {
            tokenizer: self.download_required_file(&repo, "tokenizer.json")?,
            config,
            weights: self.download_required_file(&repo, "model.safetensors")?,
        };
        info!(
            "Prepared model files (downloaded or reused from cache) in {:.2?}",
            started.elapsed()
        );
        Ok(model_files)
    }

    fn download_required_file(&self, repo: &ApiRepo, filename: &str) -> Result<PathBuf> {
        repo.get(filename).map_err(|error| {
            anyhow::anyhow!(
                "model '{}' does not provide the required file '{filename}', or the file \
                 could not be downloaded. Hugging Face response: {error}",
                self.model_id,
            )
        })
    }
}

/// Paths to a complete set of model artifacts already stored on disk.
pub(crate) struct ModelFiles {
    pub tokenizer: PathBuf,
    pub config: PathBuf,
    pub weights: PathBuf,
}
