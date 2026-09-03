use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::model_loaders::ModelRole;
use crate::model_runner::KvCacheType;
use crate::proto::ModelPaths;
use crate::utils::textproto::parse_textproto;
use anyhow::Result;
use argh::FromArgs;
use std::path::PathBuf;
use std::str::FromStr;

/// Runs model inference using files already available on disk.
#[derive(FromArgs)]
pub(crate) struct ModelRunnerProcessArgs {
    /// textproto paths for the target GGUF model and tokenizer
    #[argh(option)]
    pub(super) model: ModelPaths,
    /// textproto paths for the draft GGUF model and tokenizer
    #[argh(option)]
    pub(super) draft_model: Option<ModelPaths>,
    /// number of tokens proposed by the draft model per speculative decoding step
    #[argh(option)]
    pub(super) draft_token_count: usize,
    /// KV cache implementation used for model inference
    #[argh(option)]
    pub(super) kv_cache_type: KvCacheType,
    /// number of tokens per KV-cache page; only affects paged KV caches
    #[argh(option)]
    pub(super) kv_cache_page_token_count: usize,
    /// total KV-cache size in bytes for the target model
    #[argh(option)]
    pub(super) target_kv_cache_size_bytes: usize,
    /// unix domain socket path used by the model runner worker
    #[argh(option)]
    pub(super) socket_path: PathBuf,
}

impl FromStr for ModelPaths {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        parse_textproto(value, "model_runner.ModelPaths")
    }
}

pub(super) fn create_model_artifacts(
    paths: ModelPaths,
    model_role: ModelRole,
) -> Result<ModelArtifacts> {
    if paths.tokenizer_path.is_empty() || paths.gguf_path.is_empty() {
        anyhow::bail!("{model_role} model tokenizer_path and gguf_path must not be empty");
    }
    Ok(ModelArtifacts {
        tokenizer: PathBuf::from(paths.tokenizer_path),
        gguf: PathBuf::from(paths.gguf_path),
    })
}
