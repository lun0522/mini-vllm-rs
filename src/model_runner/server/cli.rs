use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::proto::ModelPaths;
use crate::utils::textproto::parse_textproto;
use anyhow::Result;
use argh::FromArgs;
use std::path::PathBuf;
use std::str::FromStr;

/// Runs model inference using files already available on disk.
#[derive(FromArgs)]
pub(crate) struct ModelRunnerProcessArgs {
    /// textproto paths for the main GGUF model and tokenizer
    #[argh(option)]
    pub(super) model: ModelPaths,
    /// textproto paths for the draft GGUF model and tokenizer
    #[argh(option)]
    pub(super) draft_model: Option<ModelPaths>,
    /// number of tokens proposed by the draft model per speculative decoding step
    #[argh(option)]
    pub(super) draft_token_count: usize,
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
    model_name: &str,
) -> Result<ModelArtifacts> {
    if paths.tokenizer_path.is_empty() || paths.gguf_path.is_empty() {
        anyhow::bail!("{model_name} model tokenizer_path and gguf_path must not be empty");
    }
    Ok(ModelArtifacts {
        tokenizer: PathBuf::from(paths.tokenizer_path),
        gguf: PathBuf::from(paths.gguf_path),
    })
}
