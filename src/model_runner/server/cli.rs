use crate::model_runner::KvCacheType;
use argh::FromArgs;
use std::path::PathBuf;

/// Runs model inference using files already available on disk.
#[derive(FromArgs)]
pub(crate) struct ModelRunnerProcessArgs {
    /// target GGUF model path
    #[argh(option)]
    pub(super) model_path: PathBuf,
    /// draft GGUF model path
    #[argh(option)]
    pub(super) draft_model_path: Option<PathBuf>,
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
