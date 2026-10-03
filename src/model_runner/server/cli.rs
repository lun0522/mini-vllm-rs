use crate::model_runner::ActivationDType;
use crate::model_runner::InferenceDevice;
use crate::proto::inference_config::DraftModelRunnerConfig;
use crate::proto::inference_config::KvCacheConfig as KvCacheConfigProto;
use crate::proto::inference_config::SchedulerConfig as SchedulerConfigProto;
use argh::FromArgs;
use std::path::PathBuf;

/// Runs model inference using files already available on disk.
#[derive(FromArgs)]
pub(crate) struct ModelRunnerProcessArgs {
    /// target GGUF model path
    #[argh(option)]
    pub(super) model_path: PathBuf,
    /// textproto draft model runner configuration
    #[argh(option)]
    pub(super) draft_model_runner_config: Option<DraftModelRunnerConfig>,
    /// device used for model inference
    #[argh(option)]
    pub(super) inference_device: InferenceDevice,
    /// data type used for model activations and KV caches
    #[argh(option)]
    pub(super) activation_dtype: ActivationDType,
    /// textproto KV-cache configuration
    #[argh(option)]
    pub(super) kv_cache_config: KvCacheConfigProto,
    /// textproto scheduler configuration
    #[argh(option)]
    pub(super) scheduler_config: SchedulerConfigProto,
    /// directory where a Chrome trace is written
    #[argh(option)]
    pub(crate) trace_directory: Option<PathBuf>,
    /// unix domain socket path used by the model runner worker
    #[argh(option)]
    pub(super) socket_path: PathBuf,
}
