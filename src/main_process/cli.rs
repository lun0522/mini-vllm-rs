use crate::model_runner::InferenceDevice;
use crate::model_runner::KvCacheType;
use crate::model_runner::SchedulerConfig;
use crate::model_runner::SchedulingPolicy;
use crate::proto::model_config::ModelConfig;
use crate::utils::textproto::parse_textproto;
use argh::FromArgs;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;
use thousands::Separable;

const DEFAULT_DRAFT_TOKEN_COUNT: usize = 4;
const DEFAULT_TARGET_KV_CACHE_SIZE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_MAX_BATCHED_TOKEN_COUNT: usize = 512;
const DEFAULT_MAX_ACTIVE_REQUEST_COUNT: usize = 4;
const DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT: usize = 4;

/// Runs text generation with a model from Hugging Face.
#[derive(FromArgs)]
pub(crate) struct MainProcessArgs {
    /// textproto configuration for the target GGUF model
    #[argh(option, default = "default_model_config()")]
    pub(crate) model: ModelConfig,
    /// textproto configuration for the speculative-decoding draft model
    #[argh(option)]
    pub(crate) draft_model: Option<ModelConfig>,
    /// number of tokens proposed by the draft model per speculative decoding step
    #[argh(option, default = "DEFAULT_DRAFT_TOKEN_COUNT")]
    pub(crate) draft_token_count: usize,
    /// device used for model inference
    #[argh(option, default = "InferenceDevice::Gpu")]
    pub(crate) inference_device: InferenceDevice,
    /// KV cache implementation used for model inference
    #[argh(option, default = "KvCacheType::Contiguous")]
    pub(crate) kv_cache_type: KvCacheType,
    /// total KV-cache size in bytes for the target model
    #[argh(option, default = "DEFAULT_TARGET_KV_CACHE_SIZE_BYTES")]
    pub(crate) target_kv_cache_size_bytes: usize,
    /// maximum number of tokens processed in one model batch
    #[argh(option, default = "DEFAULT_MAX_BATCHED_TOKEN_COUNT")]
    pub(crate) max_batched_token_count: usize,
    /// maximum number of requests that may hold active inference state
    #[argh(option, default = "DEFAULT_MAX_ACTIVE_REQUEST_COUNT")]
    pub(crate) max_active_request_count: usize,
    /// policy used to choose requests for the next model batch
    #[argh(option, default = "SchedulingPolicy::FirstComeFirstServed")]
    pub(crate) scheduling_policy: SchedulingPolicy,
    /// number of request-handler threads used for concurrent input preprocessing
    #[argh(option, default = "DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT")]
    pub(crate) input_preprocessing_thread_count: usize,
    /// unix domain socket exposed to local inference clients
    #[argh(option, default = "default_request_socket()")]
    pub(crate) request_socket: PathBuf,
    /// unix domain socket exposed to lifecycle-control clients
    #[argh(option, default = "default_control_socket()")]
    pub(crate) control_socket: PathBuf,
}

impl fmt::Display for MainProcessArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Model: {}", self.model.model_id)?;
        writeln!(formatter, "GGUF file: {}", self.model.model_filename)?;
        writeln!(formatter, "Tokenizer: {}", self.model.tokenizer_id)?;
        writeln!(formatter, "Revision: {}", self.model.model_revision)?;
        if let Some(model) = &self.draft_model {
            writeln!(formatter, "Draft model: {}", model.model_id)?;
            writeln!(formatter, "Draft GGUF file: {}", model.model_filename)?;
            writeln!(formatter, "Draft tokenizer: {}", model.tokenizer_id)?;
            writeln!(formatter, "Draft revision: {}", model.model_revision)?;
            writeln!(formatter, "Draft token count: {}", self.draft_token_count)?;
        } else {
            writeln!(formatter, "Draft model: disabled")?;
        }
        writeln!(formatter, "Inference device: {}", self.inference_device)?;
        writeln!(formatter, "KV cache type: {}", self.kv_cache_type)?;
        writeln!(
            formatter,
            "Target KV cache size: {} bytes",
            self.target_kv_cache_size_bytes.separate_with_commas()
        )?;
        writeln!(
            formatter,
            "Maximum batched token count: {}",
            self.max_batched_token_count.separate_with_commas()
        )?;
        writeln!(
            formatter,
            "Maximum active request count: {}",
            self.max_active_request_count.separate_with_commas()
        )?;
        writeln!(formatter, "Scheduling policy: {}", self.scheduling_policy)?;
        writeln!(
            formatter,
            "Input preprocessing thread count: {}",
            self.input_preprocessing_thread_count
        )?;
        writeln!(
            formatter,
            "Request socket: {}",
            self.request_socket.display()
        )?;
        write!(
            formatter,
            "Control socket: {}",
            self.control_socket.display()
        )
    }
}

impl FromStr for ModelConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "model_config.ModelConfig")
    }
}

pub(crate) fn parse() -> MainProcessArgs {
    normalize(argh::from_env())
}

impl MainProcessArgs {
    pub(crate) fn scheduler_config(&self) -> SchedulerConfig {
        SchedulerConfig {
            max_batched_token_count: self.max_batched_token_count,
            max_active_request_count: self.max_active_request_count,
            scheduling_policy: self.scheduling_policy,
        }
    }
}

fn default_model_config() -> ModelConfig {
    ModelConfig {
        model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_owned(),
        model_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_owned(),
        tokenizer_id: "Qwen/Qwen2.5-7B-Instruct".to_owned(),
        model_revision: "main".to_owned(),
    }
}

fn default_request_socket() -> PathBuf {
    PathBuf::from("/tmp/mini-vllm-request-handler.sock")
}

fn default_control_socket() -> PathBuf {
    PathBuf::from("/tmp/mini-vllm-main-process.sock")
}

fn normalize(mut args: MainProcessArgs) -> MainProcessArgs {
    if args.target_kv_cache_size_bytes == 0 {
        log::warn!(
            "Invalid target KV-cache size {}; using default value \
             {}",
            args.target_kv_cache_size_bytes.separate_with_commas(),
            DEFAULT_TARGET_KV_CACHE_SIZE_BYTES.separate_with_commas()
        );
        args.target_kv_cache_size_bytes = DEFAULT_TARGET_KV_CACHE_SIZE_BYTES;
    }
    if args.max_batched_token_count == 0 {
        log::warn!(
            "Invalid maximum batched token count 0; using default value {DEFAULT_MAX_BATCHED_TOKEN_COUNT}"
        );
        args.max_batched_token_count = DEFAULT_MAX_BATCHED_TOKEN_COUNT;
    }
    if args.max_active_request_count == 0 {
        log::warn!(
            "Invalid maximum active request count 0; using default value {DEFAULT_MAX_ACTIVE_REQUEST_COUNT}"
        );
        args.max_active_request_count = DEFAULT_MAX_ACTIVE_REQUEST_COUNT;
    }
    if args.input_preprocessing_thread_count == 0 {
        log::warn!(
            "Invalid input preprocessing thread count 0; using default value \
             {DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT}"
        );
        args.input_preprocessing_thread_count = DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT;
    }
    if args.model.model_revision.is_empty() {
        args.model.model_revision = "main".to_owned();
    }
    if let Some(draft_model) = args.draft_model.as_mut() {
        if draft_model.model_revision.is_empty() {
            draft_model.model_revision = "main".to_owned();
        }
        if args.draft_token_count == 0 {
            args.draft_token_count = DEFAULT_DRAFT_TOKEN_COUNT;
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_gpu_inference() {
        let args = MainProcessArgs::from_args(&["mini-vllm-rs"], &[])
            .expect("default arguments should parse");

        assert_eq!(args.inference_device, InferenceDevice::Gpu);
    }

    #[test]
    fn selects_cpu_inference() {
        let args = MainProcessArgs::from_args(&["mini-vllm-rs"], &["--inference-device", "cpu"])
            .expect("CPU arguments should parse");

        assert_eq!(args.inference_device, InferenceDevice::Cpu);
    }

    #[test]
    fn parses_scheduler_configuration() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--max-batched-token-count",
                "1024",
                "--max-active-request-count",
                "8",
                "--scheduling-policy",
                "shortest-prefill-first",
            ],
        )
        .expect("scheduler arguments should parse");

        assert_eq!(args.max_batched_token_count, 1024);
        assert_eq!(args.max_active_request_count, 8);
        assert_eq!(
            args.scheduling_policy,
            SchedulingPolicy::ShortestPrefillFirst
        );
    }

    #[test]
    fn defaults_to_four_input_preprocessing_threads() {
        let args = MainProcessArgs::from_args(&["mini-vllm-rs"], &[])
            .expect("default arguments should parse");

        assert_eq!(args.input_preprocessing_thread_count, 4);
    }
}
