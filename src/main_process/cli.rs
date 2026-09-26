use crate::model_runner::ActivationDType;
use crate::model_runner::InferenceDevice;
use crate::model_runner::KvCacheType;
use crate::model_runner::SchedulingPolicy;
use crate::proto::inference_config::draft_token_count_policy::Policy;
use crate::proto::inference_config::DraftModelConfig;
use crate::proto::inference_config::DraftTokenCountPolicy;
use crate::proto::inference_config::FixedDraftTokenCountPolicy;
use crate::proto::inference_config::ModelConfig;
use argh::FromArgs;
use log::warn;
use std::fmt;
use std::path::PathBuf;
use thousands::Separable;

const DEFAULT_TARGET_KV_CACHE_SIZE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_DRAFT_TOKEN_COUNT: u64 = 4;
const DEFAULT_MAX_BATCHED_TOKEN_COUNT: usize = 512;
const DEFAULT_MAX_ACTIVE_REQUEST_COUNT: usize = 4;
const DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT: usize = 4;

/// Runs text generation with a model from Hugging Face.
#[derive(FromArgs)]
pub(crate) struct MainProcessArgs {
    /// textproto configuration for the target GGUF model
    #[argh(option, default = "default_model_config()")]
    pub(crate) model: ModelConfig,
    /// textproto configuration for the speculative-decoding draft model and token-count policy
    #[argh(option)]
    pub(crate) draft_model: Option<DraftModelConfig>,
    /// device used for model inference
    #[argh(option, default = "InferenceDevice::Gpu")]
    pub(crate) inference_device: InferenceDevice,
    /// data type used for model activations and KV caches
    #[argh(option, default = "ActivationDType::F16")]
    pub(crate) activation_dtype: ActivationDType,
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
    /// directory where a Chrome trace is written
    #[argh(option)]
    pub(crate) trace_directory: Option<PathBuf>,
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
        if let Some(draft_model) = &self.draft_model {
            write!(formatter, "{draft_model}")?;
        } else {
            writeln!(formatter, "Draft model: disabled")?;
        }
        writeln!(formatter, "Inference device: {}", self.inference_device)?;
        writeln!(formatter, "Activation dtype: {}", self.activation_dtype)?;
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
        if let Some(trace_directory) = &self.trace_directory {
            writeln!(formatter, "Trace directory: {}", trace_directory.display())?;
        } else {
            writeln!(formatter, "Trace export: disabled")?;
        }
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

pub(crate) fn parse() -> MainProcessArgs {
    normalize(argh::from_env())
}

impl MainProcessArgs {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        self.model.validate()?;
        if let Some(draft_model) = &self.draft_model {
            draft_model.validate()?;
        }
        Ok(())
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
        warn!(
            "Invalid target KV-cache size {}; using default value \
             {}",
            args.target_kv_cache_size_bytes.separate_with_commas(),
            DEFAULT_TARGET_KV_CACHE_SIZE_BYTES.separate_with_commas()
        );
        args.target_kv_cache_size_bytes = DEFAULT_TARGET_KV_CACHE_SIZE_BYTES;
    }
    if args.max_batched_token_count == 0 {
        warn!(
            "Invalid maximum batched token count 0; using default value {DEFAULT_MAX_BATCHED_TOKEN_COUNT}"
        );
        args.max_batched_token_count = DEFAULT_MAX_BATCHED_TOKEN_COUNT;
    }
    if args.max_active_request_count == 0 {
        warn!(
            "Invalid maximum active request count 0; using default value {DEFAULT_MAX_ACTIVE_REQUEST_COUNT}"
        );
        args.max_active_request_count = DEFAULT_MAX_ACTIVE_REQUEST_COUNT;
    }
    if args.input_preprocessing_thread_count == 0 {
        warn!(
            "Invalid input preprocessing thread count 0; using default value \
             {DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT}"
        );
        args.input_preprocessing_thread_count = DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT;
    }
    if args.model.model_revision.is_empty() {
        args.model.model_revision = "main".to_owned();
    }
    if let Some(draft_model) = args.draft_model.as_mut() {
        if let Some(model) = draft_model.model.as_mut() {
            if model.model_revision.is_empty() {
                model.model_revision = "main".to_owned();
            }
        }
        if draft_model.token_count_policy.is_none() {
            warn!(
                "Draft token-count policy is missing; using a fixed draft token count of \
                 {DEFAULT_DRAFT_TOKEN_COUNT}"
            );
            draft_model.token_count_policy = Some(DraftTokenCountPolicy {
                policy: Some(Policy::Fixed(FixedDraftTokenCountPolicy {
                    draft_token_count: DEFAULT_DRAFT_TOKEN_COUNT,
                })),
            });
        }
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_draft_token_count(policy: &DraftTokenCountPolicy) -> u64 {
        let Some(Policy::Fixed(policy)) = policy.policy.as_ref() else {
            panic!("expected a fixed draft token-count policy")
        };
        policy.draft_token_count
    }

    fn initial_draft_token_count(policy: &DraftTokenCountPolicy) -> u64 {
        let Some(Policy::AcceptanceRate(policy)) = policy.policy.as_ref() else {
            panic!("expected an acceptance-rate draft token-count policy")
        };
        policy.initial_draft_token_count
    }

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
    fn selects_activation_dtype() {
        let default_args = MainProcessArgs::from_args(&["mini-vllm-rs"], &[])
            .expect("default arguments should parse");
        assert_eq!(default_args.activation_dtype, ActivationDType::F16);

        let f32_args =
            MainProcessArgs::from_args(&["mini-vllm-rs"], &["--activation-dtype", "f32"])
                .expect("F32 activation dtype should parse");
        assert_eq!(f32_args.activation_dtype, ActivationDType::F32);
    }

    #[test]
    fn selects_trace_directory() {
        let args =
            MainProcessArgs::from_args(&["mini-vllm-rs"], &["--trace-directory", "/tmp/traces"])
                .expect("trace directory should parse");

        assert_eq!(args.trace_directory, Some(PathBuf::from("/tmp/traces")));
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

    #[test]
    fn parses_a_draft_model_with_a_fixed_token_count_policy() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--draft-model",
                "model { model_id: 'draft' model_filename: 'draft.gguf' tokenizer_id: 'tokenizer' } token_count_policy { fixed { draft_token_count: 6 } }",
            ],
        )
        .expect("draft model configuration should parse");
        let args = normalize(args);
        let draft_model = args.draft_model.as_ref().unwrap();

        assert!(args.validate().is_ok());
        assert_eq!(draft_model.model.as_ref().unwrap().model_revision, "main");
        assert_eq!(
            fixed_draft_token_count(draft_model.token_count_policy.as_ref().unwrap()),
            6
        );
    }

    #[test]
    fn defaults_a_missing_draft_token_count_policy() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--draft-model",
                "model { model_id: 'draft' model_filename: 'draft.gguf' tokenizer_id: 'tokenizer' }",
            ],
        )
        .expect("draft model configuration should parse");
        let args = normalize(args);
        let policy = args
            .draft_model
            .as_ref()
            .unwrap()
            .token_count_policy
            .as_ref()
            .unwrap();

        assert!(args.validate().is_ok());
        assert_eq!(fixed_draft_token_count(policy), DEFAULT_DRAFT_TOKEN_COUNT);
    }

    #[test]
    fn parses_a_draft_model_with_an_acceptance_rate_policy() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--draft-model",
                "model { model_id: 'draft' model_filename: 'draft.gguf' tokenizer_id: 'tokenizer' } token_count_policy { acceptance_rate { initial_draft_token_count: 4 decrease_threshold: 0.4 increase_threshold: 0.8 minimum_draft_token_count: 1 maximum_draft_token_count: 8 } }",
            ],
        )
        .expect("draft model configuration should parse");
        let args = normalize(args);
        let draft_model = args.draft_model.as_ref().unwrap();

        assert!(args.validate().is_ok());
        assert_eq!(
            initial_draft_token_count(draft_model.token_count_policy.as_ref().unwrap()),
            4
        );
    }
}
