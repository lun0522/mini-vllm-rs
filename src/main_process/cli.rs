use crate::model_runner::ActivationDType;
use crate::model_runner::InferenceDevice;
use crate::model_runner::DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT;
use crate::proto::inference_config::draft_token_count_policy::Policy;
use crate::proto::inference_config::DraftModelConfig;
use crate::proto::inference_config::DraftTokenCountPolicy;
use crate::proto::inference_config::FixedDraftTokenCountPolicy;
use crate::proto::inference_config::KvCacheConfig as KvCacheConfigProto;
use crate::proto::inference_config::KvCacheType as KvCacheTypeProto;
use crate::proto::inference_config::ModelConfig;
use crate::proto::inference_config::SchedulerConfig as SchedulerConfigProto;
use crate::proto::inference_config::DEFAULT_KV_CACHE_TYPE;
use crate::proto::inference_config::DEFAULT_SCHEDULING_POLICY;
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
    /// textproto KV-cache configuration
    #[argh(option, default = "default_kv_cache_config()")]
    pub(crate) kv_cache_config: KvCacheConfigProto,
    /// textproto scheduler configuration
    #[argh(option, default = "default_scheduler_config()")]
    pub(crate) scheduler_config: SchedulerConfigProto,
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
        write!(formatter, "{}", self.kv_cache_config)?;
        write!(formatter, "{}", self.scheduler_config)?;
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
        self.kv_cache_config.validate()?;
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

fn default_scheduler_config() -> SchedulerConfigProto {
    SchedulerConfigProto {
        max_batched_token_count: DEFAULT_MAX_BATCHED_TOKEN_COUNT as u32,
        max_active_request_count: DEFAULT_MAX_ACTIVE_REQUEST_COUNT as u32,
        scheduling_policy: DEFAULT_SCHEDULING_POLICY.into(),
    }
}

fn default_kv_cache_config() -> KvCacheConfigProto {
    KvCacheConfigProto {
        kv_cache_type: DEFAULT_KV_CACHE_TYPE.into(),
        per_page_token_count: 0,
        target_kv_cache_size_bytes: DEFAULT_TARGET_KV_CACHE_SIZE_BYTES as u64,
        draft_kv_cache_size_bytes: 0,
    }
}

fn default_request_socket() -> PathBuf {
    PathBuf::from("/tmp/mini-vllm-request-handler.sock")
}

fn default_control_socket() -> PathBuf {
    PathBuf::from("/tmp/mini-vllm-main-process.sock")
}

fn normalize(mut args: MainProcessArgs) -> MainProcessArgs {
    normalize_kv_cache_config(&mut args.kv_cache_config);
    normalize_scheduler_config(&mut args.scheduler_config);
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

fn normalize_kv_cache_config(config: &mut KvCacheConfigProto) {
    if config.target_kv_cache_size_bytes == 0 {
        warn!(
            "Invalid target KV-cache size {}; using default value \
             {}",
            config.target_kv_cache_size_bytes.separate_with_commas(),
            DEFAULT_TARGET_KV_CACHE_SIZE_BYTES.separate_with_commas()
        );
        config.target_kv_cache_size_bytes = DEFAULT_TARGET_KV_CACHE_SIZE_BYTES as u64;
    }
    let kv_cache_type =
        KvCacheTypeProto::try_from(config.kv_cache_type).unwrap_or(DEFAULT_KV_CACHE_TYPE);
    if kv_cache_type != KvCacheTypeProto::Contiguous && config.per_page_token_count == 0 {
        warn!(
            "Invalid KV-cache page token count 0; using default value \
             {DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT}"
        );
        config.per_page_token_count = DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT as u32;
    }
}

fn normalize_scheduler_config(config: &mut SchedulerConfigProto) {
    if config.max_batched_token_count == 0 {
        warn!(
            "Invalid maximum batched token count 0; using default value {DEFAULT_MAX_BATCHED_TOKEN_COUNT}"
        );
        config.max_batched_token_count = DEFAULT_MAX_BATCHED_TOKEN_COUNT as u32;
    }
    if config.max_active_request_count == 0 {
        warn!(
            "Invalid maximum active request count 0; using default value {DEFAULT_MAX_ACTIVE_REQUEST_COUNT}"
        );
        config.max_active_request_count = DEFAULT_MAX_ACTIVE_REQUEST_COUNT as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::inference_config::SchedulingPolicy as SchedulingPolicyProto;

    fn fixed_draft_token_count(policy: &DraftTokenCountPolicy) -> u64 {
        let Some(Policy::Fixed(policy)) = policy.policy.as_ref() else {
            panic!("expected a fixed draft token-count policy")
        };
        policy.draft_token_count
    }

    fn initial_draft_token_count(policy: &DraftTokenCountPolicy) -> u64 {
        match policy.policy.as_ref() {
            Some(Policy::AcceptanceRate(policy)) => policy.initial_draft_token_count,
            Some(Policy::AcceptedLength(policy)) => policy.initial_draft_token_count,
            _ => panic!("expected an adaptive draft token-count policy"),
        }
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
                "--scheduler-config",
                "max_batched_token_count: 1024 max_active_request_count: 8 scheduling_policy: SCHEDULING_POLICY_SHORTEST_PREFILL_FIRST",
            ],
        )
        .expect("scheduler arguments should parse");

        assert_eq!(args.scheduler_config.max_batched_token_count, 1024);
        assert_eq!(args.scheduler_config.max_active_request_count, 8);
        assert_eq!(
            args.scheduler_config.scheduling_policy,
            SchedulingPolicyProto::ShortestPrefillFirst as i32
        );
    }

    #[test]
    fn parses_kv_cache_configuration() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--kv-cache-config",
                "kv_cache_type: KV_CACHE_TYPE_PAGED_PREFIX per_page_token_count: 32 target_kv_cache_size_bytes: 1073741824 draft_kv_cache_size_bytes: 268435456",
            ],
        )
        .expect("KV-cache arguments should parse");

        assert_eq!(
            args.kv_cache_config.kv_cache_type,
            KvCacheTypeProto::PagedPrefix as i32
        );
        assert_eq!(args.kv_cache_config.per_page_token_count, 32);
        assert_eq!(
            args.kv_cache_config.target_kv_cache_size_bytes,
            1_073_741_824
        );
        assert_eq!(args.kv_cache_config.draft_kv_cache_size_bytes, 268_435_456);
    }

    #[test]
    fn defaults_omitted_kv_cache_configuration_fields() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &["--kv-cache-config", "kv_cache_type: KV_CACHE_TYPE_PAGED"],
        )
        .expect("KV-cache arguments should parse");
        let args = normalize(args);

        assert_eq!(
            args.kv_cache_config.per_page_token_count,
            DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT as u32
        );
        assert_eq!(
            args.kv_cache_config.target_kv_cache_size_bytes,
            DEFAULT_TARGET_KV_CACHE_SIZE_BYTES as u64
        );
        assert_eq!(args.kv_cache_config.draft_kv_cache_size_bytes, 0);
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

    #[test]
    fn parses_a_draft_model_with_an_accepted_length_policy() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--draft-model",
                "model { model_id: 'draft' model_filename: 'draft.gguf' tokenizer_id: 'tokenizer' } token_count_policy { accepted_length { initial_draft_token_count: 4 smoothing_factor: 0.2 minimum_draft_token_count: 1 maximum_draft_token_count: 12 } }",
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
