use crate::model_runner::ActivationDType;
use crate::model_runner::InferenceDevice;
use crate::model_runner::DEFAULT_CPU_DECODE_TOKENS_PER_SECOND;
use crate::model_runner::DEFAULT_CPU_PREFILL_TOKENS_PER_SECOND;
use crate::model_runner::DEFAULT_GPU_DECODE_TOKENS_PER_SECOND;
use crate::model_runner::DEFAULT_GPU_PREFILL_TOKENS_PER_SECOND;
use crate::model_runner::DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT;
use crate::model_runner::DEFAULT_THROUGHPUT_PROFILE_INPUT_TOKEN_COUNT;
use crate::model_runner::DEFAULT_THROUGHPUT_PROFILE_MAX_OUTPUT_TOKEN_COUNT;
use crate::proto::inference_config::draft_token_count_policy::Policy;
use crate::proto::inference_config::BackendThroughputProfile as BackendThroughputProfileProto;
use crate::proto::inference_config::BackendThroughputProfiles;
use crate::proto::inference_config::DraftModelConfig;
use crate::proto::inference_config::DraftTokenCountPolicy;
use crate::proto::inference_config::FixedDraftTokenCountPolicy;
use crate::proto::inference_config::KvCacheConfig as KvCacheConfigProto;
use crate::proto::inference_config::KvCacheType as KvCacheTypeProto;
use crate::proto::inference_config::ModelConfig;
use crate::proto::inference_config::PrefixCacheCoordinationPolicy as PrefixCacheCoordinationPolicyProto;
use crate::proto::inference_config::PrefixCachePublicationPolicy as PrefixCachePublicationPolicyProto;
use crate::proto::inference_config::RequestOrchestratorConfig as RequestOrchestratorConfigProto;
use crate::proto::inference_config::RequestRoutingPolicy as RequestRoutingPolicyProto;
use crate::proto::inference_config::SchedulerConfig as SchedulerConfigProto;
use crate::proto::inference_config::SchedulingPolicy as SchedulingPolicyProto;
use crate::proto::inference_config::DEFAULT_KV_CACHE_TYPE;
use crate::proto::inference_config::DEFAULT_PREFIX_CACHE_COORDINATION_POLICY;
use crate::proto::inference_config::DEFAULT_REQUEST_ROUTING_POLICY;
use crate::proto::inference_config::DEFAULT_SCHEDULING_POLICY;
use argh::FromArgs;
use log::warn;
use std::fmt;
use std::fs;
use std::path::PathBuf;
use thousands::Separable;

const DEFAULT_TARGET_KV_CACHE_SIZE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const DEFAULT_DRAFT_TOKEN_COUNT: u64 = 4;
const DEFAULT_MAX_BATCHED_TOKEN_COUNT: usize = 512;
const DEFAULT_MAX_ACTIVE_REQUEST_COUNT: usize = 4;
const DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT: usize = 4;
const DEFAULT_PREFIX_CACHE_PUBLICATION_POLICY: PrefixCachePublicationPolicyProto =
    PrefixCachePublicationPolicyProto::OnRequestCompletion;

/// Runs text generation with a model from Hugging Face.
pub(crate) struct MainProcessArgs {
    pub(crate) model: ModelConfig,
    pub(crate) draft_model: Option<DraftModelConfig>,
    pub(crate) inference_device: InferenceDevice,
    pub(crate) activation_dtype: ActivationDType,
    pub(crate) kv_cache_config: KvCacheConfigProto,
    pub(crate) scheduler_config: SchedulerConfigProto,
    pub(crate) request_orchestrator_config: RequestOrchestratorConfigProto,
    pub(crate) input_preprocessing_thread_count: usize,
    pub(crate) trace_directory: Option<PathBuf>,
    pub(crate) request_socket: PathBuf,
    pub(crate) control_socket: PathBuf,
}

impl FromArgs for MainProcessArgs {
    fn from_args(command_name: &[&str], args: &[&str]) -> Result<Self, argh::EarlyExit> {
        RawMainProcessArgs::from_args(command_name, args)?
            .resolve()
            .map_err(argh::EarlyExit::from)
    }

    fn redact_arg_values(
        command_name: &[&str],
        args: &[&str],
    ) -> Result<Vec<String>, argh::EarlyExit> {
        RawMainProcessArgs::redact_arg_values(command_name, args)
    }
}

impl argh::TopLevelCommand for MainProcessArgs {}

/// Runs text generation with a model from Hugging Face.
#[derive(FromArgs)]
struct RawMainProcessArgs {
    /// textproto configuration for the target GGUF model
    #[argh(option, default = "default_model_config()")]
    model: ModelConfig,
    /// textproto configuration for the speculative-decoding draft model and token-count policy
    #[argh(option)]
    draft_model: Option<DraftModelConfig>,
    /// device used for model inference
    #[argh(option, default = "InferenceDevice::Gpu")]
    inference_device: InferenceDevice,
    /// data type used for model activations and KV caches
    #[argh(option, default = "ActivationDType::F16")]
    activation_dtype: ActivationDType,
    /// textproto KV-cache configuration
    #[argh(option, default = "default_kv_cache_config()")]
    kv_cache_config: KvCacheConfigProto,
    /// textproto scheduler configuration
    #[argh(option, default = "default_scheduler_config()")]
    scheduler_config: SchedulerConfigProto,
    /// inline textproto request orchestrator configuration
    #[argh(option)]
    request_orchestrator_config: Option<RequestOrchestratorConfigProto>,
    /// path to a textproto backend throughput profiles file
    #[argh(option)]
    backend_throughput_profiles_file: Option<PathBuf>,
    /// number of request-handler threads used for concurrent input preprocessing
    #[argh(option, default = "DEFAULT_INPUT_PREPROCESSING_THREAD_COUNT")]
    input_preprocessing_thread_count: usize,
    /// directory where a Chrome trace is written
    #[argh(option)]
    trace_directory: Option<PathBuf>,
    /// unix domain socket exposed to local inference clients
    #[argh(option, default = "default_request_socket()")]
    request_socket: PathBuf,
    /// unix domain socket exposed to lifecycle-control clients
    #[argh(option, default = "default_control_socket()")]
    control_socket: PathBuf,
}

impl RawMainProcessArgs {
    fn resolve(self) -> Result<MainProcessArgs, String> {
        let request_orchestrator_config = resolve_request_orchestrator_config(
            self.request_orchestrator_config,
            self.backend_throughput_profiles_file,
        )?;
        Ok(MainProcessArgs {
            model: self.model,
            draft_model: self.draft_model,
            inference_device: self.inference_device,
            activation_dtype: self.activation_dtype,
            kv_cache_config: self.kv_cache_config,
            scheduler_config: self.scheduler_config,
            request_orchestrator_config,
            input_preprocessing_thread_count: self.input_preprocessing_thread_count,
            trace_directory: self.trace_directory,
            request_socket: self.request_socket,
            control_socket: self.control_socket,
        })
    }
}

fn resolve_request_orchestrator_config(
    config: Option<RequestOrchestratorConfigProto>,
    throughput_profiles_file: Option<PathBuf>,
) -> Result<RequestOrchestratorConfigProto, String> {
    let has_inline_throughput_profiles = config
        .as_ref()
        .is_some_and(|config| config.throughput_profiles.is_some());
    let mut config = config.unwrap_or_else(default_request_orchestrator_config);
    if let Some(path) = throughput_profiles_file {
        if has_inline_throughput_profiles {
            return Err(
                "backend throughput profiles cannot be supplied both inline and through --backend-throughput-profiles-file"
                    .to_owned(),
            );
        }
        let contents = fs::read_to_string(&path).map_err(|error| {
            format!(
                "failed to read backend throughput profiles file {}: {error}",
                path.display()
            )
        })?;
        let profiles: BackendThroughputProfiles = contents.parse().map_err(|error| {
            format!(
                "failed to parse backend throughput profiles file {}: {error}",
                path.display()
            )
        })?;
        if profiles.profiles.is_empty() {
            return Err(format!(
                "backend throughput profiles file {} contains no profiles",
                path.display()
            ));
        }
        config.throughput_profiles = Some(profiles);
    }
    Ok(config)
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
        write!(formatter, "{}", self.request_orchestrator_config)?;
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
        self.scheduler_config.validate()?;
        self.request_orchestrator_config.validate()?;
        self.scheduler_config
            .prefix_cache_coordination_policy()
            .validate(self.kv_cache_config.kv_cache_type())?;
        self.request_orchestrator_config
            .routing_policy()
            .validate(self.kv_cache_config.kv_cache_type())?;
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
        prefix_cache_coordination_policy: DEFAULT_PREFIX_CACHE_COORDINATION_POLICY.into(),
    }
}

fn default_request_orchestrator_config() -> RequestOrchestratorConfigProto {
    RequestOrchestratorConfigProto {
        routing_policy: DEFAULT_REQUEST_ROUTING_POLICY.into(),
        throughput_profiles: Some(BackendThroughputProfiles {
            profiles: vec![default_backend_throughput_profile()],
        }),
    }
}

fn default_backend_throughput_profile() -> BackendThroughputProfileProto {
    BackendThroughputProfileProto {
        input_token_count: DEFAULT_THROUGHPUT_PROFILE_INPUT_TOKEN_COUNT,
        max_output_token_count: DEFAULT_THROUGHPUT_PROFILE_MAX_OUTPUT_TOKEN_COUNT,
        cpu_prefill_tokens_per_second: DEFAULT_CPU_PREFILL_TOKENS_PER_SECOND,
        cpu_decode_tokens_per_second: DEFAULT_CPU_DECODE_TOKENS_PER_SECOND,
        gpu_prefill_tokens_per_second: DEFAULT_GPU_PREFILL_TOKENS_PER_SECOND,
        gpu_decode_tokens_per_second: DEFAULT_GPU_DECODE_TOKENS_PER_SECOND,
    }
}

fn default_kv_cache_config() -> KvCacheConfigProto {
    KvCacheConfigProto {
        kv_cache_type: DEFAULT_KV_CACHE_TYPE.into(),
        per_page_token_count: 0,
        target_kv_cache_size_bytes: DEFAULT_TARGET_KV_CACHE_SIZE_BYTES as u64,
        draft_kv_cache_size_bytes: 0,
        prefix_cache_publication_policy: DEFAULT_PREFIX_CACHE_PUBLICATION_POLICY.into(),
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
    normalize_request_orchestrator_config(&mut args.request_orchestrator_config);
    if args.input_preprocessing_thread_count == 0 {
        warn!(
            "Input preprocessing thread count is 0; using default value \
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
            "Target KV-cache size is 0; using default value {}",
            DEFAULT_TARGET_KV_CACHE_SIZE_BYTES.separate_with_commas()
        );
        config.target_kv_cache_size_bytes = DEFAULT_TARGET_KV_CACHE_SIZE_BYTES as u64;
    }
    let kv_cache_type = match KvCacheTypeProto::try_from(config.kv_cache_type) {
        Ok(KvCacheTypeProto::Unspecified) => {
            warn!("KV-cache type is unspecified; using default value contiguous");
            config.kv_cache_type = DEFAULT_KV_CACHE_TYPE.into();
            DEFAULT_KV_CACHE_TYPE
        }
        Ok(kv_cache_type) => kv_cache_type,
        Err(_) => return,
    };
    if kv_cache_type != KvCacheTypeProto::Contiguous && config.per_page_token_count == 0 {
        warn!(
            "KV-cache page token count is 0; using default value \
             {DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT}"
        );
        config.per_page_token_count = DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT as u32;
    }
    if let Ok(PrefixCachePublicationPolicyProto::Unspecified) =
        PrefixCachePublicationPolicyProto::try_from(config.prefix_cache_publication_policy)
    {
        config.prefix_cache_publication_policy = DEFAULT_PREFIX_CACHE_PUBLICATION_POLICY.into();
    }
}

fn normalize_scheduler_config(config: &mut SchedulerConfigProto) {
    if config.max_batched_token_count == 0 {
        warn!(
            "Maximum batched token count is 0; using default value {DEFAULT_MAX_BATCHED_TOKEN_COUNT}"
        );
        config.max_batched_token_count = DEFAULT_MAX_BATCHED_TOKEN_COUNT as u32;
    }
    if config.max_active_request_count == 0 {
        warn!(
            "Maximum active request count is 0; using default value {DEFAULT_MAX_ACTIVE_REQUEST_COUNT}"
        );
        config.max_active_request_count = DEFAULT_MAX_ACTIVE_REQUEST_COUNT as u32;
    }
    if let Ok(SchedulingPolicyProto::Unspecified) =
        SchedulingPolicyProto::try_from(config.scheduling_policy)
    {
        warn!("Scheduling policy is unspecified; using default value first-come-first-served");
        config.scheduling_policy = DEFAULT_SCHEDULING_POLICY.into();
    }
    if let Ok(PrefixCacheCoordinationPolicyProto::Unspecified) =
        PrefixCacheCoordinationPolicyProto::try_from(config.prefix_cache_coordination_policy)
    {
        config.prefix_cache_coordination_policy = DEFAULT_PREFIX_CACHE_COORDINATION_POLICY.into();
    }
}

fn normalize_request_orchestrator_config(config: &mut RequestOrchestratorConfigProto) {
    if let Ok(RequestRoutingPolicyProto::Unspecified) =
        RequestRoutingPolicyProto::try_from(config.routing_policy)
    {
        warn!("Request routing policy is unspecified; using default value round-robin");
        config.routing_policy = DEFAULT_REQUEST_ROUTING_POLICY.into();
    }
    if config.throughput_profiles.is_none() {
        warn!("Backend throughput profiles are missing; using the default benchmark profile");
        config.throughput_profiles = Some(BackendThroughputProfiles {
            profiles: vec![default_backend_throughput_profile()],
        });
    }
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
                concat!(
                    "max_batched_token_count: 1024 max_active_request_count: 8 ",
                    "scheduling_policy: SCHEDULING_POLICY_SHORTEST_PREFILL_FIRST ",
                    "prefix_cache_coordination_policy: ",
                    "PREFIX_CACHE_COORDINATION_POLICY_EXACT_INPUT_SINGLE_FLIGHT_UNTIL_REQUEST_COMPLETION",
                ),
            ],
        )
        .expect("scheduler arguments should parse");

        assert_eq!(args.scheduler_config.max_batched_token_count, 1024);
        assert_eq!(args.scheduler_config.max_active_request_count, 8);
        assert_eq!(
            args.scheduler_config.scheduling_policy,
            SchedulingPolicyProto::ShortestPrefillFirst as i32
        );
        assert_eq!(
            args.scheduler_config.prefix_cache_coordination_policy,
            PrefixCacheCoordinationPolicyProto::ExactInputSingleFlightUntilRequestCompletion as i32
        );
    }

    #[test]
    fn parses_request_orchestrator_configuration() {
        for (policy_name, expected_policy) in [
            (
                "REQUEST_ROUTING_POLICY_ROUND_ROBIN",
                RequestRoutingPolicyProto::RoundRobin,
            ),
            (
                "REQUEST_ROUTING_POLICY_LEAST_REQUESTS",
                RequestRoutingPolicyProto::LeastRequests,
            ),
            (
                "REQUEST_ROUTING_POLICY_LEAST_ESTIMATED_WORK",
                RequestRoutingPolicyProto::LeastEstimatedWork,
            ),
            (
                "REQUEST_ROUTING_POLICY_MOST_CACHED_PREFIX",
                RequestRoutingPolicyProto::MostCachedPrefix,
            ),
            (
                "REQUEST_ROUTING_POLICY_CACHE_AWARE_LEAST_ESTIMATED_WORK",
                RequestRoutingPolicyProto::CacheAwareLeastEstimatedWork,
            ),
        ] {
            let config = format!(
                "routing_policy: {policy_name} \
                 throughput_profiles {{ profiles {{ input_token_count: 64 max_output_token_count: 16 \
                 cpu_prefill_tokens_per_second: 100 cpu_decode_tokens_per_second: 50 \
                 gpu_prefill_tokens_per_second: 200 gpu_decode_tokens_per_second: 150 }} \
                 profiles {{ input_token_count: 512 max_output_token_count: 128 \
                 cpu_prefill_tokens_per_second: 300 cpu_decode_tokens_per_second: 250 \
                 gpu_prefill_tokens_per_second: 400 gpu_decode_tokens_per_second: 350 }} }}"
            );
            let args = MainProcessArgs::from_args(
                &["mini-vllm-rs"],
                &["--request-orchestrator-config", &config],
            )
            .expect("request orchestrator arguments should parse");

            assert_eq!(
                args.request_orchestrator_config.routing_policy,
                expected_policy as i32
            );
            assert_eq!(
                args.request_orchestrator_config.throughput_profiles,
                Some(BackendThroughputProfiles {
                    profiles: vec![
                        BackendThroughputProfileProto {
                            input_token_count: 64,
                            max_output_token_count: 16,
                            cpu_prefill_tokens_per_second: 100,
                            cpu_decode_tokens_per_second: 50,
                            gpu_prefill_tokens_per_second: 200,
                            gpu_decode_tokens_per_second: 150,
                        },
                        BackendThroughputProfileProto {
                            input_token_count: 512,
                            max_output_token_count: 128,
                            cpu_prefill_tokens_per_second: 300,
                            cpu_decode_tokens_per_second: 250,
                            gpu_prefill_tokens_per_second: 400,
                            gpu_decode_tokens_per_second: 350,
                        },
                    ],
                })
            );
        }
    }

    #[test]
    fn reads_backend_throughput_profiles_from_file() {
        let directory = tempfile::tempdir().expect("temporary directory should be created");
        let path = directory
            .path()
            .join("backend-throughput-profiles.textproto");
        fs::write(
            &path,
            concat!(
                "profiles { input_token_count: 64 max_output_token_count: 16 ",
                "cpu_prefill_tokens_per_second: 100 cpu_decode_tokens_per_second: 50 ",
                "gpu_prefill_tokens_per_second: 200 gpu_decode_tokens_per_second: 150 }",
            ),
        )
        .expect("backend throughput profiles should be written");

        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--request-orchestrator-config",
                "routing_policy: REQUEST_ROUTING_POLICY_LEAST_REQUESTS",
                "--backend-throughput-profiles-file",
                path.to_str().expect("temporary path should be UTF-8"),
            ],
        )
        .expect("backend throughput profiles file should parse");

        assert_eq!(
            args.request_orchestrator_config.routing_policy,
            RequestRoutingPolicyProto::LeastRequests as i32
        );
        assert_eq!(
            args.request_orchestrator_config.throughput_profiles,
            Some(BackendThroughputProfiles {
                profiles: vec![BackendThroughputProfileProto {
                    input_token_count: 64,
                    max_output_token_count: 16,
                    cpu_prefill_tokens_per_second: 100,
                    cpu_decode_tokens_per_second: 50,
                    gpu_prefill_tokens_per_second: 200,
                    gpu_decode_tokens_per_second: 150,
                }],
            })
        );
    }

    #[test]
    fn reads_recommended_backend_throughput_profiles() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("benchmarks/backend_throughput_profiles.textproto");
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--backend-throughput-profiles-file",
                path.to_str().expect("profile path should be UTF-8"),
            ],
        )
        .expect("recommended backend throughput profiles should parse");

        args.validate()
            .expect("recommended backend throughput profiles should be valid");
        assert_eq!(
            args.request_orchestrator_config
                .throughput_profiles
                .expect("throughput profiles should be present")
                .profiles
                .len(),
            12
        );
    }

    #[test]
    fn rejects_invalid_backend_throughput_profiles_file() {
        let directory = tempfile::tempdir().expect("temporary directory should be created");
        let path = directory
            .path()
            .join("backend-throughput-profiles.textproto");
        fs::write(&path, "not valid textproto")
            .expect("backend throughput profiles should be written");

        let result = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--backend-throughput-profiles-file",
                path.to_str().expect("temporary path should be UTF-8"),
            ],
        );
        let Err(error) = result else {
            panic!("invalid backend throughput profiles should fail")
        };

        assert!(error
            .output
            .contains("failed to parse backend throughput profiles file"));
    }

    #[test]
    fn rejects_inline_and_file_backend_throughput_profiles() {
        let result = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--request-orchestrator-config",
                concat!(
                    "throughput_profiles { profiles { input_token_count: 64 max_output_token_count: 16 ",
                    "cpu_prefill_tokens_per_second: 100 cpu_decode_tokens_per_second: 50 ",
                    "gpu_prefill_tokens_per_second: 200 gpu_decode_tokens_per_second: 150 } }",
                ),
                "--backend-throughput-profiles-file",
                "backend-throughput-profiles.textproto",
            ],
        );
        let Err(error) = result else {
            panic!("backend throughput profile sources should be mutually exclusive")
        };

        assert_eq!(
            error.output,
            "backend throughput profiles cannot be supplied both inline and through --backend-throughput-profiles-file"
        );
    }

    #[test]
    fn parses_kv_cache_configuration() {
        let args = MainProcessArgs::from_args(
            &["mini-vllm-rs"],
            &[
                "--kv-cache-config",
                concat!(
                    "kv_cache_type: KV_CACHE_TYPE_PAGED_PREFIX per_page_token_count: 32 ",
                    "target_kv_cache_size_bytes: 1073741824 ",
                    "draft_kv_cache_size_bytes: 268435456 ",
                    "prefix_cache_publication_policy: ",
                    "PREFIX_CACHE_PUBLICATION_POLICY_INCREMENTAL",
                ),
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
        assert_eq!(
            args.kv_cache_config.prefix_cache_publication_policy,
            PrefixCachePublicationPolicyProto::Incremental as i32
        );
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
        assert_eq!(
            args.kv_cache_config.prefix_cache_publication_policy,
            PrefixCachePublicationPolicyProto::OnRequestCompletion as i32
        );
    }

    #[test]
    fn normalizes_an_empty_kv_cache_configuration() {
        let mut config = KvCacheConfigProto::default();

        normalize_kv_cache_config(&mut config);

        assert_eq!(config.kv_cache_type, KvCacheTypeProto::Contiguous as i32);
        assert_eq!(
            config.target_kv_cache_size_bytes,
            DEFAULT_TARGET_KV_CACHE_SIZE_BYTES as u64
        );
        assert_eq!(
            config.prefix_cache_publication_policy,
            PrefixCachePublicationPolicyProto::OnRequestCompletion as i32
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn normalizes_an_empty_scheduler_configuration() {
        let mut config = SchedulerConfigProto::default();

        normalize_scheduler_config(&mut config);

        assert_eq!(
            config.max_batched_token_count,
            DEFAULT_MAX_BATCHED_TOKEN_COUNT as u32
        );
        assert_eq!(
            config.max_active_request_count,
            DEFAULT_MAX_ACTIVE_REQUEST_COUNT as u32
        );
        assert_eq!(
            config.scheduling_policy,
            SchedulingPolicyProto::FirstComeFirstServed as i32
        );
        assert_eq!(
            config.prefix_cache_coordination_policy,
            PrefixCacheCoordinationPolicyProto::Independent as i32
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn normalizes_an_empty_request_orchestrator_configuration() {
        let mut config = RequestOrchestratorConfigProto::default();

        normalize_request_orchestrator_config(&mut config);

        assert_eq!(
            config.routing_policy,
            RequestRoutingPolicyProto::RoundRobin as i32
        );
        assert_eq!(
            config.throughput_profiles,
            Some(BackendThroughputProfiles {
                profiles: vec![default_backend_throughput_profile()],
            })
        );
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_exact_input_single_flight_without_prefix_caching() {
        let mut args = normalize(
            MainProcessArgs::from_args(&["mini-vllm-rs"], &[])
                .expect("default arguments should parse"),
        );
        args.scheduler_config.prefix_cache_coordination_policy =
            PrefixCacheCoordinationPolicyProto::ExactInputSingleFlightUntilRequestCompletion.into();

        let error = args.validate().unwrap_err();

        assert_eq!(
            error.to_string(),
            "exact-input single-flight requires paged-prefix KV caching"
        );

        args.kv_cache_config.kv_cache_type = KvCacheTypeProto::PagedPrefix.into();
        args.kv_cache_config.per_page_token_count = DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT as u32;
        assert!(args.validate().is_ok());
    }

    #[test]
    fn rejects_most_cached_prefix_routing_without_prefix_caching() {
        let mut args = normalize(
            MainProcessArgs::from_args(&["mini-vllm-rs"], &[])
                .expect("default arguments should parse"),
        );
        args.request_orchestrator_config.routing_policy =
            RequestRoutingPolicyProto::MostCachedPrefix.into();

        let error = args.validate().unwrap_err();

        assert_eq!(
            error.to_string(),
            "most-cached-prefix routing requires paged-prefix KV caching"
        );

        args.kv_cache_config.kv_cache_type = KvCacheTypeProto::PagedPrefix.into();
        args.kv_cache_config.per_page_token_count = DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT as u32;
        assert!(args.validate().is_ok());
    }

    #[test]
    fn rejects_cache_aware_least_estimated_work_without_prefix_caching() {
        let mut args = normalize(
            MainProcessArgs::from_args(&["mini-vllm-rs"], &[])
                .expect("default arguments should parse"),
        );
        args.request_orchestrator_config.routing_policy =
            RequestRoutingPolicyProto::CacheAwareLeastEstimatedWork.into();

        let error = args.validate().unwrap_err();

        assert_eq!(
            error.to_string(),
            "cache-aware-least-estimated-work routing requires paged-prefix KV caching"
        );

        args.kv_cache_config.kv_cache_type = KvCacheTypeProto::PagedPrefix.into();
        args.kv_cache_config.per_page_token_count = DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT as u32;
        assert!(args.validate().is_ok());
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
                concat!(
                    "model { model_id: 'draft' model_filename: 'draft.gguf' ",
                    "tokenizer_id: 'tokenizer' } ",
                    "token_count_policy { fixed { draft_token_count: 6 } }",
                ),
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
                concat!(
                    "model { model_id: 'draft' model_filename: 'draft.gguf' ",
                    "tokenizer_id: 'tokenizer' } token_count_policy { acceptance_rate { ",
                    "initial_draft_token_count: 4 decrease_threshold: 0.4 ",
                    "increase_threshold: 0.8 minimum_draft_token_count: 1 ",
                    "maximum_draft_token_count: 8 } }",
                ),
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
                concat!(
                    "model { model_id: 'draft' model_filename: 'draft.gguf' ",
                    "tokenizer_id: 'tokenizer' } token_count_policy { accepted_length { ",
                    "initial_draft_token_count: 4 smoothing_factor: 0.2 ",
                    "minimum_draft_token_count: 1 maximum_draft_token_count: 12 } }",
                ),
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
