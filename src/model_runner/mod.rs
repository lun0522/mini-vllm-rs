pub(crate) mod client;
pub(crate) mod server;

use std::fmt;
use std::str::FromStr;

use crate::proto::inference_config::BackendThroughputProfile as BackendThroughputProfileProto;
use crate::proto::inference_config::BackendThroughputProfiles as BackendThroughputProfilesProto;
use crate::proto::inference_config::KvCacheConfig as KvCacheConfigProto;
use crate::proto::inference_config::KvCacheType as KvCacheTypeProto;
use crate::proto::inference_config::PrefixCacheCoordinationPolicy as PrefixCacheCoordinationPolicyProto;
use crate::proto::inference_config::PrefixCachePublicationPolicy as PrefixCachePublicationPolicyProto;
use crate::proto::inference_config::RequestOrchestratorConfig as RequestOrchestratorConfigProto;
use crate::proto::inference_config::RequestRoutingPolicy as RequestRoutingPolicyProto;
use crate::proto::inference_config::SchedulerConfig as SchedulerConfigProto;
use crate::proto::inference_config::SchedulingPolicy as SchedulingPolicyProto;
use candle_core::DType;

pub(crate) const DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT: usize = 16;

// Use the phase-specific rates measured for the 128-input, 512-output workload
// in benchmarks/backend_throughput_profiles.md as the fallback profile.
pub(crate) const DEFAULT_THROUGHPUT_PROFILE_INPUT_TOKEN_COUNT: u64 = 128;
pub(crate) const DEFAULT_THROUGHPUT_PROFILE_MAX_OUTPUT_TOKEN_COUNT: u64 = 512;
pub(crate) const DEFAULT_CPU_PREFILL_TOKENS_PER_SECOND: u64 = 75;
pub(crate) const DEFAULT_CPU_DECODE_TOKENS_PER_SECOND: u64 = 48;
pub(crate) const DEFAULT_GPU_PREFILL_TOKENS_PER_SECOND: u64 = 12_086;
pub(crate) const DEFAULT_GPU_DECODE_TOKENS_PER_SECOND: u64 = 396;

/// Phase-specific inference throughput measured for one backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BackendThroughput {
    pub(crate) prefill_tokens_per_second: u64,
    pub(crate) decode_tokens_per_second: u64,
}

/// Backend throughput measured for one representative request workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BackendThroughputProfile {
    pub(crate) input_token_count: u64,
    pub(crate) max_output_token_count: u64,
    pub(crate) cpu: BackendThroughput,
    pub(crate) gpu: BackendThroughput,
}

impl Default for BackendThroughputProfile {
    fn default() -> Self {
        Self {
            input_token_count: DEFAULT_THROUGHPUT_PROFILE_INPUT_TOKEN_COUNT,
            max_output_token_count: DEFAULT_THROUGHPUT_PROFILE_MAX_OUTPUT_TOKEN_COUNT,
            cpu: BackendThroughput {
                prefill_tokens_per_second: DEFAULT_CPU_PREFILL_TOKENS_PER_SECOND,
                decode_tokens_per_second: DEFAULT_CPU_DECODE_TOKENS_PER_SECOND,
            },
            gpu: BackendThroughput {
                prefill_tokens_per_second: DEFAULT_GPU_PREFILL_TOKENS_PER_SECOND,
                decode_tokens_per_second: DEFAULT_GPU_DECODE_TOKENS_PER_SECOND,
            },
        }
    }
}

impl From<BackendThroughputProfileProto> for BackendThroughputProfile {
    fn from(profile: BackendThroughputProfileProto) -> Self {
        Self {
            input_token_count: profile.input_token_count,
            max_output_token_count: profile.max_output_token_count,
            cpu: BackendThroughput {
                prefill_tokens_per_second: profile.cpu_prefill_tokens_per_second,
                decode_tokens_per_second: profile.cpu_decode_tokens_per_second,
            },
            gpu: BackendThroughput {
                prefill_tokens_per_second: profile.gpu_prefill_tokens_per_second,
                decode_tokens_per_second: profile.gpu_decode_tokens_per_second,
            },
        }
    }
}

impl From<BackendThroughputProfile> for BackendThroughputProfileProto {
    fn from(profile: BackendThroughputProfile) -> Self {
        Self {
            input_token_count: profile.input_token_count,
            max_output_token_count: profile.max_output_token_count,
            cpu_prefill_tokens_per_second: profile.cpu.prefill_tokens_per_second,
            cpu_decode_tokens_per_second: profile.cpu.decode_tokens_per_second,
            gpu_prefill_tokens_per_second: profile.gpu.prefill_tokens_per_second,
            gpu_decode_tokens_per_second: profile.gpu.decode_tokens_per_second,
        }
    }
}

impl fmt::Display for BackendThroughputProfile {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "input_token_count={}, max_output_token_count={}, \
             cpu_prefill_tokens_per_second={}, cpu_decode_tokens_per_second={}, \
             gpu_prefill_tokens_per_second={}, gpu_decode_tokens_per_second={}",
            self.input_token_count,
            self.max_output_token_count,
            self.cpu.prefill_tokens_per_second,
            self.cpu.decode_tokens_per_second,
            self.gpu.prefill_tokens_per_second,
            self.gpu.decode_tokens_per_second,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RequestOrchestratorConfig {
    pub(crate) routing_policy: RequestRoutingPolicy,
    pub(crate) throughput_profiles: Vec<BackendThroughputProfile>,
}

impl From<RequestOrchestratorConfigProto> for RequestOrchestratorConfig {
    fn from(config: RequestOrchestratorConfigProto) -> Self {
        Self {
            routing_policy: config.routing_policy().into(),
            throughput_profiles: config
                .throughput_profiles
                .into_iter()
                .flat_map(|profiles| profiles.profiles)
                .map(Into::into)
                .collect(),
        }
    }
}

impl From<RequestOrchestratorConfig> for RequestOrchestratorConfigProto {
    fn from(config: RequestOrchestratorConfig) -> Self {
        Self {
            routing_policy: RequestRoutingPolicyProto::from(config.routing_policy).into(),
            throughput_profiles: Some(BackendThroughputProfilesProto {
                profiles: config
                    .throughput_profiles
                    .into_iter()
                    .map(Into::into)
                    .collect(),
            }),
        }
    }
}

impl fmt::Display for RequestOrchestratorConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "routing_policy={}", self.routing_policy)?;
        for profile in &self.throughput_profiles {
            write!(formatter, ", throughput_profile=({profile})")?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RequestRoutingPolicy {
    RoundRobin,
    LeastRequests,
    LeastEstimatedWork,
    MostCachedPrefix,
    CacheAwareLeastEstimatedWork,
}

impl From<RequestRoutingPolicyProto> for RequestRoutingPolicy {
    fn from(policy: RequestRoutingPolicyProto) -> Self {
        match policy {
            RequestRoutingPolicyProto::Unspecified => unreachable!(
                "validated request orchestrator configuration should specify a routing policy"
            ),
            RequestRoutingPolicyProto::RoundRobin => Self::RoundRobin,
            RequestRoutingPolicyProto::LeastRequests => Self::LeastRequests,
            RequestRoutingPolicyProto::LeastEstimatedWork => Self::LeastEstimatedWork,
            RequestRoutingPolicyProto::MostCachedPrefix => Self::MostCachedPrefix,
            RequestRoutingPolicyProto::CacheAwareLeastEstimatedWork => {
                Self::CacheAwareLeastEstimatedWork
            }
        }
    }
}

impl From<RequestRoutingPolicy> for RequestRoutingPolicyProto {
    fn from(policy: RequestRoutingPolicy) -> Self {
        match policy {
            RequestRoutingPolicy::RoundRobin => Self::RoundRobin,
            RequestRoutingPolicy::LeastRequests => Self::LeastRequests,
            RequestRoutingPolicy::LeastEstimatedWork => Self::LeastEstimatedWork,
            RequestRoutingPolicy::MostCachedPrefix => Self::MostCachedPrefix,
            RequestRoutingPolicy::CacheAwareLeastEstimatedWork => {
                Self::CacheAwareLeastEstimatedWork
            }
        }
    }
}

impl fmt::Display for RequestRoutingPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::RoundRobin => "round-robin",
            Self::LeastRequests => "least-requests",
            Self::LeastEstimatedWork => "least-estimated-work",
            Self::MostCachedPrefix => "most-cached-prefix",
            Self::CacheAwareLeastEstimatedWork => "cache-aware-least-estimated-work",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KvCacheConfig {
    pub(crate) kv_cache_type: KvCacheType,
    pub(crate) target_kv_cache_size_bytes: usize,
    pub(crate) draft_kv_cache_size_bytes: Option<usize>,
    pub(crate) prefix_cache_publication_policy: PrefixCachePublicationPolicy,
}

impl From<KvCacheConfigProto> for KvCacheConfig {
    fn from(config: KvCacheConfigProto) -> Self {
        let kv_cache_type = match KvCacheTypeProto::try_from(config.kv_cache_type)
            .expect("validated KV-cache configuration should contain a supported cache type")
        {
            KvCacheTypeProto::Unspecified => {
                unreachable!("validated KV-cache configuration should specify a cache type")
            }
            KvCacheTypeProto::Contiguous => KvCacheType::Contiguous,
            KvCacheTypeProto::Paged => KvCacheType::Paged {
                per_page_token_count: config.per_page_token_count as usize,
                enable_prefix_caching: false,
            },
            KvCacheTypeProto::PagedPrefix => KvCacheType::Paged {
                per_page_token_count: config.per_page_token_count as usize,
                enable_prefix_caching: true,
            },
        };
        let prefix_cache_publication_policy = PrefixCachePublicationPolicyProto::try_from(
            config.prefix_cache_publication_policy,
        )
            .expect("validated KV-cache configuration should contain a supported prefix-cache publication policy")
            .into();
        Self {
            kv_cache_type,
            target_kv_cache_size_bytes: config.target_kv_cache_size_bytes as usize,
            draft_kv_cache_size_bytes: (config.draft_kv_cache_size_bytes != 0)
                .then_some(config.draft_kv_cache_size_bytes as usize),
            prefix_cache_publication_policy,
        }
    }
}

impl From<KvCacheConfig> for KvCacheConfigProto {
    fn from(config: KvCacheConfig) -> Self {
        let (kv_cache_type, per_page_token_count) = match config.kv_cache_type {
            KvCacheType::Contiguous => (KvCacheTypeProto::Contiguous, 0),
            KvCacheType::Paged {
                per_page_token_count,
                enable_prefix_caching: false,
            } => (KvCacheTypeProto::Paged, per_page_token_count as u32),
            KvCacheType::Paged {
                per_page_token_count,
                enable_prefix_caching: true,
            } => (KvCacheTypeProto::PagedPrefix, per_page_token_count as u32),
        };
        Self {
            kv_cache_type: kv_cache_type.into(),
            per_page_token_count,
            target_kv_cache_size_bytes: config.target_kv_cache_size_bytes as u64,
            draft_kv_cache_size_bytes: config
                .draft_kv_cache_size_bytes
                .map(|size| size as u64)
                .unwrap_or(0),
            prefix_cache_publication_policy: PrefixCachePublicationPolicyProto::from(
                config.prefix_cache_publication_policy,
            )
            .into(),
        }
    }
}

impl fmt::Display for KvCacheConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "kv_cache_type={} target_kv_cache_size_bytes={} prefix_cache_publication_policy={}",
            self.kv_cache_type,
            self.target_kv_cache_size_bytes,
            self.prefix_cache_publication_policy,
        )?;
        if let Some(size_bytes) = self.draft_kv_cache_size_bytes {
            write!(formatter, " draft_kv_cache_size_bytes={size_bytes}")
        } else {
            formatter.write_str(" draft_kv_cache_size_bytes=inferred")
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefixCachePublicationPolicy {
    OnRequestCompletion,
    Incremental,
}

impl From<PrefixCachePublicationPolicyProto> for PrefixCachePublicationPolicy {
    fn from(policy: PrefixCachePublicationPolicyProto) -> Self {
        match policy {
            PrefixCachePublicationPolicyProto::Unspecified => unreachable!(
                "validated KV-cache configuration should specify a prefix-cache publication policy"
            ),
            PrefixCachePublicationPolicyProto::OnRequestCompletion => Self::OnRequestCompletion,
            PrefixCachePublicationPolicyProto::Incremental => Self::Incremental,
        }
    }
}

impl From<PrefixCachePublicationPolicy> for PrefixCachePublicationPolicyProto {
    fn from(policy: PrefixCachePublicationPolicy) -> Self {
        match policy {
            PrefixCachePublicationPolicy::OnRequestCompletion => Self::OnRequestCompletion,
            PrefixCachePublicationPolicy::Incremental => Self::Incremental,
        }
    }
}

impl fmt::Display for PrefixCachePublicationPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::OnRequestCompletion => "on-request-completion",
            Self::Incremental => "incremental",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ActivationDType {
    F32,
    F16,
}

impl ActivationDType {
    fn cli_value(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }
}

impl From<ActivationDType> for DType {
    fn from(value: ActivationDType) -> Self {
        match value {
            ActivationDType::F32 => Self::F32,
            ActivationDType::F16 => Self::F16,
        }
    }
}

impl fmt::Display for ActivationDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.cli_value())
    }
}

impl FromStr for ActivationDType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            unsupported => Err(format!("unsupported activation dtype: {unsupported}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SchedulingPolicy {
    /// Admits requests in arrival order, schedules all available decode work first, then lets
    /// active prefills consume the remaining token budget in admission order.
    FirstComeFirstServed,
    /// Admits shorter prefills first, schedules all available decode work first, then gives the
    /// remaining token budget to active prefills with the fewest tokens left.
    ShortestPrefillFirst,
    /// Admits requests in arrival order, schedules all available decode work first, then rotates
    /// which active prefill receives the remaining token budget first across iterations.
    RoundRobin,
}

impl SchedulingPolicy {
    fn cli_value(self) -> &'static str {
        match self {
            Self::FirstComeFirstServed => "first-come-first-served",
            Self::ShortestPrefillFirst => "shortest-prefill-first",
            Self::RoundRobin => "round-robin",
        }
    }
}

impl fmt::Display for SchedulingPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.cli_value())
    }
}

impl FromStr for SchedulingPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "first-come-first-served" => Ok(Self::FirstComeFirstServed),
            "shortest-prefill-first" => Ok(Self::ShortestPrefillFirst),
            "round-robin" => Ok(Self::RoundRobin),
            unsupported => Err(format!("unsupported scheduling policy: {unsupported}")),
        }
    }
}

impl From<SchedulingPolicyProto> for SchedulingPolicy {
    fn from(policy: SchedulingPolicyProto) -> Self {
        match policy {
            SchedulingPolicyProto::Unspecified => {
                unreachable!("validated scheduler configuration should specify a policy")
            }
            SchedulingPolicyProto::FirstComeFirstServed => Self::FirstComeFirstServed,
            SchedulingPolicyProto::ShortestPrefillFirst => Self::ShortestPrefillFirst,
            SchedulingPolicyProto::RoundRobin => Self::RoundRobin,
        }
    }
}

impl From<SchedulingPolicy> for SchedulingPolicyProto {
    fn from(policy: SchedulingPolicy) -> Self {
        match policy {
            SchedulingPolicy::FirstComeFirstServed => Self::FirstComeFirstServed,
            SchedulingPolicy::ShortestPrefillFirst => Self::ShortestPrefillFirst,
            SchedulingPolicy::RoundRobin => Self::RoundRobin,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerConfig {
    pub(crate) max_batched_token_count: usize,
    pub(crate) max_active_request_count: usize,
    pub(crate) scheduling_policy: SchedulingPolicy,
    pub(crate) prefix_cache_coordination_policy: PrefixCacheCoordinationPolicy,
}

impl From<SchedulerConfigProto> for SchedulerConfig {
    fn from(config: SchedulerConfigProto) -> Self {
        Self {
            max_batched_token_count: config.max_batched_token_count as usize,
            max_active_request_count: config.max_active_request_count as usize,
            scheduling_policy: config.scheduling_policy().into(),
            prefix_cache_coordination_policy: config.prefix_cache_coordination_policy().into(),
        }
    }
}

impl From<SchedulerConfig> for SchedulerConfigProto {
    fn from(config: SchedulerConfig) -> Self {
        Self {
            max_batched_token_count: config.max_batched_token_count as u32,
            max_active_request_count: config.max_active_request_count as u32,
            scheduling_policy: SchedulingPolicyProto::from(config.scheduling_policy).into(),
            prefix_cache_coordination_policy: PrefixCacheCoordinationPolicyProto::from(
                config.prefix_cache_coordination_policy,
            )
            .into(),
        }
    }
}

impl fmt::Display for SchedulerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "max_batched_tokens={} max_active_requests={} scheduling_policy={} prefix_cache_coordination_policy={}",
            self.max_batched_token_count,
            self.max_active_request_count,
            self.scheduling_policy,
            self.prefix_cache_coordination_policy,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefixCacheCoordinationPolicy {
    Independent,
    ExactInputSingleFlightUntilRequestCompletion,
    ExactInputSingleFlightUntilPrefixPublished,
}

impl From<PrefixCacheCoordinationPolicyProto> for PrefixCacheCoordinationPolicy {
    fn from(policy: PrefixCacheCoordinationPolicyProto) -> Self {
        match policy {
            PrefixCacheCoordinationPolicyProto::Unspecified => unreachable!(
                "validated scheduler configuration should specify a prefix-cache coordination policy"
            ),
            PrefixCacheCoordinationPolicyProto::Independent => Self::Independent,
            PrefixCacheCoordinationPolicyProto::ExactInputSingleFlightUntilRequestCompletion => {
                Self::ExactInputSingleFlightUntilRequestCompletion
            }
            PrefixCacheCoordinationPolicyProto::ExactInputSingleFlightUntilPrefixPublished => {
                Self::ExactInputSingleFlightUntilPrefixPublished
            }
        }
    }
}

impl From<PrefixCacheCoordinationPolicy> for PrefixCacheCoordinationPolicyProto {
    fn from(policy: PrefixCacheCoordinationPolicy) -> Self {
        match policy {
            PrefixCacheCoordinationPolicy::Independent => Self::Independent,
            PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilRequestCompletion => {
                Self::ExactInputSingleFlightUntilRequestCompletion
            }
            PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilPrefixPublished => {
                Self::ExactInputSingleFlightUntilPrefixPublished
            }
        }
    }
}

impl fmt::Display for PrefixCacheCoordinationPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Independent => "independent",
            Self::ExactInputSingleFlightUntilRequestCompletion => {
                "exact-input-single-flight-until-request-completion"
            }
            Self::ExactInputSingleFlightUntilPrefixPublished => {
                "exact-input-single-flight-until-prefix-published"
            }
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum InferenceDevice {
    Cpu,
    Gpu,
}

impl InferenceDevice {
    pub fn cli_value(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

impl fmt::Display for InferenceDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("CPU"),
            Self::Gpu => formatter.write_str("GPU"),
        }
    }
}

impl FromStr for InferenceDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            unsupported => Err(format!("unsupported inference device: {unsupported}")),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KvCacheType {
    Contiguous,
    Paged {
        per_page_token_count: usize,
        enable_prefix_caching: bool,
    },
}

impl fmt::Display for KvCacheType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contiguous => formatter.write_str("contiguous"),
            Self::Paged {
                per_page_token_count,
                enable_prefix_caching,
            } => {
                let name = if *enable_prefix_caching {
                    "paged-prefix"
                } else {
                    "paged"
                };
                write!(formatter, "{name}:{per_page_token_count}")
            }
        }
    }
}

impl FromStr for KvCacheType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let (cache_type, per_page_token_count) = match value.split_once(':') {
            Some((cache_type, value)) => {
                let per_page_token_count = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid KV-cache page token count: {value}"))?;
                if per_page_token_count == 0 {
                    return Err("KV-cache page token count must be greater than zero".to_owned());
                }
                (cache_type, per_page_token_count)
            }
            None => (value, DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT),
        };
        match cache_type {
            "contiguous" if value.contains(':') => {
                Err("contiguous KV cache does not accept a page token count".to_owned())
            }
            "contiguous" => Ok(Self::Contiguous),
            "paged" => Ok(Self::Paged {
                per_page_token_count,
                enable_prefix_caching: false,
            }),
            "paged-prefix" => Ok(Self::Paged {
                per_page_token_count,
                enable_prefix_caching: true,
            }),
            unsupported => Err(format!(
                "unsupported KV cache implementation: {unsupported}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_activation_dtypes() {
        assert_eq!("f32".parse(), Ok(ActivationDType::F32));
        assert_eq!("f16".parse(), Ok(ActivationDType::F16));
        assert!("bf16".parse::<ActivationDType>().is_err());
        assert_eq!(ActivationDType::F32.to_string(), "f32");
        assert_eq!(ActivationDType::F16.to_string(), "f16");
        assert_eq!(DType::from(ActivationDType::F32), DType::F32);
        assert_eq!(DType::from(ActivationDType::F16), DType::F16);
    }

    #[test]
    fn parses_inference_devices() {
        assert_eq!("cpu".parse(), Ok(InferenceDevice::Cpu));
        assert_eq!("gpu".parse(), Ok(InferenceDevice::Gpu));
        assert!("mixed".parse::<InferenceDevice>().is_err());
        assert!("tpu".parse::<InferenceDevice>().is_err());
        assert_eq!(InferenceDevice::Cpu.to_string(), "CPU");
        assert_eq!(InferenceDevice::Gpu.to_string(), "GPU");
    }

    #[test]
    fn round_trips_request_orchestrator_configuration_through_proto() {
        for (routing_policy, expected_display) in [
            (RequestRoutingPolicy::RoundRobin, "round-robin"),
            (RequestRoutingPolicy::LeastRequests, "least-requests"),
            (
                RequestRoutingPolicy::LeastEstimatedWork,
                "least-estimated-work",
            ),
            (RequestRoutingPolicy::MostCachedPrefix, "most-cached-prefix"),
            (
                RequestRoutingPolicy::CacheAwareLeastEstimatedWork,
                "cache-aware-least-estimated-work",
            ),
        ] {
            let config = RequestOrchestratorConfig {
                routing_policy,
                throughput_profiles: vec![BackendThroughputProfile {
                    input_token_count: 128,
                    max_output_token_count: 64,
                    cpu: BackendThroughput {
                        prefill_tokens_per_second: 300,
                        decode_tokens_per_second: 200,
                    },
                    gpu: BackendThroughput {
                        prefill_tokens_per_second: 400,
                        decode_tokens_per_second: 250,
                    },
                }],
            };

            assert_eq!(
                config.to_string(),
                format!(
                    "routing_policy={expected_display}, \
                     throughput_profile=(input_token_count=128, max_output_token_count=64, \
                     cpu_prefill_tokens_per_second=300, cpu_decode_tokens_per_second=200, \
                     gpu_prefill_tokens_per_second=400, gpu_decode_tokens_per_second=250)"
                )
            );
            assert_eq!(
                RequestOrchestratorConfig::from(RequestOrchestratorConfigProto::from(
                    config.clone(),
                )),
                config
            );
        }
    }

    #[test]
    fn parses_scheduling_policies() {
        assert_eq!(
            "first-come-first-served".parse(),
            Ok(SchedulingPolicy::FirstComeFirstServed)
        );
        assert_eq!(
            "shortest-prefill-first".parse(),
            Ok(SchedulingPolicy::ShortestPrefillFirst)
        );
        assert_eq!("round-robin".parse(), Ok(SchedulingPolicy::RoundRobin));
        assert!("unknown".parse::<SchedulingPolicy>().is_err());
    }

    #[test]
    fn displays_scheduler_configuration() {
        let config = SchedulerConfig {
            max_batched_token_count: 512,
            max_active_request_count: 4,
            scheduling_policy: SchedulingPolicy::FirstComeFirstServed,
            prefix_cache_coordination_policy: PrefixCacheCoordinationPolicy::Independent,
        };

        assert_eq!(
            config.to_string(),
            concat!(
                "max_batched_tokens=512 max_active_requests=4 ",
                "scheduling_policy=first-come-first-served ",
                "prefix_cache_coordination_policy=independent",
            )
        );
    }

    #[test]
    fn round_trips_scheduler_configuration_through_proto() {
        let config = SchedulerConfig {
            max_batched_token_count: 1_024,
            max_active_request_count: 8,
            scheduling_policy: SchedulingPolicy::ShortestPrefillFirst,
            prefix_cache_coordination_policy:
                PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilRequestCompletion,
        };

        assert_eq!(
            SchedulerConfig::from(SchedulerConfigProto::from(config)),
            config
        );
    }

    #[test]
    fn round_trips_kv_cache_configuration_through_proto() {
        let config = KvCacheConfig {
            kv_cache_type: KvCacheType::Paged {
                per_page_token_count: 32,
                enable_prefix_caching: true,
            },
            target_kv_cache_size_bytes: 2 * 1024 * 1024 * 1024,
            draft_kv_cache_size_bytes: Some(512 * 1024 * 1024),
            prefix_cache_publication_policy: PrefixCachePublicationPolicy::Incremental,
        };

        assert_eq!(
            KvCacheConfig::from(KvCacheConfigProto::from(config)),
            config
        );
    }

    #[test]
    fn displays_kv_cache_configuration() {
        let config = KvCacheConfig {
            kv_cache_type: KvCacheType::Contiguous,
            target_kv_cache_size_bytes: 2 * 1024 * 1024 * 1024,
            draft_kv_cache_size_bytes: None,
            prefix_cache_publication_policy: PrefixCachePublicationPolicy::OnRequestCompletion,
        };

        assert_eq!(
            config.to_string(),
            "kv_cache_type=contiguous target_kv_cache_size_bytes=2147483648 \
             prefix_cache_publication_policy=on-request-completion \
             draft_kv_cache_size_bytes=inferred"
        );
    }

    #[test]
    fn parses_kv_cache_types() {
        assert_eq!("contiguous".parse(), Ok(KvCacheType::Contiguous));
        assert_eq!(
            "paged".parse(),
            Ok(KvCacheType::Paged {
                per_page_token_count: DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT,
                enable_prefix_caching: false,
            })
        );
        assert_eq!(
            "paged-prefix:32".parse(),
            Ok(KvCacheType::Paged {
                per_page_token_count: 32,
                enable_prefix_caching: true,
            })
        );
        assert!("paged:0".parse::<KvCacheType>().is_err());
        assert!("paged:invalid".parse::<KvCacheType>().is_err());
        assert!("contiguous:32".parse::<KvCacheType>().is_err());
    }

    #[test]
    fn displays_paged_prefix_caching_state() {
        assert_eq!(
            KvCacheType::Paged {
                per_page_token_count: 32,
                enable_prefix_caching: false,
            }
            .to_string(),
            "paged:32"
        );
        assert_eq!(
            KvCacheType::Paged {
                per_page_token_count: 32,
                enable_prefix_caching: true,
            }
            .to_string(),
            "paged-prefix:32"
        );
    }
}
