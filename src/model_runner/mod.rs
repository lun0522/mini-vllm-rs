pub(crate) mod client;
pub(crate) mod server;

use std::fmt;
use std::str::FromStr;

use crate::proto::inference_config::KvCacheConfig as KvCacheConfigProto;
use crate::proto::inference_config::KvCacheType as KvCacheTypeProto;
use crate::proto::inference_config::SchedulerConfig as SchedulerConfigProto;
use crate::proto::inference_config::SchedulingPolicy as SchedulingPolicyProto;
use crate::proto::inference_config::DEFAULT_SCHEDULING_POLICY;
use candle_core::DType;

pub(crate) const DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KvCacheConfig {
    pub(crate) kv_cache_type: KvCacheType,
    pub(crate) target_kv_cache_size_bytes: usize,
    pub(crate) draft_kv_cache_size_bytes: Option<usize>,
}

impl From<KvCacheConfigProto> for KvCacheConfig {
    fn from(config: KvCacheConfigProto) -> Self {
        let kv_cache_type = match KvCacheTypeProto::try_from(config.kv_cache_type)
            .expect("validated KV-cache configuration should contain a supported cache type")
        {
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
        Self {
            kv_cache_type,
            target_kv_cache_size_bytes: config.target_kv_cache_size_bytes as usize,
            draft_kv_cache_size_bytes: (config.draft_kv_cache_size_bytes != 0)
                .then_some(config.draft_kv_cache_size_bytes as usize),
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
        }
    }
}

impl fmt::Display for KvCacheConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "kv_cache_type={} target_kv_cache_size_bytes={}",
            self.kv_cache_type, self.target_kv_cache_size_bytes
        )?;
        if let Some(size_bytes) = self.draft_kv_cache_size_bytes {
            write!(formatter, " draft_kv_cache_size_bytes={size_bytes}")
        } else {
            formatter.write_str(" draft_kv_cache_size_bytes=inferred")
        }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerConfig {
    pub(crate) max_batched_token_count: usize,
    pub(crate) max_active_request_count: usize,
    pub(crate) scheduling_policy: SchedulingPolicy,
}

impl From<SchedulerConfigProto> for SchedulerConfig {
    fn from(config: SchedulerConfigProto) -> Self {
        let scheduling_policy = match SchedulingPolicyProto::try_from(config.scheduling_policy)
            .unwrap_or(DEFAULT_SCHEDULING_POLICY)
        {
            SchedulingPolicyProto::FirstComeFirstServed => SchedulingPolicy::FirstComeFirstServed,
            SchedulingPolicyProto::ShortestPrefillFirst => SchedulingPolicy::ShortestPrefillFirst,
            SchedulingPolicyProto::RoundRobin => SchedulingPolicy::RoundRobin,
        };
        Self {
            max_batched_token_count: config.max_batched_token_count as usize,
            max_active_request_count: config.max_active_request_count as usize,
            scheduling_policy,
        }
    }
}

impl From<SchedulerConfig> for SchedulerConfigProto {
    fn from(config: SchedulerConfig) -> Self {
        let scheduling_policy = match config.scheduling_policy {
            SchedulingPolicy::FirstComeFirstServed => SchedulingPolicyProto::FirstComeFirstServed,
            SchedulingPolicy::ShortestPrefillFirst => SchedulingPolicyProto::ShortestPrefillFirst,
            SchedulingPolicy::RoundRobin => SchedulingPolicyProto::RoundRobin,
        };
        Self {
            max_batched_token_count: config.max_batched_token_count as u32,
            max_active_request_count: config.max_active_request_count as u32,
            scheduling_policy: scheduling_policy.into(),
        }
    }
}

impl fmt::Display for SchedulerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "max_batched_tokens={} max_active_requests={} scheduling_policy={}",
            self.max_batched_token_count, self.max_active_request_count, self.scheduling_policy,
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InferenceDevice {
    Cpu,
    Gpu,
    Mixed,
}

impl InferenceDevice {
    fn cli_value(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Mixed => "mixed",
        }
    }
}

impl fmt::Display for InferenceDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("CPU"),
            Self::Gpu => formatter.write_str("GPU"),
            Self::Mixed => formatter.write_str("Mixed"),
        }
    }
}

impl FromStr for InferenceDevice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "mixed" => Ok(Self::Mixed),
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
        assert_eq!("mixed".parse(), Ok(InferenceDevice::Mixed));
        assert!("tpu".parse::<InferenceDevice>().is_err());
        assert_eq!(InferenceDevice::Cpu.to_string(), "CPU");
        assert_eq!(InferenceDevice::Gpu.to_string(), "GPU");
        assert_eq!(InferenceDevice::Mixed.to_string(), "Mixed");
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
        };

        assert_eq!(
            config.to_string(),
            "max_batched_tokens=512 max_active_requests=4 scheduling_policy=first-come-first-served"
        );
    }

    #[test]
    fn round_trips_scheduler_configuration_through_proto() {
        let config = SchedulerConfig {
            max_batched_token_count: 1_024,
            max_active_request_count: 8,
            scheduling_policy: SchedulingPolicy::ShortestPrefillFirst,
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
        };

        assert_eq!(
            config.to_string(),
            "kv_cache_type=contiguous target_kv_cache_size_bytes=2147483648 \
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
