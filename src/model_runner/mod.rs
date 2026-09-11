pub(crate) mod client;
pub(crate) mod server;

use std::fmt;
use std::str::FromStr;

pub(crate) const DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SchedulingPolicy {
    FirstComeFirstServed,
    ShortestPrefillFirst,
}

impl SchedulingPolicy {
    fn cli_value(self) -> &'static str {
        match self {
            Self::FirstComeFirstServed => "first-come-first-served",
            Self::ShortestPrefillFirst => "shortest-prefill-first",
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
}

impl InferenceDevice {
    fn cli_value(self) -> &'static str {
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
    fn parses_inference_devices() {
        assert_eq!("cpu".parse(), Ok(InferenceDevice::Cpu));
        assert_eq!("gpu".parse(), Ok(InferenceDevice::Gpu));
        assert!("mixed".parse::<InferenceDevice>().is_err());
        assert_eq!(InferenceDevice::Cpu.to_string(), "CPU");
        assert_eq!(InferenceDevice::Gpu.to_string(), "GPU");
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
