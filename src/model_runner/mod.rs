pub(crate) mod client;
pub(crate) mod server;

use std::fmt;
use std::str::FromStr;

pub(crate) const DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT: usize = 16;

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
