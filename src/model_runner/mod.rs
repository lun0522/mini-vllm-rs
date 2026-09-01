pub(crate) mod client;
pub(crate) mod server;

use std::fmt;
use std::str::FromStr;

pub(crate) const DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT: usize = 16;

#[derive(Clone, Copy)]
pub(crate) enum KvCacheType {
    Contiguous,
    Paged { per_page_token_count: usize },
}

impl fmt::Display for KvCacheType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contiguous => formatter.write_str("contiguous"),
            Self::Paged {
                per_page_token_count,
            } => write!(formatter, "paged({per_page_token_count})"),
        }
    }
}

impl FromStr for KvCacheType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "contiguous" => Ok(Self::Contiguous),
            "paged" => Ok(Self::Paged {
                per_page_token_count: DEFAULT_KV_CACHE_PAGE_TOKEN_COUNT,
            }),
            unsupported => Err(format!(
                "unsupported KV cache implementation: {unsupported}"
            )),
        }
    }
}
