pub(crate) mod client;
pub(crate) mod server;

use std::fmt;
use std::str::FromStr;

#[derive(Clone, Copy)]
pub(crate) enum KvCacheType {
    Contiguous,
}

impl fmt::Display for KvCacheType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contiguous => formatter.write_str("contiguous"),
        }
    }
}

impl FromStr for KvCacheType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "contiguous" => Ok(Self::Contiguous),
            unsupported => Err(format!(
                "unsupported KV cache implementation: {unsupported}"
            )),
        }
    }
}
