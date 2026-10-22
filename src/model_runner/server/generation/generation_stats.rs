use crate::proto::model_runner::TextGenerationFinishReason;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationFinishReason {
    EndOfSequenceToken,
    MaxNewTokensReached,
}

impl From<GenerationFinishReason> for TextGenerationFinishReason {
    fn from(finish_reason: GenerationFinishReason) -> Self {
        match finish_reason {
            GenerationFinishReason::EndOfSequenceToken => Self::EndOfSequenceToken,
            GenerationFinishReason::MaxNewTokensReached => Self::MaxNewTokensReached,
        }
    }
}

impl fmt::Display for GenerationFinishReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EndOfSequenceToken => formatter.write_str("end_of_sequence_token"),
            Self::MaxNewTokensReached => formatter.write_str("max_new_tokens_reached"),
        }
    }
}

pub struct TextGenerationStats {
    pub execution: GenerationExecutionStats,
    pub prefix_cache: PrefixCacheGenerationStats,
}

pub struct PrefixCacheGenerationStats {
    pub target: PrefixCacheModelStats,
    pub draft: Option<PrefixCacheModelStats>,
}

pub struct PrefixCacheModelStats {
    pub restored_token_count: usize,
    pub newly_indexed_token_count: usize,
}

pub struct GenerationExecutionStats {
    pub finish_reason: GenerationFinishReason,
    pub input_token_count: u64,
    pub output_token_count: u64,
    pub draft_stats: Option<DraftGenerationStats>,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
}

pub struct DraftGenerationStats {
    pub accepted_token_count: usize,
    pub proposed_token_count: usize,
    pub selected_token_count_histogram: BTreeMap<usize, usize>,
}

pub struct PrefixTokenCounts {
    pub target: usize,
    pub draft: Option<usize>,
}
