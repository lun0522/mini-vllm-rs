use crate::proto::model_runner::TextGenerationFinishReason;
use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GenerationFinishReason {
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

pub(super) struct TextGenerationStats {
    pub(super) execution: GenerationExecutionStats,
    pub(super) prefix_cache: PrefixCacheGenerationStats,
}

pub(super) struct PrefixCacheGenerationStats {
    pub(super) target: PrefixCacheModelStats,
    pub(super) draft: Option<PrefixCacheModelStats>,
}

pub(super) struct PrefixCacheModelStats {
    pub(super) restored_token_count: usize,
    pub(super) newly_indexed_token_count: usize,
}

pub(super) struct GenerationExecutionStats {
    pub(super) finish_reason: GenerationFinishReason,
    pub(super) input_token_count: u64,
    pub(super) output_token_count: u64,
    pub(super) draft_stats: Option<DraftGenerationStats>,
    pub(super) prefill_duration: Duration,
    pub(super) decode_duration: Duration,
}

pub(super) struct DraftGenerationStats {
    pub(super) accepted_token_count: usize,
    pub(super) proposed_token_count: usize,
    pub(super) selected_token_count_histogram: BTreeMap<usize, usize>,
}

pub(super) struct PrefixTokenCounts {
    pub(super) target: usize,
    pub(super) draft: Option<usize>,
}
