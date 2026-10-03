use super::textproto::parse_textproto;
use anyhow::Result as AnyhowResult;
use std::fmt;
use std::str::FromStr;
use thousands::Separable;

include!(concat!(env!("OUT_DIR"), "/inference_config.rs"));

use draft_token_count_policy::Policy;

pub(crate) const DEFAULT_SCHEDULING_POLICY: SchedulingPolicy =
    SchedulingPolicy::FirstComeFirstServed;
pub(crate) const DEFAULT_KV_CACHE_TYPE: KvCacheType = KvCacheType::Contiguous;

impl ModelConfig {
    pub(crate) fn validate(&self) -> AnyhowResult<()> {
        anyhow::ensure!(!self.model_id.is_empty(), "model_id must not be empty");
        anyhow::ensure!(
            !self.model_filename.is_empty(),
            "model_filename must not be empty"
        );
        anyhow::ensure!(
            !self.tokenizer_id.is_empty(),
            "tokenizer_id must not be empty"
        );
        anyhow::ensure!(
            self.model_filename.to_ascii_lowercase().ends_with(".gguf"),
            "model filename must identify a .gguf file"
        );
        Ok(())
    }
}

impl FromStr for ModelConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.ModelConfig")
    }
}

impl DraftModelConfig {
    pub(crate) fn validate(&self) -> AnyhowResult<()> {
        self.model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft model configuration is missing a model"))?
            .validate()?;
        self.token_count_policy
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("draft model configuration is missing a token-count policy")
            })?
            .validate()?;
        Ok(())
    }
}

impl FromStr for DraftModelConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.DraftModelConfig")
    }
}

impl fmt::Display for DraftModelConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let model = self.model.as_ref().ok_or(fmt::Error)?;
        let token_count_policy = self.token_count_policy.as_ref().ok_or(fmt::Error)?;
        writeln!(formatter, "Draft model: {}", model.model_id)?;
        writeln!(formatter, "Draft GGUF file: {}", model.model_filename)?;
        writeln!(formatter, "Draft tokenizer: {}", model.tokenizer_id)?;
        writeln!(formatter, "Draft revision: {}", model.model_revision)?;
        writeln!(formatter, "Draft token count policy: {token_count_policy}")
    }
}

impl DraftModelRunnerConfig {
    pub(crate) fn validate(&self) -> AnyhowResult<()> {
        anyhow::ensure!(
            !self.model_path.is_empty(),
            "draft model runner configuration requires a model path"
        );
        self.token_count_policy
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("draft model runner configuration requires a token-count policy")
            })?
            .validate()
    }
}

impl FromStr for DraftModelRunnerConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.DraftModelRunnerConfig")
    }
}

impl DraftTokenCountPolicy {
    pub(crate) fn validate(&self) -> AnyhowResult<()> {
        match self
            .policy
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft token-count policy is missing"))?
        {
            Policy::Fixed(policy) => validate_draft_token_count(policy.draft_token_count),
            Policy::AcceptanceRate(policy) => validate_acceptance_rate_policy(policy),
            Policy::AcceptedLength(policy) => validate_accepted_length_policy(policy),
        }
    }
}

impl fmt::Display for DraftTokenCountPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.policy.as_ref().ok_or(fmt::Error)? {
            Policy::Fixed(policy) => write!(formatter, "fixed ({})", policy.draft_token_count),
            Policy::AcceptanceRate(policy) => write!(
                formatter,
                "acceptance rate (initial: {}, bounds: [{}, {}], thresholds: [{}, {}])",
                policy.initial_draft_token_count,
                policy.minimum_draft_token_count,
                policy.maximum_draft_token_count,
                policy.decrease_threshold,
                policy.increase_threshold,
            ),
            Policy::AcceptedLength(policy) => write!(
                formatter,
                "accepted length (initial: {}, bounds: [{}, {}], smoothing factor: {})",
                policy.initial_draft_token_count,
                policy.minimum_draft_token_count,
                policy.maximum_draft_token_count,
                policy.smoothing_factor,
            ),
        }
    }
}

impl KvCacheType {
    fn cli_value(self) -> &'static str {
        match self {
            Self::Contiguous => "contiguous",
            Self::Paged => "paged",
            Self::PagedPrefix => "paged-prefix",
        }
    }
}

impl FromStr for KvCacheConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.KvCacheConfig")
    }
}

impl KvCacheConfig {
    pub(crate) fn validate(&self) -> AnyhowResult<()> {
        anyhow::ensure!(
            self.target_kv_cache_size_bytes > 0,
            "target KV-cache size must be greater than zero"
        );
        usize::try_from(self.target_kv_cache_size_bytes)
            .map_err(|_| anyhow::anyhow!("target KV-cache size does not fit in usize"))?;
        let kv_cache_type = KvCacheType::try_from(self.kv_cache_type)
            .map_err(|_| anyhow::anyhow!("unsupported KV-cache type {}", self.kv_cache_type))?;
        if kv_cache_type != KvCacheType::Contiguous {
            anyhow::ensure!(
                self.per_page_token_count > 0,
                "KV-cache page token count must be greater than zero"
            );
        }
        Ok(())
    }
}

impl fmt::Display for KvCacheConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kv_cache_type = KvCacheType::try_from(self.kv_cache_type).map_err(|_| fmt::Error)?;
        if kv_cache_type == KvCacheType::Contiguous {
            writeln!(formatter, "KV cache type: {}", kv_cache_type.cli_value())?;
        } else {
            writeln!(
                formatter,
                "KV cache type: {}:{}",
                kv_cache_type.cli_value(),
                self.per_page_token_count
            )?;
        }
        writeln!(
            formatter,
            "Target KV cache size: {} bytes",
            self.target_kv_cache_size_bytes.separate_with_commas()
        )?;
        match self.draft_kv_cache_size_bytes {
            0 => writeln!(
                formatter,
                "Draft KV cache size: inferred from target token capacity"
            ),
            size_bytes => writeln!(
                formatter,
                "Draft KV cache size: {} bytes",
                size_bytes.separate_with_commas()
            ),
        }
    }
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

impl FromStr for SchedulerConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.SchedulerConfig")
    }
}

impl fmt::Display for SchedulerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            formatter,
            "Maximum batched token count: {}",
            self.max_batched_token_count.separate_with_commas()
        )?;
        writeln!(
            formatter,
            "Maximum active request count: {}",
            self.max_active_request_count.separate_with_commas()
        )?;
        let scheduling_policy =
            SchedulingPolicy::try_from(self.scheduling_policy).unwrap_or(DEFAULT_SCHEDULING_POLICY);
        writeln!(
            formatter,
            "Scheduling policy: {}",
            scheduling_policy.cli_value()
        )
    }
}

fn validate_draft_token_count(count: u64) -> AnyhowResult<()> {
    anyhow::ensure!(
        count > 0,
        "draft token count {count} must be greater than zero"
    );
    usize::try_from(count)
        .map(|_| ())
        .map_err(|_| anyhow::anyhow!("draft token count {count} does not fit in usize"))
}

fn validate_acceptance_rate_policy(
    policy: &AcceptanceRateDraftTokenCountPolicy,
) -> AnyhowResult<()> {
    validate_adaptive_draft_token_counts(
        policy.initial_draft_token_count,
        policy.minimum_draft_token_count,
        policy.maximum_draft_token_count,
    )?;
    anyhow::ensure!(
        policy.decrease_threshold.is_finite() && (0.0..=1.0).contains(&policy.decrease_threshold),
        "decrease threshold {} must be between zero and one",
        policy.decrease_threshold
    );
    anyhow::ensure!(
        policy.increase_threshold.is_finite() && (0.0..=1.0).contains(&policy.increase_threshold),
        "increase threshold {} must be between zero and one",
        policy.increase_threshold
    );
    anyhow::ensure!(
        policy.decrease_threshold < policy.increase_threshold,
        "decrease threshold {} must be less than increase threshold {}",
        policy.decrease_threshold,
        policy.increase_threshold
    );
    Ok(())
}

fn validate_accepted_length_policy(
    policy: &AcceptedLengthDraftTokenCountPolicy,
) -> AnyhowResult<()> {
    validate_adaptive_draft_token_counts(
        policy.initial_draft_token_count,
        policy.minimum_draft_token_count,
        policy.maximum_draft_token_count,
    )?;
    anyhow::ensure!(
        policy.smoothing_factor.is_finite()
            && policy.smoothing_factor > 0.0
            && policy.smoothing_factor <= 1.0,
        "smoothing factor {} must be greater than zero and at most one",
        policy.smoothing_factor
    );
    Ok(())
}

fn validate_adaptive_draft_token_counts(
    initial: u64,
    minimum: u64,
    maximum: u64,
) -> AnyhowResult<()> {
    anyhow::ensure!(
        minimum > 0,
        "minimum draft token count {minimum} must be greater than zero"
    );
    anyhow::ensure!(
        minimum <= initial && initial <= maximum,
        "initial draft token count {initial} must be within bounds [{minimum}, {maximum}]"
    );
    for (name, count) in [
        ("initial", initial),
        ("minimum", minimum),
        ("maximum", maximum),
    ] {
        usize::try_from(count).map_err(|_| {
            anyhow::anyhow!("{name} draft token count {count} does not fit in usize")
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_scheduler_configuration() {
        let config = SchedulerConfig {
            max_batched_token_count: 1_024,
            max_active_request_count: 8,
            scheduling_policy: SchedulingPolicy::ShortestPrefillFirst.into(),
        };

        assert_eq!(
            config.to_string(),
            "Maximum batched token count: 1,024\n\
             Maximum active request count: 8\n\
             Scheduling policy: shortest-prefill-first\n"
        );
    }

    #[test]
    fn displays_kv_cache_configuration() {
        let config = KvCacheConfig {
            kv_cache_type: KvCacheType::PagedPrefix.into(),
            per_page_token_count: 32,
            target_kv_cache_size_bytes: 2_147_483_648,
            draft_kv_cache_size_bytes: 536_870_912,
        };

        assert_eq!(
            config.to_string(),
            "KV cache type: paged-prefix:32\n\
             Target KV cache size: 2,147,483,648 bytes\n\
             Draft KV cache size: 536,870,912 bytes\n"
        );
    }

    #[test]
    fn treats_zero_draft_kv_cache_size_as_unspecified() {
        let config = KvCacheConfig {
            kv_cache_type: KvCacheType::Contiguous.into(),
            per_page_token_count: 0,
            target_kv_cache_size_bytes: 1024,
            draft_kv_cache_size_bytes: 0,
        };

        assert!(config.validate().is_ok());
        assert!(config
            .to_string()
            .contains("Draft KV cache size: inferred from target token capacity"));
    }

    #[test]
    fn rejects_an_unnormalized_paged_kv_cache_configuration() {
        let config = KvCacheConfig {
            kv_cache_type: KvCacheType::Paged.into(),
            per_page_token_count: 0,
            target_kv_cache_size_bytes: 1024,
            draft_kv_cache_size_bytes: 0,
        };

        assert_eq!(
            config.validate().unwrap_err().to_string(),
            "KV-cache page token count must be greater than zero"
        );
    }

    fn fixed_policy(draft_token_count: u64) -> DraftTokenCountPolicy {
        DraftTokenCountPolicy {
            policy: Some(Policy::Fixed(FixedDraftTokenCountPolicy {
                draft_token_count,
            })),
        }
    }

    #[test]
    fn resolves_a_fixed_draft_token_count() {
        let policy = fixed_policy(4);
        assert!(policy.validate().is_ok());
        assert_eq!(policy.to_string(), "fixed (4)");
    }

    #[test]
    fn rejects_missing_and_zero_draft_token_counts() {
        assert!(DraftTokenCountPolicy::default().validate().is_err());
        assert_eq!(
            fixed_policy(0).validate().unwrap_err().to_string(),
            "draft token count 0 must be greater than zero"
        );
    }

    fn acceptance_rate_policy() -> DraftTokenCountPolicy {
        DraftTokenCountPolicy {
            policy: Some(Policy::AcceptanceRate(
                AcceptanceRateDraftTokenCountPolicy {
                    initial_draft_token_count: 4,
                    decrease_threshold: 0.4,
                    increase_threshold: 0.8,
                    minimum_draft_token_count: 1,
                    maximum_draft_token_count: 8,
                },
            )),
        }
    }

    #[test]
    fn validates_an_acceptance_rate_policy() {
        let policy = acceptance_rate_policy();
        assert!(policy.validate().is_ok());
        assert_eq!(
            policy.to_string(),
            "acceptance rate (initial: 4, bounds: [1, 8], thresholds: [0.4, 0.8])"
        );
    }

    #[test]
    fn rejects_invalid_acceptance_rate_policies() {
        let mut policy = acceptance_rate_policy();
        let Policy::AcceptanceRate(config) = policy.policy.as_mut().unwrap() else {
            unreachable!()
        };
        config.initial_draft_token_count = 9;
        assert_eq!(
            policy.validate().unwrap_err().to_string(),
            "initial draft token count 9 must be within bounds [1, 8]"
        );

        let mut policy = acceptance_rate_policy();
        let Policy::AcceptanceRate(config) = policy.policy.as_mut().unwrap() else {
            unreachable!()
        };
        config.decrease_threshold = 0.9;
        assert_eq!(
            policy.validate().unwrap_err().to_string(),
            "decrease threshold 0.9 must be less than increase threshold 0.8"
        );
    }

    fn accepted_length_policy() -> DraftTokenCountPolicy {
        DraftTokenCountPolicy {
            policy: Some(Policy::AcceptedLength(
                AcceptedLengthDraftTokenCountPolicy {
                    initial_draft_token_count: 4,
                    smoothing_factor: 0.2,
                    minimum_draft_token_count: 1,
                    maximum_draft_token_count: 12,
                },
            )),
        }
    }

    #[test]
    fn validates_an_accepted_length_policy() {
        let policy = accepted_length_policy();
        assert!(policy.validate().is_ok());
        assert_eq!(
            policy.to_string(),
            "accepted length (initial: 4, bounds: [1, 12], smoothing factor: 0.2)"
        );
    }

    #[test]
    fn rejects_invalid_accepted_length_policies() {
        let mut policy = accepted_length_policy();
        let Policy::AcceptedLength(config) = policy.policy.as_mut().unwrap() else {
            unreachable!()
        };
        config.smoothing_factor = 0.0;
        assert_eq!(
            policy.validate().unwrap_err().to_string(),
            "smoothing factor 0 must be greater than zero and at most one"
        );

        let mut policy = accepted_length_policy();
        let Policy::AcceptedLength(config) = policy.policy.as_mut().unwrap() else {
            unreachable!()
        };
        config.maximum_draft_token_count = 3;
        assert_eq!(
            policy.validate().unwrap_err().to_string(),
            "initial draft token count 4 must be within bounds [1, 3]"
        );
    }
}
