use super::textproto::parse_textproto;
use anyhow::Result as AnyhowResult;
use std::fmt;
use std::str::FromStr;

include!(concat!(env!("OUT_DIR"), "/inference_config.rs"));

use draft_token_count_policy::Policy;

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
        }
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
    anyhow::ensure!(
        policy.minimum_draft_token_count > 0,
        "minimum draft token count {} must be greater than zero",
        policy.minimum_draft_token_count
    );
    anyhow::ensure!(
        policy.minimum_draft_token_count <= policy.initial_draft_token_count
            && policy.initial_draft_token_count <= policy.maximum_draft_token_count,
        "initial draft token count {} must be within bounds [{}, {}]",
        policy.initial_draft_token_count,
        policy.minimum_draft_token_count,
        policy.maximum_draft_token_count
    );
    for (name, count) in [
        ("initial", policy.initial_draft_token_count),
        ("minimum", policy.minimum_draft_token_count),
        ("maximum", policy.maximum_draft_token_count),
    ] {
        usize::try_from(count).map_err(|_| {
            anyhow::anyhow!("{name} draft token count {count} does not fit in usize")
        })?;
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
