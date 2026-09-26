use crate::proto::inference_config::draft_token_count_policy::Policy;
use crate::proto::inference_config::DraftTokenCountPolicy;
use anyhow::Result;
use std::fmt;

impl DraftTokenCountPolicy {
    pub(crate) fn validate(&self) -> Result<()> {
        match self
            .policy
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft token-count policy is missing"))?
        {
            Policy::Fixed(policy) => validate_draft_token_count(policy.draft_token_count),
            Policy::AcceptanceRate(policy) => validate_acceptance_rate_policy(policy),
        }
    }

    // TODO: Replace this with construction of a runtime token-count policy that updates after
    // each speculative verification step.
    pub(crate) fn draft_token_count(&self) -> usize {
        let count = match self
            .policy
            .as_ref()
            .expect("validated draft token-count policy should select a policy")
        {
            Policy::Fixed(policy) => policy.draft_token_count,
            Policy::AcceptanceRate(policy) => policy.initial_draft_token_count,
        };
        usize::try_from(count).expect("validated draft token count should fit in usize")
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

fn validate_draft_token_count(count: u64) -> Result<()> {
    anyhow::ensure!(count > 0, "draft token count must be greater than zero");
    usize::try_from(count)
        .map(|_| ())
        .map_err(|_| anyhow::anyhow!("draft token count {count} does not fit in usize"))
}

fn validate_acceptance_rate_policy(
    policy: &crate::proto::inference_config::AcceptanceRateDraftTokenCountPolicy,
) -> Result<()> {
    anyhow::ensure!(
        policy.minimum_draft_token_count > 0,
        "minimum draft token count must be greater than zero"
    );
    anyhow::ensure!(
        policy.minimum_draft_token_count <= policy.initial_draft_token_count
            && policy.initial_draft_token_count <= policy.maximum_draft_token_count,
        "initial draft token count must be within the configured bounds"
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
        "decrease threshold must be between zero and one"
    );
    anyhow::ensure!(
        policy.increase_threshold.is_finite() && (0.0..=1.0).contains(&policy.increase_threshold),
        "increase threshold must be between zero and one"
    );
    anyhow::ensure!(
        policy.decrease_threshold < policy.increase_threshold,
        "decrease threshold must be less than increase threshold"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::inference_config::AcceptanceRateDraftTokenCountPolicy;
    use crate::proto::inference_config::FixedDraftTokenCountPolicy;

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
        assert_eq!(policy.draft_token_count(), 4);
        assert_eq!(policy.to_string(), "fixed (4)");
    }

    #[test]
    fn rejects_missing_and_zero_draft_token_counts() {
        assert!(DraftTokenCountPolicy::default().validate().is_err());
        assert!(fixed_policy(0).validate().is_err());
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
        assert_eq!(policy.draft_token_count(), 4);
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
        assert!(policy.validate().is_err());

        let mut policy = acceptance_rate_policy();
        let Policy::AcceptanceRate(config) = policy.policy.as_mut().unwrap() else {
            unreachable!()
        };
        config.decrease_threshold = 0.9;
        assert!(policy.validate().is_err());
    }
}
