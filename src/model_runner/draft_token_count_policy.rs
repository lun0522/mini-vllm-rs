use crate::proto::inference_config::draft_token_count_policy::Policy;
use crate::proto::inference_config::DraftTokenCountPolicy;
use anyhow::Result;
use std::fmt;

impl DraftTokenCountPolicy {
    pub(crate) fn draft_token_count(&self) -> Result<usize> {
        let count = match self
            .policy
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft token-count policy is missing"))?
        {
            Policy::Fixed(policy) => policy.draft_token_count,
        };
        anyhow::ensure!(count > 0, "draft token count must be greater than zero");
        usize::try_from(count)
            .map_err(|_| anyhow::anyhow!("draft token count {count} does not fit in usize"))
    }
}

impl fmt::Display for DraftTokenCountPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.policy.as_ref().ok_or(fmt::Error)? {
            Policy::Fixed(policy) => write!(formatter, "fixed ({})", policy.draft_token_count),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(policy.draft_token_count().unwrap(), 4);
        assert_eq!(policy.to_string(), "fixed (4)");
    }

    #[test]
    fn rejects_missing_and_zero_draft_token_counts() {
        assert!(DraftTokenCountPolicy::default()
            .draft_token_count()
            .is_err());
        assert!(fixed_policy(0).draft_token_count().is_err());
    }
}
