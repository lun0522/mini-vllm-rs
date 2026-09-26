use crate::proto::inference_config::draft_token_count_policy::Policy;
use crate::proto::inference_config::DraftTokenCountPolicy as DraftTokenCountPolicyProto;
use anyhow::Result;

#[derive(Clone, Copy)]
pub(super) enum DraftTokenCountPolicy {
    Fixed {
        draft_token_count: usize,
    },
    AcceptanceRate {
        initial_draft_token_count: usize,
        decrease_threshold: f32,
        increase_threshold: f32,
        minimum_draft_token_count: usize,
        maximum_draft_token_count: usize,
    },
}

impl TryFrom<&DraftTokenCountPolicyProto> for DraftTokenCountPolicy {
    type Error = anyhow::Error;

    fn try_from(policy: &DraftTokenCountPolicyProto) -> Result<Self> {
        policy.validate()?;
        match policy
            .policy
            .as_ref()
            .expect("validated draft token-count policy should select a policy")
        {
            Policy::Fixed(policy) => Ok(Self::Fixed {
                draft_token_count: usize::try_from(policy.draft_token_count)
                    .expect("validated draft token count should fit in usize"),
            }),
            Policy::AcceptanceRate(policy) => Ok(Self::AcceptanceRate {
                initial_draft_token_count: usize::try_from(policy.initial_draft_token_count)
                    .expect("validated initial draft token count should fit in usize"),
                decrease_threshold: policy.decrease_threshold,
                increase_threshold: policy.increase_threshold,
                minimum_draft_token_count: usize::try_from(policy.minimum_draft_token_count)
                    .expect("validated minimum draft token count should fit in usize"),
                maximum_draft_token_count: usize::try_from(policy.maximum_draft_token_count)
                    .expect("validated maximum draft token count should fit in usize"),
            }),
        }
    }
}

pub(super) struct DraftTokenCountController {
    current_draft_token_count: usize,
    policy: DraftTokenCountPolicy,
}

impl DraftTokenCountController {
    pub(super) fn new(policy: DraftTokenCountPolicy) -> Self {
        let current_draft_token_count = match policy {
            DraftTokenCountPolicy::Fixed { draft_token_count } => draft_token_count,
            DraftTokenCountPolicy::AcceptanceRate {
                initial_draft_token_count,
                ..
            } => initial_draft_token_count,
        };
        Self {
            current_draft_token_count,
            policy,
        }
    }

    pub(super) fn draft_token_count(&self) -> usize {
        self.current_draft_token_count
    }

    pub(super) fn update(
        &mut self,
        accepted_token_count: usize,
        proposed_token_count: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            accepted_token_count <= proposed_token_count,
            "accepted draft token count {accepted_token_count} cannot exceed proposed draft token \
            count {proposed_token_count}"
        );
        if proposed_token_count == 0 {
            return Ok(());
        }

        let DraftTokenCountPolicy::AcceptanceRate {
            decrease_threshold,
            increase_threshold,
            minimum_draft_token_count,
            maximum_draft_token_count,
            ..
        } = self.policy
        else {
            return Ok(());
        };
        let acceptance_rate = accepted_token_count as f32 / proposed_token_count as f32;
        if acceptance_rate < decrease_threshold {
            self.current_draft_token_count = self
                .current_draft_token_count
                .saturating_sub(1)
                .max(minimum_draft_token_count);
        } else if acceptance_rate > increase_threshold {
            self.current_draft_token_count = self
                .current_draft_token_count
                .saturating_add(1)
                .min(maximum_draft_token_count);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::inference_config::AcceptanceRateDraftTokenCountPolicy;

    fn acceptance_rate_policy() -> DraftTokenCountPolicy {
        DraftTokenCountPolicy::try_from(&DraftTokenCountPolicyProto {
            policy: Some(Policy::AcceptanceRate(
                AcceptanceRateDraftTokenCountPolicy {
                    initial_draft_token_count: 4,
                    decrease_threshold: 0.4,
                    increase_threshold: 0.8,
                    minimum_draft_token_count: 1,
                    maximum_draft_token_count: 8,
                },
            )),
        })
        .unwrap()
    }

    #[test]
    fn keeps_a_fixed_draft_token_count() {
        let mut controller = DraftTokenCountController::new(DraftTokenCountPolicy::Fixed {
            draft_token_count: 4,
        });

        controller.update(0, 4).unwrap();
        assert_eq!(controller.draft_token_count(), 4);
    }

    #[test]
    fn adjusts_an_acceptance_rate_policy_within_its_bounds() {
        let mut controller = DraftTokenCountController::new(acceptance_rate_policy());

        controller.update(1, 4).unwrap();
        assert_eq!(controller.draft_token_count(), 3);
        controller.update(0, 3).unwrap();
        controller.update(0, 2).unwrap();
        controller.update(0, 1).unwrap();
        assert_eq!(controller.draft_token_count(), 1);

        for _ in 0..10 {
            controller.update(1, 1).unwrap();
        }
        assert_eq!(controller.draft_token_count(), 8);
    }

    #[test]
    fn leaves_the_count_unchanged_at_or_between_thresholds() {
        let mut controller = DraftTokenCountController::new(acceptance_rate_policy());

        controller.update(2, 5).unwrap();
        assert_eq!(controller.draft_token_count(), 4);
        controller.update(3, 5).unwrap();
        assert_eq!(controller.draft_token_count(), 4);
        controller.update(4, 5).unwrap();
        assert_eq!(controller.draft_token_count(), 4);
    }

    #[test]
    fn rejects_more_accepted_tokens_than_proposed_tokens() {
        let mut controller = DraftTokenCountController::new(acceptance_rate_policy());

        assert_eq!(
            controller.update(2, 1).unwrap_err().to_string(),
            "accepted draft token count 2 cannot exceed proposed draft token count 1"
        );
        assert_eq!(controller.draft_token_count(), 4);
    }
}
