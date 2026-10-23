use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Tensor;
use candle_transformers::utils::apply_repeat_penalty;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

pub struct SampleWithProbabilities {
    pub token_id: u32,
    pub probabilities: Vec<f32>,
}

/// Sampling configuration and RNG state owned by a single request and model role.
pub struct Sampler {
    temperature: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    rng: StdRng,
}

impl Sampler {
    pub fn new(temperature: f32, seed: u64, repeat_penalty: f32, repeat_last_n: usize) -> Self {
        Self {
            temperature: f64::from(temperature),
            repeat_penalty,
            repeat_last_n,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
        committed_tokens: &[u32],
        appended_tokens: &[u32],
    ) -> Result<u32> {
        let token_id = if self.is_greedy() {
            let logits = self
                .apply_repetition_penalty(logits, committed_tokens, appended_tokens)?
                .to_dtype(DType::F32)?;
            logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?
        } else {
            self.sample_with_probabilities(logits, committed_tokens, appended_tokens)?
                .token_id
        };
        Ok(token_id)
    }

    /// Samples at positive temperature and retains the distribution for speculative verification.
    pub fn sample_with_probabilities(
        &mut self,
        logits: &Tensor,
        committed_tokens: &[u32],
        appended_tokens: &[u32],
    ) -> Result<SampleWithProbabilities> {
        let probabilities =
            self.calculate_probabilities(logits, committed_tokens, appended_tokens)?;
        let token_id = self.sample_probabilities(&probabilities)?;
        Ok(SampleWithProbabilities {
            token_id,
            probabilities,
        })
    }

    /// Returns `None` when the draft token matches the greedy target token, or
    /// `Some(target_token_id)` when it does not.
    pub fn verify_greedy_draft_token(
        &mut self,
        draft_token: u32,
        logits: &Tensor,
        committed_tokens: &[u32],
        appended_tokens: &[u32],
    ) -> Result<Option<u32>> {
        let target_token = self.sample(logits, committed_tokens, appended_tokens)?;
        Ok((target_token != draft_token).then_some(target_token))
    }

    /// Returns `None` when the draft token is accepted, or `Some(replacement_token_id)` when
    /// the draft token is rejected and a replacement is sampled from the residual distribution.
    ///
    /// Let p be the target distribution and q the draft distribution, both after temperature
    /// and repetition penalties. For a draft token x sampled from q:
    /// - Accept x with probability min(1, p(x) / q(x)), returning None.
    /// - Otherwise calculate weight(token) = max(p(token) - q(token), 0) for every
    ///   token in the vocabulary. Let total_weight be the sum of these weights.
    ///   Sample a replacement with probability weight(token) / total_weight.
    /// This makes the resulting token follow p while allowing draft tokens to be reused.
    pub fn verify_stochastic_draft_token(
        &mut self,
        draft_token: u32,
        logits: &Tensor,
        committed_tokens: &[u32],
        appended_tokens: &[u32],
        draft_probabilities: &[f32],
    ) -> Result<Option<u32>> {
        let target_probabilities =
            self.calculate_probabilities(logits, committed_tokens, appended_tokens)?;
        ensure!(
            target_probabilities.len() == draft_probabilities.len(),
            "target and draft probability distributions must have the same vocabulary size"
        );
        ensure!(
            target_probabilities.iter().all(|p| p.is_finite()),
            "target probability distribution must be finite"
        );

        // Normalize in f64: softmax's f32 output can sum to slightly more or less than one.
        let target_sum: f64 = target_probabilities.iter().map(|&p| f64::from(p)).sum();
        let draft_sum: f64 = draft_probabilities.iter().map(|&q| f64::from(q)).sum();
        // The draft token was sampled from this distribution, so its index is valid and q > 0.
        let token_id = draft_token as usize;
        let q = f64::from(draft_probabilities[token_id]) / draft_sum;
        let p = f64::from(target_probabilities[token_id]) / target_sum;
        if p >= q || self.rng.random::<f64>() < p / q {
            return Ok(None);
        }

        let residual: Vec<f64> = target_probabilities
            .iter()
            .zip(draft_probabilities)
            .map(|(&p, &q)| (f64::from(p) / target_sum - f64::from(q) / draft_sum).max(0.0))
            .collect();
        // WeightedIndex normalizes the residual. Rejection implies strictly positive mass, so
        // there is no fallback to p (which would change the output distribution).
        let distribution = WeightedIndex::new(&residual)
            .context("cannot sample the speculative residual distribution")?;
        Ok(Some(distribution.sample(&mut self.rng) as u32))
    }

    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        committed_tokens: &[u32],
        appended_tokens: &[u32],
    ) -> Result<Tensor> {
        // The repetition window can span committed context and the speculative prefix.
        let total_token_count = committed_tokens.len() + appended_tokens.len();
        let repeat_start = total_token_count.saturating_sub(self.repeat_last_n);
        let mut repeat_tokens = Vec::with_capacity(total_token_count - repeat_start);
        if repeat_start < committed_tokens.len() {
            repeat_tokens.extend_from_slice(&committed_tokens[repeat_start..]);
        }
        let appended_start = repeat_start.saturating_sub(committed_tokens.len());
        repeat_tokens.extend_from_slice(&appended_tokens[appended_start..]);
        Ok(apply_repeat_penalty(
            logits,
            self.repeat_penalty,
            &repeat_tokens,
        )?)
    }

    /// Calculates the full distribution used for positive-temperature sampling and verification.
    fn calculate_probabilities(
        &self,
        logits: &Tensor,
        committed_tokens: &[u32],
        appended_tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let logits = self
            .apply_repetition_penalty(logits, committed_tokens, appended_tokens)?
            .to_dtype(DType::F32)?;
        let scaled_logits = (&logits / self.temperature)?;
        Ok(candle_nn::ops::softmax_last_dim(&scaled_logits)?.to_vec1()?)
    }

    /// Samples normalized probabilities or unnormalized nonnegative weights.
    fn sample_probabilities(&mut self, probabilities: &[f32]) -> Result<u32> {
        let distribution = WeightedIndex::new(probabilities)
            .context("cannot sample an invalid probability distribution")?;
        Ok(distribution.sample(&mut self.rng) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_transformers::generation::LogitsProcessor;
    use candle_transformers::generation::Sampling;

    #[test]
    fn preserves_candle_sampling_sequences() -> Result<()> {
        let logits = Tensor::new(&[0.2f32, -0.3, 1.1, 0.6], &Device::Cpu)?;
        for temperature in [0.0f32, 0.00001, 0.7, 1.0, 2.0] {
            for seed in [0, 42, u64::MAX] {
                for repeat_penalty in [1.0, 1.5] {
                    let mut sampler = Sampler::new(temperature, seed, repeat_penalty, 3);
                    let sampling = if temperature == 0.0 {
                        Sampling::ArgMax
                    } else {
                        Sampling::All {
                            temperature: f64::from(temperature),
                        }
                    };
                    let mut reference = LogitsProcessor::from_sampling(seed, sampling);
                    let reference_logits =
                        apply_repeat_penalty(&logits, repeat_penalty, &[1, 3, 1])?;
                    for _ in 0..64 {
                        assert_eq!(
                            sampler.sample(&logits, &[2, 1], &[3, 1])?,
                            reference.sample(&reference_logits)?
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn samplers_have_independent_rng_state() -> Result<()> {
        let logits = Tensor::new(&[0.0f32, 0.5, 1.0], &Device::Cpu)?;
        let mut target = Sampler::new(0.7, 0, 1.0, 64);
        let mut draft = Sampler::new(0.7, 1, 1.0, 64);
        let mut reference_target = Sampler::new(0.7, 0, 1.0, 64);
        for _ in 0..32 {
            draft.sample(&logits, &[], &[])?;
            assert_eq!(
                target.sample(&logits, &[], &[])?,
                reference_target.sample(&logits, &[], &[])?
            );
        }
        Ok(())
    }

    #[test]
    fn samples_temperature_scaled_logits_using_expected_probabilities() -> Result<()> {
        let logits = Tensor::new(
            &[0.0f32, 2.0 * 2.0f32.ln(), 2.0 * 3.0f32.ln()],
            &Device::Cpu,
        )?;
        let mut sampler = Sampler::new(2.0, 0, 1.0, 64);
        let sample = sampler.sample_with_probabilities(&logits, &[], &[])?;
        for (actual, expected) in sample
            .probabilities
            .iter()
            .zip([1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn never_samples_tokens_with_zero_probability() -> Result<()> {
        let logits = Tensor::new(&[f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY], &Device::Cpu)?;
        let mut sampler = Sampler::new(1.0, 0, 1.0, 64);
        for _ in 0..32 {
            assert_eq!(sampler.sample(&logits, &[], &[])?, 1);
        }
        Ok(())
    }

    fn logits_from_weights(weights: &[f32]) -> Result<Tensor> {
        Tensor::new(
            weights.iter().map(|weight| weight.ln()).collect::<Vec<_>>(),
            &Device::Cpu,
        )
        .map_err(Into::into)
    }

    #[test]
    fn accepts_identical_distributions_and_preserves_tiny_residual_mass() -> Result<()> {
        let mut sampler = Sampler::new(1.0, 42, 1.0, 64);
        for token in [0, 1, 2] {
            assert_eq!(
                sampler.verify_stochastic_draft_token(
                    token,
                    &logits_from_weights(&[0.2, 0.3, 0.5])?,
                    &[],
                    &[],
                    &[0.2, 0.3, 0.5]
                )?,
                None
            );
        }
        assert_eq!(
            sampler.verify_stochastic_draft_token(
                0,
                &logits_from_weights(&[0.0, 1e-30, 1.0])?,
                &[],
                &[],
                &[1e-30, 0.0, 1.0]
            )?,
            Some(1)
        );
        Ok(())
    }

    #[test]
    fn speculative_sampling_recovers_the_target_distribution() -> Result<()> {
        // Includes equal models, disjoint support, overlap, zero weights, and f32 sums
        // slightly different from one. Checking acceptance also catches target-only resampling.
        for (p, q) in [
            ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
            ([0.0, 0.25, 0.75], [1.0, 0.0, 0.0]),
            ([0.6, 0.3, 0.1], [0.1, 0.3, 0.6]),
            ([0.0, 0.4, 0.6], [0.2, 0.0, 0.8]),
            ([0.1, 0.2, 0.7000001], [0.7000001, 0.2, 0.1]),
        ] {
            let mut target = Sampler::new(1.0, 42, 1.0, 64);
            let mut draft = Sampler::new(1.0, 43, 1.0, 64);
            let mut counts = [0usize; 3];
            let mut accepted = 0usize;
            const SAMPLES: usize = 40_000;
            let target_logits = logits_from_weights(&p)?;
            let draft_logits = logits_from_weights(&q)?;
            for _ in 0..SAMPLES {
                let sample = draft.sample_with_probabilities(&draft_logits, &[], &[])?;
                let replacement = target.verify_stochastic_draft_token(
                    sample.token_id,
                    &target_logits,
                    &[],
                    &[],
                    &sample.probabilities,
                )?;
                accepted += usize::from(replacement.is_none());
                counts[replacement.unwrap_or(sample.token_id) as usize] += 1;
            }
            let p_sum: f64 = p.iter().map(|&p| f64::from(p)).sum();
            let q_sum: f64 = q.iter().map(|&q| f64::from(q)).sum();
            let expected_acceptance: f64 = p
                .iter()
                .zip(q)
                .map(|(&p, q)| (f64::from(p) / p_sum).min(f64::from(q) / q_sum))
                .sum();
            assert!((accepted as f64 / SAMPLES as f64 - expected_acceptance).abs() < 0.012);
            for (count, p) in counts.into_iter().zip(p) {
                assert!((count as f64 / SAMPLES as f64 - f64::from(p) / p_sum).abs() < 0.012);
            }
        }
        Ok(())
    }

    #[test]
    fn rejects_incompatible_speculative_distributions() -> Result<()> {
        let mut sampler = Sampler::new(1.0, 0, 1.0, 64);
        for (token, p, q) in [
            (0, &[1.0, 0.0][..], &[][..]),
            (0, &[1.0, 0.0], &[1.0]),
            (0, &[0.0, 0.0], &[0.5, 0.5]),
            (0, &[f32::NAN, 1.0], &[0.5, 0.5]),
            (0, &[-1.0, 2.0], &[0.5, 0.5]),
        ] {
            let logits = logits_from_weights(p)?;
            assert!(sampler
                .verify_stochastic_draft_token(token, &logits, &[], &[], q)
                .is_err());
        }
        Ok(())
    }

    #[test]
    fn applies_repetition_penalty_across_committed_and_appended_tokens() -> Result<()> {
        let logits = Tensor::new(&[2.0f32, 4.0, -3.0, 6.0], &Device::Cpu)?;
        let penalized =
            Sampler::new(1.0, 0, 2.0, 3).apply_repetition_penalty(&logits, &[0, 1], &[2, 1])?;
        assert_eq!(penalized.to_vec1::<f32>()?, vec![2.0, 2.0, -6.0, 6.0]);
        Ok(())
    }

    #[test]
    fn limits_repetition_window_to_the_latest_tokens() -> Result<()> {
        let logits = Tensor::new(&[2.0f32, 4.0, 6.0, 8.0], &Device::Cpu)?;
        for (repeat_last_n, expected) in [
            (0, vec![2.0, 4.0, 6.0, 8.0]),
            (1, vec![2.0, 4.0, 6.0, 4.0]),
            (8, vec![1.0, 2.0, 3.0, 4.0]),
        ] {
            let penalized = Sampler::new(1.0, 0, 2.0, repeat_last_n).apply_repetition_penalty(
                &logits,
                &[0],
                &[1, 2, 3],
            )?;
            assert_eq!(penalized.to_vec1::<f32>()?, expected);
        }
        Ok(())
    }
}
