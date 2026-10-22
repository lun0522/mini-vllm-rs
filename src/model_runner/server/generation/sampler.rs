use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Tensor;
use candle_transformers::utils::apply_repeat_penalty;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;

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

    pub fn sample(
        &mut self,
        logits: &Tensor,
        committed_tokens: &[u32],
        appended_tokens: &[u32],
    ) -> Result<u32> {
        let logits = self
            .apply_repetition_penalty(logits, committed_tokens, appended_tokens)?
            .to_dtype(DType::F32)?;
        if self.temperature == 0.0 {
            return Ok(logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?);
        }
        let scaled_logits = (&logits / self.temperature)?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scaled_logits)?.to_vec1()?;
        self.sample_probabilities(&probabilities)
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
            draft.sample_probabilities(&[0.2, 0.3, 0.5])?;
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
        let mut reference = Sampler::new(2.0, 0, 1.0, 64);
        for _ in 0..64 {
            assert_eq!(
                sampler.sample(&logits, &[], &[])?,
                reference.sample_probabilities(&[1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0])?,
            );
        }
        Ok(())
    }

    #[test]
    fn samples_unnormalized_weights_without_selecting_zero_weights() -> Result<()> {
        let mut sampler = Sampler::new(1.0, 0, 1.0, 64);
        for _ in 0..32 {
            assert_eq!(sampler.sample_probabilities(&[0.0, 5.0, 0.0])?, 1);
        }
        Ok(())
    }

    #[test]
    fn rejects_invalid_probability_weights() {
        let mut sampler = Sampler::new(1.0, 0, 1.0, 64);
        for probabilities in [&[][..], &[0.0, 0.0], &[-1.0, 2.0], &[f32::NAN, 1.0]] {
            assert!(sampler.sample_probabilities(probabilities).is_err());
        }
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
