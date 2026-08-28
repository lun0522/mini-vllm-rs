pub(crate) mod process;
pub(crate) mod server;

use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::model_downloader::ModelFiles;
use crate::proto::GenerateText;
use crate::proto::TextGenerationStats;
use anyhow::Context;
use anyhow::Result;
use candle_core::DType;
use candle_core::Device;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::utils::apply_repeat_penalty;
use log::info;
use std::time::Instant;

/// Owns a loaded model and executes commands received in protobuf format.
pub(crate) struct ModelRunner {
    loaded_model: LoadedModel,
    // TODO: Use the loaded draft model and draft token count for speculative decoding.
    #[expect(dead_code, reason = "speculative decoding is not implemented yet")]
    loaded_draft_model: Option<LoadedModel>,
    #[expect(dead_code, reason = "speculative decoding is not implemented yet")]
    draft_token_count: usize,
}

impl ModelRunner {
    pub(crate) fn new(
        model_files: &ModelFiles,
        draft_model_files: Option<&ModelFiles>,
        draft_token_count: usize,
    ) -> Result<Self> {
        let device = Self::get_inference_device()?;
        let loaded_model = LoadedModel::new(model_files, device)?;
        let loaded_draft_model = draft_model_files
            .map(|draft_model_files| {
                let mut loaded_draft_model =
                    LoadedModel::new(draft_model_files, loaded_model.device().clone())?;
                loaded_draft_model.substitute_tokenizer(&loaded_model);
                Ok::<_, anyhow::Error>(loaded_draft_model)
            })
            .transpose()?;
        info!(
            "Selected inference device {:?} with data type {:?}",
            loaded_model.device(),
            loaded_model.dtype()
        );
        Ok(Self {
            loaded_model,
            loaded_draft_model,
            draft_token_count,
        })
    }

    pub(crate) fn generate_text(
        &mut self,
        command: &GenerateText,
        mut push_fragment: impl FnMut(&str) -> Result<()>,
        mut is_cancelled: impl FnMut() -> bool,
    ) -> Result<TextGenerationStats> {
        let max_new_tokens = usize::try_from(command.max_new_tokens)
            .context("max_new_tokens does not fit in usize")?;
        let repeat_last_n = usize::try_from(command.repeat_last_n)
            .context("repeat_last_n does not fit in usize")?;
        let (model, tokenizer, device) = self.loaded_model.start_inference();
        let model_prompt = Self::format_chat_prompt(&command.prompt);
        let encoding = tokenizer
            .encode(model_prompt, true)
            .map_err(anyhow::Error::msg)
            .context("failed to tokenize the prompt")?;
        let mut tokens = encoding.get_ids().to_vec();
        let prompt_token_count = tokens.len();
        let eos_tokens: Vec<u32> = ["<|endoftext|>", "<|im_end|>"]
            .iter()
            .filter_map(|token| tokenizer.token_to_id(token))
            .collect();
        let mut logits_processor = LogitsProcessor::new(0, None, None);
        let mut token_stream = tokenizer.decode_stream(true);

        let generation_started = Instant::now();
        let mut prefill_finished = None;
        for step in 0..max_new_tokens {
            if is_cancelled() {
                anyhow::bail!("generation request was cancelled");
            }
            // Feed the full prompt once, then one token at a time. The model backend
            // retains the KV cache between these calls.
            let context_size = if step == 0 { tokens.len() } else { 1 };
            let start_pos = tokens.len() - context_size;
            let input = Tensor::new(&tokens[start_pos..], device)?.unsqueeze(0)?;
            let logits = model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let repeat_start = tokens.len().saturating_sub(repeat_last_n);
            let logits =
                apply_repeat_penalty(&logits, command.repeat_penalty, &tokens[repeat_start..])?;
            let next_token = logits_processor.sample(&logits)?;

            if eos_tokens.contains(&next_token) {
                if step == 0 {
                    prefill_finished = Some(Instant::now());
                }
                break;
            }
            tokens.push(next_token);
            if let Some(fragment) = token_stream
                .step(next_token)
                .map_err(anyhow::Error::msg)
                .context("failed to decode generated token")?
            {
                push_fragment(&fragment)?;
            }
            if step == 0 {
                prefill_finished = Some(Instant::now());
            }
        }

        let decode_finished = Instant::now();
        let prefill_finished = prefill_finished.unwrap_or(decode_finished);
        let output_token_count = tokens.len() - prompt_token_count;

        Ok(TextGenerationStats {
            input_token_count: u64::try_from(prompt_token_count)
                .context("input token count does not fit in u64")?,
            output_token_count: u64::try_from(output_token_count)
                .context("output token count does not fit in u64")?,
            prefill_duration_milliseconds: Self::duration_milliseconds(
                prefill_finished.duration_since(generation_started),
            ),
            decode_duration_milliseconds: Self::duration_milliseconds(
                decode_finished.duration_since(prefill_finished),
            ),
        })
    }

    fn format_chat_prompt(prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn duration_milliseconds(duration: std::time::Duration) -> u64 {
        u64::try_from(duration.as_millis()).unwrap_or_default()
    }

    fn get_inference_device() -> Result<Device> {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).context("failed to initialize the Metal device")
        }

        #[cfg(not(feature = "metal"))]
        {
            log::warn!("Metal support is disabled; CPU inference will be slower");
            Ok(Device::Cpu)
        }
    }
}
