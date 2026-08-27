pub(crate) mod process;
pub(crate) mod server;

use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::model_downloader::ModelFiles;
use crate::proto::GenerateText;
use crate::utils::generated_text_output::create_generated_text_output;
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
}

impl ModelRunner {
    pub(crate) fn new(model_files: &ModelFiles) -> Result<Self> {
        let device = Self::get_inference_device()?;
        let loaded_model = LoadedModel::new(model_files, device)?;
        info!(
            "Selected inference device {:?} with data type {:?}",
            loaded_model.device(),
            loaded_model.dtype()
        );
        Ok(Self { loaded_model })
    }

    pub(crate) fn generate_text(&mut self, command: &GenerateText) -> Result<()> {
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
        let mut generated_text_output = create_generated_text_output(command.stream_output);

        let generation_started = Instant::now();
        generated_text_output.start();
        for step in 0..max_new_tokens {
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
                break;
            }
            tokens.push(next_token);
            if let Some(fragment) = token_stream
                .step(next_token)
                .map_err(anyhow::Error::msg)
                .context("failed to decode generated token")?
            {
                generated_text_output.push_fragment(&fragment)?;
            }
        }
        generated_text_output.finish()?;

        let token_count = tokens.len() - prompt_token_count;
        let elapsed_time = generation_started.elapsed();
        let tokens_per_second = token_count as f64 / elapsed_time.as_secs_f64();
        info!(
            "Generated {token_count} tokens in {elapsed_time:.2?} \
             ({tokens_per_second:.2} tokens/s)"
        );

        Ok(())
    }

    fn format_chat_prompt(prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
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
