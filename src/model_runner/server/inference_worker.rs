use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::model_loaders::KvCache;
use crate::model_loaders::ModelRole;
use crate::model_runner::KvCacheType;
use crate::proto::generate_text_event;
use crate::proto::GenerateText;
use crate::proto::GenerateTextEvent;
use crate::proto::TextGenerationStats;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use log::info;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tonic::Status;

use super::kv_cache::create_kv_cache;
use super::text_generation;
use super::tokenizer::load_tokenizer;
use super::tokenizer::validate_tokenizer_compatibility;

/// Owns the loaded models and executes requests on the inference thread.
pub(super) struct ModelRunner {
    loaded_model: LoadedModel,
    target_kv_cache: Box<dyn KvCache>,
    tokenizer: Tokenizer,
    // TODO: Use the loaded draft model and draft token count for speculative decoding.
    #[expect(dead_code, reason = "speculative decoding is not implemented yet")]
    loaded_draft_model: Option<LoadedModel>,
    #[expect(dead_code, reason = "speculative decoding is not implemented yet")]
    draft_kv_cache: Option<Box<dyn KvCache>>,
    #[expect(dead_code, reason = "speculative decoding is not implemented yet")]
    draft_token_count: usize,
}

impl ModelRunner {
    pub(super) fn new(
        model_artifacts: &ModelArtifacts,
        draft_model_artifacts: Option<&ModelArtifacts>,
        draft_token_count: usize,
        kv_cache_type: KvCacheType,
    ) -> Result<Self> {
        let device = Self::get_inference_device()?;
        let tokenizer = load_tokenizer(&model_artifacts.tokenizer)
            .context("failed to load the target model tokenizer")?;
        if let Some(draft_model_artifacts) = draft_model_artifacts {
            let draft_tokenizer = load_tokenizer(&draft_model_artifacts.tokenizer)
                .context("failed to load the draft model tokenizer")?;
            validate_tokenizer_compatibility(&tokenizer, &draft_tokenizer)?;
            // Inference uses the target tokenizer after compatibility is confirmed,
            // so the draft tokenizer is no longer needed.
            drop(draft_tokenizer);
        }
        let loaded_model =
            LoadedModel::new(model_artifacts, device, &tokenizer, ModelRole::Target)?;
        let loaded_draft_model = draft_model_artifacts
            .map(|draft_model_artifacts| {
                LoadedModel::new(
                    draft_model_artifacts,
                    loaded_model.device().clone(),
                    &tokenizer,
                    ModelRole::Draft,
                )
            })
            .transpose()?;
        info!(
            "Selected inference device {:?} for quantized GGUF inference",
            loaded_model.device()
        );
        let target_kv_cache = create_kv_cache(
            kv_cache_type,
            loaded_model.kv_cache_bytes_per_token(),
            loaded_model.layer_count(),
            ModelRole::Target,
        );
        let draft_kv_cache = loaded_draft_model.as_ref().map(|model| {
            create_kv_cache(
                kv_cache_type,
                model.kv_cache_bytes_per_token(),
                model.layer_count(),
                ModelRole::Draft,
            )
        });
        Ok(Self {
            loaded_model,
            target_kv_cache,
            tokenizer,
            draft_kv_cache,
            loaded_draft_model,
            draft_token_count,
        })
    }

    fn generate_text(
        &mut self,
        command: &GenerateText,
        push_fragment: impl FnMut(&str) -> Result<()>,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<TextGenerationStats> {
        self.target_kv_cache.clear();
        text_generation::generate_text(
            &mut self.loaded_model,
            self.target_kv_cache.as_mut(),
            &self.tokenizer,
            command,
            push_fragment,
            is_cancelled,
        )
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

pub(super) struct InferenceRequest {
    pub(super) generate_text: GenerateText,
    pub(super) event_sender: mpsc::Sender<Result<GenerateTextEvent, Status>>,
}

pub(super) fn run(
    mut model_runner: ModelRunner,
    mut inference_receiver: mpsc::Receiver<InferenceRequest>,
) {
    // A single thread owns one model today. This queue can later feed a batching
    // scheduler or route requests to multiple device-specific inference workers.
    while let Some(request) = inference_receiver.blocking_recv() {
        process_request(&mut model_runner, request);
    }
}

fn process_request(model_runner: &mut ModelRunner, request: InferenceRequest) {
    let mut buffered_text = String::new();
    let stream_output = request.generate_text.stream_output;
    let result = model_runner.generate_text(
        &request.generate_text,
        |fragment| {
            if stream_output {
                send_event(
                    &request.event_sender,
                    generate_text_event::Event::Text(fragment.to_owned()),
                )
            } else {
                buffered_text.push_str(fragment);
                Ok(())
            }
        },
        || request.event_sender.is_closed(),
    );

    match result {
        Ok(stats) => {
            send_generation_result(&request.event_sender, stream_output, buffered_text, stats)
        }
        Err(error) => {
            let _ = request
                .event_sender
                .blocking_send(Err(Status::internal(format!(
                    "model runner generation failed: {error:#}"
                ))));
        }
    }
}

fn send_generation_result(
    event_sender: &mpsc::Sender<Result<GenerateTextEvent, Status>>,
    stream_output: bool,
    buffered_text: String,
    stats: TextGenerationStats,
) {
    if !stream_output
        && send_event(
            event_sender,
            generate_text_event::Event::Text(buffered_text),
        )
        .is_err()
    {
        return;
    }
    let _ = send_event(event_sender, generate_text_event::Event::Stats(stats));
}

fn send_event(
    event_sender: &mpsc::Sender<Result<GenerateTextEvent, Status>>,
    event: generate_text_event::Event,
) -> Result<()> {
    event_sender
        .blocking_send(Ok(GenerateTextEvent { event: Some(event) }))
        .context("generation response stream was dropped")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receive_event(
        receiver: &mut mpsc::Receiver<Result<GenerateTextEvent, Status>>,
    ) -> generate_text_event::Event {
        receiver
            .blocking_recv()
            .expect("event channel closed")
            .expect("generation returned an error")
            .event
            .expect("generation event was empty")
    }

    #[test]
    fn generation_result_orders_buffered_text_before_final_stats() {
        let stats = TextGenerationStats {
            input_token_count: 3,
            output_token_count: 2,
            ..Default::default()
        };

        let (sender, mut receiver) = mpsc::channel(2);
        send_generation_result(&sender, false, "answer".to_owned(), stats);
        assert!(matches!(
            receive_event(&mut receiver),
            generate_text_event::Event::Text(text) if text == "answer"
        ));
        assert!(matches!(
            receive_event(&mut receiver),
            generate_text_event::Event::Stats(received) if received == stats
        ));

        let (sender, mut receiver) = mpsc::channel(1);
        send_generation_result(&sender, true, String::new(), stats);
        assert!(matches!(
            receive_event(&mut receiver),
            generate_text_event::Event::Stats(received) if received == stats
        ));
        assert!(receiver.try_recv().is_err());
    }
}
