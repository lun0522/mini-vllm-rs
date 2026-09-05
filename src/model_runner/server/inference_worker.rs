use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::ModelInfo;
use crate::model_loaders::ModelRole;
use crate::model_runner::KvCacheType;
use crate::proto::model_runner::generate_text_event;
use crate::proto::model_runner::GenerateTextEvent;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::GetModelMetadataResponse;
use crate::proto::model_runner::TextGenerationStats;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use log::info;
use std::path::Path;
use tokio::sync::mpsc;
use tonic::Status;

use super::kv_cache::create_kv_cache;
use super::text_generation;
use super::ModelAndKvCache;

/// Owns the loaded models and executes requests on the inference thread.
pub(super) struct ModelRunner {
    target: ModelAndKvCache,
    draft: Option<ModelAndKvCache>,
    draft_token_count: usize,
}

impl ModelRunner {
    pub(super) fn new(
        model_path: &Path,
        draft_model_path: Option<&Path>,
        draft_token_count: usize,
        kv_cache_type: KvCacheType,
        target_kv_cache_size_bytes: usize,
    ) -> Result<Self> {
        let device = Self::get_inference_device()?;
        let loaded_model = LoadedModel::new(model_path, device)?;
        let loaded_draft_model = draft_model_path
            .map(|draft_model_path| {
                LoadedModel::new(draft_model_path, loaded_model.device().clone())
            })
            .transpose()?;
        info!(
            "Selected inference device {:?} for quantized GGUF inference",
            loaded_model.device()
        );
        let target_kv_cache = create_kv_cache(
            kv_cache_type,
            &loaded_model,
            ModelRole::Target,
            target_kv_cache_size_bytes,
        )?;
        let target_kv_cache_token_capacity = target_kv_cache.token_capacity();
        let draft = loaded_draft_model
            .map(|model| {
                let draft_kv_cache_size_bytes =
                    compute_kv_cache_size_bytes(model.info(), target_kv_cache_token_capacity)?;
                let kv_cache = create_kv_cache(
                    kv_cache_type,
                    &model,
                    ModelRole::Draft,
                    draft_kv_cache_size_bytes,
                )?;
                Ok::<_, anyhow::Error>(ModelAndKvCache::new(model, kv_cache))
            })
            .transpose()?;
        Ok(Self {
            target: ModelAndKvCache::new(loaded_model, target_kv_cache),
            draft,
            draft_token_count,
        })
    }

    pub(super) fn model_metadata(&self) -> GetModelMetadataResponse {
        let target_model = self.target.model.borrow().metadata();
        let draft_model = self
            .draft
            .as_ref()
            .map(|draft| draft.model.borrow().metadata());
        GetModelMetadataResponse {
            target_model: Some(target_model),
            draft_model,
        }
    }

    fn generate_text(
        &mut self,
        request: &GenerateTextRequest,
        push_token: impl FnMut(u32) -> Result<()>,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<TextGenerationStats> {
        self.target.kv_cache.borrow_mut().clear()?;
        if let Some(draft) = self.draft.as_ref() {
            draft.kv_cache.borrow_mut().clear()?;
        }
        text_generation::generate_text(
            &self.target,
            self.draft.as_ref(),
            self.draft_token_count,
            request,
            push_token,
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

fn compute_kv_cache_size_bytes(model_info: &ModelInfo, token_capacity: usize) -> Result<usize> {
    model_info
        .kv_cache_bytes_per_token()
        .checked_mul(model_info.layer_count)
        .and_then(|size| size.checked_mul(token_capacity))
        .and_then(|size| size.checked_mul(2))
        .context("KV-cache size exceeds usize")
}

pub(super) struct InferenceRequest {
    pub(super) generate_text: GenerateTextRequest,
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
    let result = model_runner.generate_text(
        &request.generate_text,
        |token_id| {
            send_event(
                &request.event_sender,
                generate_text_event::Event::TokenId(token_id),
            )
        },
        || request.event_sender.is_closed(),
    );

    match result {
        Ok(stats) => {
            let _ = send_event(
                &request.event_sender,
                generate_text_event::Event::Stats(stats),
            );
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
    use candle_core::DType;

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
    fn generation_result_follows_generated_tokens_with_final_stats() {
        let stats = TextGenerationStats {
            input_token_count: 3,
            output_token_count: 2,
            ..Default::default()
        };

        let (sender, mut receiver) = mpsc::channel(2);
        send_event(&sender, generate_text_event::Event::TokenId(42)).unwrap();
        send_event(&sender, generate_text_event::Event::Stats(stats)).unwrap();
        assert!(matches!(
            receive_event(&mut receiver),
            generate_text_event::Event::TokenId(42)
        ));
        assert!(matches!(
            receive_event(&mut receiver),
            generate_text_event::Event::Stats(received) if received == stats
        ));
        assert!(receiver.try_recv().is_err());
    }

    #[test]
    fn derives_draft_cache_size_for_target_token_capacity() -> Result<()> {
        let draft_model_info = ModelInfo {
            layer_count: 4,
            kv_head_count: 2,
            head_dimension: 8,
            activation_dtype: DType::F32,
        };
        let target_token_capacity = 128;

        let size_bytes = compute_kv_cache_size_bytes(&draft_model_info, target_token_capacity)?;

        let derived_token_capacity = size_bytes
            / 2
            / draft_model_info.layer_count
            / draft_model_info.kv_cache_bytes_per_token();
        assert_eq!(derived_token_capacity, target_token_capacity);
        Ok(())
    }
}
