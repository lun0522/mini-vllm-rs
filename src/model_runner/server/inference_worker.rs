use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::model_downloader::ModelArtifacts;
use crate::proto::generate_text_event;
use crate::proto::GenerateText;
use crate::proto::GenerateTextEvent;
use crate::proto::TextGenerationStats;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use log::info;
use tokio::sync::mpsc;
use tonic::Status;

use super::text_generation;

/// Owns the loaded models and executes requests on the inference thread.
pub(super) struct ModelRunner {
    loaded_model: LoadedModel,
    // TODO: Use the loaded draft model and draft token count for speculative decoding.
    #[expect(dead_code, reason = "speculative decoding is not implemented yet")]
    loaded_draft_model: Option<LoadedModel>,
    #[expect(dead_code, reason = "speculative decoding is not implemented yet")]
    draft_token_count: usize,
}

impl ModelRunner {
    pub(super) fn new(
        model_artifacts: &ModelArtifacts,
        draft_model_artifacts: Option<&ModelArtifacts>,
        draft_token_count: usize,
    ) -> Result<Self> {
        let device = Self::get_inference_device()?;
        let loaded_model = LoadedModel::new(model_artifacts, device)?;
        let loaded_draft_model = draft_model_artifacts
            .map(|draft_model_artifacts| {
                let mut loaded_draft_model =
                    LoadedModel::new(draft_model_artifacts, loaded_model.device().clone())?;
                loaded_draft_model.substitute_tokenizer(&loaded_model);
                Ok::<_, anyhow::Error>(loaded_draft_model)
            })
            .transpose()?;
        info!(
            "Selected inference device {:?} for quantized GGUF inference",
            loaded_model.device()
        );
        Ok(Self {
            loaded_model,
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
        text_generation::generate_text(&mut self.loaded_model, command, push_fragment, is_cancelled)
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
