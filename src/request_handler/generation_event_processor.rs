use crate::proto::model_runner::generate_text_event as model_runner_generate_text_event;
use crate::proto::model_runner::GenerateTextEvent as ModelRunnerGenerateTextEvent;
use crate::proto::model_runner::TextGenerationStats;
use crate::proto::request_handler::generate_text_event;
use crate::proto::request_handler::GenerateTextEvent;
use tonic::Status;

use super::tokenizer::IncrementalTokenDecoder;

pub(super) struct GenerationEventProcessor {
    decoder: IncrementalTokenDecoder,
    stream_output: bool,
    buffered_text: String,
}

pub(super) struct ProcessedGenerationEvents {
    pub(super) events: Vec<GenerateTextEvent>,
    pub(super) finished: bool,
}

impl GenerationEventProcessor {
    pub(super) fn new(decoder: IncrementalTokenDecoder, stream_output: bool) -> Self {
        Self {
            decoder,
            stream_output,
            buffered_text: String::new(),
        }
    }

    pub(super) fn process(
        &mut self,
        event: ModelRunnerGenerateTextEvent,
    ) -> Result<ProcessedGenerationEvents, Box<Status>> {
        let event = event
            .event
            .ok_or_else(|| Box::new(Status::internal("model-runner event is empty")))?;
        match event {
            model_runner_generate_text_event::Event::TokenId(token_id) => {
                let fragment = self.decoder.step(token_id).map_err(|error| {
                    Box::new(Status::internal(format!(
                        "failed to decode model output: {error:#}"
                    )))
                })?;
                let mut events = Vec::new();
                if let Some(fragment) = fragment {
                    if self.stream_output {
                        events.push(text_event(fragment));
                    } else {
                        self.buffered_text.push_str(&fragment);
                    }
                }
                Ok(ProcessedGenerationEvents {
                    events,
                    finished: false,
                })
            }
            model_runner_generate_text_event::Event::Stats(stats) => {
                let events = if self.stream_output {
                    vec![stats_event(stats)]
                } else {
                    vec![
                        text_event(std::mem::take(&mut self.buffered_text)),
                        stats_event(stats),
                    ]
                };
                Ok(ProcessedGenerationEvents {
                    events,
                    finished: true,
                })
            }
        }
    }
}

fn text_event(text: String) -> GenerateTextEvent {
    GenerateTextEvent {
        event: Some(generate_text_event::Event::Text(text)),
    }
}

fn stats_event(stats: TextGenerationStats) -> GenerateTextEvent {
    GenerateTextEvent {
        event: Some(generate_text_event::Event::Stats(stats)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::model_runner::TextGenerationStats;
    use tokenizers::Tokenizer;

    fn processor(stream_output: bool) -> GenerationEventProcessor {
        let tokenizer = Tokenizer::from_bytes(
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}}"#,
        )
        .unwrap();
        GenerationEventProcessor::new(IncrementalTokenDecoder::new(tokenizer), stream_output)
    }

    fn token_event(token_id: u32) -> ModelRunnerGenerateTextEvent {
        ModelRunnerGenerateTextEvent {
            event: Some(model_runner_generate_text_event::Event::TokenId(token_id)),
        }
    }

    fn stats_event() -> ModelRunnerGenerateTextEvent {
        ModelRunnerGenerateTextEvent {
            event: Some(model_runner_generate_text_event::Event::Stats(
                TextGenerationStats::default(),
            )),
        }
    }

    fn event_text(event: &GenerateTextEvent) -> Option<&str> {
        match event.event.as_ref() {
            Some(generate_text_event::Event::Text(text)) => Some(text),
            _ => None,
        }
    }

    #[test]
    fn converts_token_ids_to_streamed_text_events() {
        let mut processor = processor(true);

        let first = processor.process(token_event(1)).unwrap();
        let second = processor.process(token_event(2)).unwrap();
        let final_events = processor.process(stats_event()).unwrap();

        assert_eq!(first.events.len(), 1);
        assert_eq!(event_text(&first.events[0]), Some("hello"));
        assert!(!first.finished);
        assert_eq!(second.events.len(), 1);
        assert_eq!(event_text(&second.events[0]), Some(" world"));
        assert!(!second.finished);
        assert_eq!(final_events.events.len(), 1);
        assert!(matches!(
            final_events.events[0].event,
            Some(generate_text_event::Event::Stats(_))
        ));
        assert!(final_events.finished);
    }

    #[test]
    fn buffers_text_and_emits_it_before_statistics() {
        let mut processor = processor(false);

        assert!(processor.process(token_event(1)).unwrap().events.is_empty());
        assert!(processor.process(token_event(2)).unwrap().events.is_empty());
        let final_events = processor.process(stats_event()).unwrap();

        assert_eq!(final_events.events.len(), 2);
        assert_eq!(event_text(&final_events.events[0]), Some("hello world"));
        assert!(matches!(
            final_events.events[1].event,
            Some(generate_text_event::Event::Stats(_))
        ));
        assert!(final_events.finished);
    }

    #[test]
    fn rejects_empty_model_runner_events() {
        let error = processor(true)
            .process(ModelRunnerGenerateTextEvent { event: None })
            .err()
            .expect("empty events should be rejected");

        assert_eq!(error.code(), tonic::Code::Internal);
        assert_eq!(error.message(), "model-runner event is empty");
    }
}
