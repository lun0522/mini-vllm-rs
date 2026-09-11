use crate::proto::model_runner::GenerateTextEvent;
use crate::proto::model_runner::GenerateTextRequest;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::mpsc;
use tonic::Status;

pub(super) struct InferenceRequest {
    pub(super) queued_at: Instant,
    pub(super) generate_text: GenerateTextRequest,
    pub(super) event_sender: mpsc::Sender<Result<GenerateTextEvent, Status>>,
}

/// Owns request payloads and response channels independently of scheduling policy.
pub(super) struct RequestManager {
    requests: HashMap<u64, InferenceRequest>,
}

impl RequestManager {
    pub(super) fn new() -> Self {
        Self {
            requests: HashMap::new(),
        }
    }

    pub(super) fn add_request(&mut self, request: InferenceRequest) -> Result<()> {
        let request_id = request.generate_text.request_id;
        if self.requests.contains_key(&request_id) {
            bail!("inference request {request_id} already exists");
        }
        self.requests.insert(request_id, request);
        Ok(())
    }

    pub(super) fn remove_request(&mut self, request_id: u64) -> Result<InferenceRequest> {
        self.requests
            .remove(&request_id)
            .with_context(|| format!("inference request {request_id} does not exist"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(request_id: u64) -> InferenceRequest {
        let (event_sender, _) = mpsc::channel(1);
        InferenceRequest {
            queued_at: Instant::now(),
            generate_text: GenerateTextRequest {
                request_id,
                ..Default::default()
            },
            event_sender,
        }
    }

    #[test]
    fn stores_and_removes_requests_by_id() -> Result<()> {
        let mut requests = RequestManager::new();
        requests.add_request(request(7))?;

        let request = requests.remove_request(7)?;

        assert_eq!(request.generate_text.request_id, 7);
        assert!(requests.requests.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_duplicate_request_ids() -> Result<()> {
        let mut requests = RequestManager::new();
        requests.add_request(request(7))?;

        let error = requests.add_request(request(7)).unwrap_err().to_string();

        assert_eq!(error, "inference request 7 already exists");
        Ok(())
    }
}
