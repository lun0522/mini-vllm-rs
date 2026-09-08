use crate::model_runner::KvCacheType;
use crate::models::loaded_model::LoadedModel;
use crate::proto::model_runner::model_runner_command;
use crate::proto::model_runner::model_runner_service_server::ModelRunnerService;
use crate::proto::model_runner::model_runner_service_server::ModelRunnerServiceServer;
use crate::proto::model_runner::CommandResult;
use crate::proto::model_runner::GenerateTextEvent;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::GetModelMetadataRequest;
use crate::proto::model_runner::GetModelMetadataResponse;
use crate::proto::model_runner::ModelRunnerCommand;
use crate::utils::rpc_shutdown::RpcShutdown;
use anyhow::Context;
use anyhow::Result;
use candle_core::Tensor;
use std::cell::RefCell;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tokio::net::UnixListener;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::wrappers::UnixListenerStream;
use tonic::Request;
use tonic::Response;
use tonic::Status;

mod cli;
mod inference_worker;
mod kv_cache;
mod text_generation;

pub(crate) use cli::ModelRunnerProcessArgs;
use inference_worker::InferenceRequest;
use inference_worker::ModelRunner;
use kv_cache::KvCacheBackend;

pub(crate) const PROCESS_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_MODEL_RUNNER";
const INFERENCE_QUEUE_CAPACITY: usize = 32;
const GENERATION_EVENT_QUEUE_CAPACITY: usize = 32;

pub(super) struct ModelAndKvCache {
    model: RefCell<LoadedModel>,
    kv_cache: RefCell<KvCacheBackend>,
}

impl ModelAndKvCache {
    fn new(model: LoadedModel, kv_cache: KvCacheBackend) -> Self {
        Self {
            model: RefCell::new(model),
            kv_cache: RefCell::new(kv_cache),
        }
    }

    fn forward(&self, input: &Tensor, start_position: usize) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache = self.kv_cache.borrow_mut();
        model.model().forward(input, start_position, &mut *kv_cache)
    }

    fn forward_for_speculative_verification(
        &self,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache = self.kv_cache.borrow_mut();
        model
            .model()
            .forward_for_speculative_verification(input, start_position, &mut *kv_cache)
    }

    /// Restores reusable prefix pages and returns the prefill start position.
    fn restore_cached_prefix(&self, token_ids: &[u32]) -> Result<usize> {
        self.kv_cache.borrow_mut().restore_cached_prefix(token_ids)
    }

    fn evicted_cached_token_count(&self) -> usize {
        self.kv_cache.borrow().evicted_cached_token_count()
    }

    fn truncate_cache(&self, target_token_count: usize) -> Result<()> {
        self.kv_cache.borrow_mut().truncate(target_token_count)
    }

    fn clear_cache(&self) -> Result<()> {
        self.kv_cache.borrow_mut().clear()
    }

    fn finish_request(&self, token_ids: &[u32]) -> Result<()> {
        self.kv_cache.borrow_mut().finish_request(token_ids)
    }
}

pub(crate) async fn run(args: ModelRunnerProcessArgs) -> Result<()> {
    run_server(
        &args.model_path,
        args.draft_model_path,
        args.draft_token_count,
        args.kv_cache_type,
        args.target_kv_cache_size_bytes,
        &args.socket_path,
    )
    .await
}

async fn run_server(
    model_path: &Path,
    draft_model_path: Option<PathBuf>,
    draft_token_count: usize,
    kv_cache_type: KvCacheType,
    target_kv_cache_size_bytes: usize,
    socket_path: &Path,
) -> Result<()> {
    // Bind only after model initialization succeeds so the socket itself is a
    // readiness signal for the parent process.
    let model_runner = ModelRunner::new(
        model_path,
        draft_model_path.as_deref(),
        draft_token_count,
        kv_cache_type,
        target_kv_cache_size_bytes,
    )?;
    let model_metadata = model_runner.model_metadata();
    let token_capacity = model_runner.token_capacity();
    let listener = UnixListener::bind(socket_path)
        .context("failed to bind the model runner Unix domain socket")?;
    let (inference_sender, inference_receiver) = mpsc::channel(INFERENCE_QUEUE_CAPACITY);
    let inference_thread = std::thread::Builder::new()
        .name("inference-worker".to_owned())
        .spawn(move || inference_worker::run(model_runner, inference_receiver))
        .context("failed to start the model runner inference thread")?;
    let (shutdown, shutdown_receiver) = RpcShutdown::channel();
    let service = ModelRunnerRpcService {
        inference_sender,
        model_metadata,
        token_capacity,
        request_id: RequestId::new(),
        shutdown,
    };

    let server_result = tonic::transport::Server::builder()
        .add_service(ModelRunnerServiceServer::new(service))
        .serve_with_incoming_shutdown(UnixListenerStream::new(listener), async {
            let _ = shutdown_receiver.await;
        })
        .await
        .context("model runner RPC server failed");
    inference_thread
        .join()
        .map_err(|_| anyhow::anyhow!("model runner inference thread panicked"))?;
    server_result
}

struct ModelRunnerRpcService {
    inference_sender: mpsc::Sender<InferenceRequest>,
    model_metadata: GetModelMetadataResponse,
    token_capacity: usize,
    request_id: RequestId,
    shutdown: RpcShutdown,
}

struct RequestId(AtomicU64);

impl RequestId {
    fn new() -> Self {
        Self(AtomicU64::new(1))
    }

    fn next(&self) -> Result<u64, &'static str> {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |request_id| {
                request_id.checked_add(1)
            })
            .map_err(|_| "request ID space is exhausted")
    }
}

#[tonic::async_trait]
impl ModelRunnerService for ModelRunnerRpcService {
    type GenerateTextStream = ReceiverStream<Result<GenerateTextEvent, Status>>;

    async fn get_model_metadata(
        &self,
        _request: Request<GetModelMetadataRequest>,
    ) -> Result<Response<GetModelMetadataResponse>, Status> {
        Ok(Response::new(self.model_metadata))
    }

    async fn generate_text(
        &self,
        request: Request<GenerateTextRequest>,
    ) -> Result<Response<Self::GenerateTextStream>, Status> {
        let mut request = request.into_inner();
        normalize_generate_text_request(&mut request, self.token_capacity)
            .map_err(Status::invalid_argument)?;
        let request_id = self.request_id.next().map_err(Status::resource_exhausted)?;
        let queued_at = Instant::now();
        let (event_sender, event_receiver) = mpsc::channel(GENERATION_EVENT_QUEUE_CAPACITY);
        self.inference_sender
            .send(InferenceRequest {
                request_id,
                queued_at,
                generate_text: request,
                event_sender,
            })
            .await
            .map_err(|_| Status::unavailable("model runner inference thread stopped"))?;
        Ok(Response::new(ReceiverStream::new(event_receiver)))
    }

    async fn handle_command(
        &self,
        request: Request<ModelRunnerCommand>,
    ) -> Result<Response<CommandResult>, Status> {
        let command = request
            .into_inner()
            .command
            .ok_or_else(|| Status::invalid_argument("model runner command is empty"))?;
        match command {
            model_runner_command::Command::Shutdown(_) => self.shutdown.trigger()?,
        }
        Ok(Response::new(CommandResult {}))
    }
}

fn normalize_generate_text_request(
    request: &mut GenerateTextRequest,
    token_capacity: usize,
) -> Result<(), String> {
    if request.input_token_ids.is_empty() {
        return Err("input token IDs must not be empty".to_owned());
    }
    let input_token_count = request.input_token_ids.len();
    if input_token_count > token_capacity {
        return Err(format!(
            "request input contains {input_token_count} tokens but the configured KV-cache \
             capacity is {token_capacity}"
        ));
    }

    // The final generated token remains pending rather than being written to the cache, so one
    // output token can be generated even when the input already fills the cache.
    let maximum_new_token_count = u64::try_from(token_capacity - input_token_count + 1)
        .map_err(|_| "maximum new token count does not fit in u64".to_owned())?;
    if request.max_new_tokens > maximum_new_token_count {
        log::warn!(
            "Requested {} new tokens, but the KV cache can hold at most {} for this input; \
             reducing max_new_tokens to {}",
            request.max_new_tokens,
            maximum_new_token_count,
            maximum_new_token_count,
        );
        request.max_new_tokens = maximum_new_token_count;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::normalize_generate_text_request;
    use crate::proto::model_runner::GenerateTextRequest;

    #[test]
    fn limits_generation_to_the_available_cache_capacity() {
        let mut request = GenerateTextRequest {
            input_token_ids: vec![1, 2, 3],
            max_new_tokens: 3,
            ..Default::default()
        };

        normalize_generate_text_request(&mut request, 4).unwrap();

        assert_eq!(request.max_new_tokens, 2);
    }

    #[test]
    fn rejects_empty_model_input() {
        let mut request = GenerateTextRequest::default();

        let error = normalize_generate_text_request(&mut request, 16).unwrap_err();

        assert_eq!(error, "input token IDs must not be empty");
    }

    #[test]
    fn rejects_input_that_exceeds_cache_capacity() {
        let mut request = GenerateTextRequest {
            input_token_ids: vec![1; 17],
            max_new_tokens: 0,
            ..Default::default()
        };

        let error = normalize_generate_text_request(&mut request, 16).unwrap_err();

        assert!(error.contains("input contains 17 tokens"));
    }

    #[test]
    fn allows_one_output_token_when_input_fills_cache() {
        let mut request = GenerateTextRequest {
            input_token_ids: vec![1; 16],
            max_new_tokens: 8,
            ..Default::default()
        };

        normalize_generate_text_request(&mut request, 16).unwrap();

        assert_eq!(request.max_new_tokens, 1);
    }
}
