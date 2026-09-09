use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::request_handler::GenerateText;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use std::sync::mpsc;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;
use tokio::sync::oneshot;

use super::tokenizer::IncrementalTokenDecoder;
use super::tokenizer::TokenizerWrapper;

pub(super) struct PreprocessedRequest {
    pub(super) generate_text_request: GenerateTextRequest,
    pub(super) token_decoder: IncrementalTokenDecoder,
    pub(super) stream_output: bool,
}

enum Command {
    ProcessRequest {
        request: GenerateText,
        result_sender: oneshot::Sender<Result<PreprocessedRequest>>,
    },
    Shutdown,
}

/// Owns a worker thread pool that concurrently runs CPU-bound request preprocessing outside the
/// async RPC runtime. A worker becomes available again after returning its preprocessed request;
/// it does not wait for model inference to finish.
pub(super) struct InputPreprocessingPool {
    command_sender: mpsc::Sender<Command>,
    worker_threads: Vec<JoinHandle<()>>,
}

impl InputPreprocessingPool {
    pub(super) fn new(tokenizer: TokenizerWrapper, worker_thread_count: usize) -> Result<Self> {
        let tokenizer = Arc::new(tokenizer);
        let (command_sender, command_receiver) = mpsc::channel();
        let command_receiver = Arc::new(Mutex::new(command_receiver));
        let mut worker_threads = Vec::with_capacity(worker_thread_count);
        for worker_index in 0..worker_thread_count {
            let tokenizer = Arc::clone(&tokenizer);
            let command_receiver = Arc::clone(&command_receiver);
            let worker_thread = std::thread::Builder::new()
                .name(format!("input-preprocessor-{worker_index}"))
                .spawn(move || {
                    if let Err(error) = run_worker(worker_index, tokenizer, command_receiver) {
                        log::error!("Input preprocessing worker {worker_index} failed: {error:#}");
                    }
                })
                .with_context(|| {
                    format!("failed to start input preprocessing worker {worker_index}")
                })?;
            worker_threads.push(worker_thread);
        }
        Ok(Self {
            command_sender,
            worker_threads,
        })
    }

    pub(super) async fn preprocess(&self, request: GenerateText) -> Result<PreprocessedRequest> {
        let (result_sender, result_receiver) = oneshot::channel();
        self.command_sender
            .send(Command::ProcessRequest {
                request,
                result_sender,
            })
            .context("input preprocessing workers stopped")?;
        result_receiver
            .await
            .context("input preprocessing worker dropped its result")?
    }
}

impl Drop for InputPreprocessingPool {
    fn drop(&mut self) {
        for _ in &self.worker_threads {
            let _ = self.command_sender.send(Command::Shutdown);
        }
        for worker_thread in self.worker_threads.drain(..) {
            if worker_thread.join().is_err() {
                log::error!("Input preprocessing worker panicked during shutdown");
            }
        }
    }
}

fn run_worker(
    worker_index: usize,
    tokenizer: Arc<TokenizerWrapper>,
    command_receiver: Arc<Mutex<mpsc::Receiver<Command>>>,
) -> Result<()> {
    loop {
        let command = match command_receiver.lock() {
            Ok(receiver) => receiver
                .recv()
                .context("input preprocessing command channel disconnected")?,
            Err(_) => bail!("input preprocessing command receiver lock was poisoned"),
        };
        match command {
            Command::ProcessRequest {
                request,
                result_sender,
            } => {
                log::info!("Input preprocessing state: worker_index={worker_index} status=started");
                let stream_output = request.stream_output;
                let result =
                    tokenizer
                        .create_generate_text_request(request)
                        .map(|generate_text_request| PreprocessedRequest {
                            generate_text_request,
                            token_decoder: tokenizer.create_token_decoder(),
                            stream_output,
                        });
                log::info!(
                    "Input preprocessing state: worker_index={worker_index} status=finished success={}",
                    result.is_ok()
                );
                // A dropped receiver means the RPC was cancelled while preprocessing was running;
                // it is not a worker failure, so the worker remains available for another command.
                let _ = result_sender.send(result);
            }
            Command::Shutdown => return Ok(()),
        }
    }
}
