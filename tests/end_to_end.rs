use hyper_util::rt::TokioIo;
use std::fmt;
use std::fs;
use std::fs::File;
use std::io;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::str::FromStr;
use std::time::Duration;
use tokio::net::UnixStream;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tower::service_fn;

#[allow(dead_code)]
mod inference_config {
    tonic::include_proto!("inference_config");
}

#[allow(dead_code)]
mod model_runner {
    tonic::include_proto!("model_runner");
}

mod request_handler {
    tonic::include_proto!("request_handler");
}

mod main_process {
    tonic::include_proto!("main_process");
}

const MODEL: &str = concat!(
    "model_id: \"bartowski/Qwen2.5-0.5B-Instruct-GGUF\" ",
    "model_filename: \"Qwen2.5-0.5B-Instruct-Q4_K_M.gguf\" ",
    "tokenizer_id: \"Qwen/Qwen2.5-0.5B-Instruct\""
);
const PROMPT: &str = "Reply with a short greeting.";
const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);
const GENERATION_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const TRACE_DIRECTORY_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_TEST_TRACE_DIRECTORY";
const INFERENCE_DEVICE_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_TEST_INFERENCE_DEVICE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestInferenceDevice {
    Cpu,
    Gpu,
    Mixed,
}

impl TestInferenceDevice {
    fn from_environment() -> anyhow::Result<Self> {
        let value = std::env::var(INFERENCE_DEVICE_ENVIRONMENT_VARIABLE)
            .unwrap_or_else(|_| Self::Cpu.to_string());
        value.parse()
    }

    fn validate_compiled_backend(self) -> anyhow::Result<()> {
        if self == Self::Cpu {
            return Ok(());
        }

        #[cfg(target_os = "macos")]
        anyhow::ensure!(
            cfg!(feature = "metal"),
            "{self} E2E testing on macOS requires the `metal` Cargo feature"
        );
        #[cfg(target_os = "linux")]
        anyhow::ensure!(
            cfg!(feature = "cuda"),
            "{self} E2E testing on Linux requires the `cuda` Cargo feature"
        );
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        anyhow::bail!("{self} E2E testing is unsupported on this platform");

        Ok(())
    }

    fn benchmark_backend_ids(self) -> &'static [u64] {
        match self {
            Self::Cpu | Self::Gpu => &[1],
            Self::Mixed => &[1, 2],
        }
    }
}

impl fmt::Display for TestInferenceDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Mixed => "mixed",
        })
    }
}

impl FromStr for TestInferenceDevice {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "mixed" => Ok(Self::Mixed),
            _ => anyhow::bail!(
                "unsupported {INFERENCE_DEVICE_ENVIRONMENT_VARIABLE} value {value:?}; expected cpu, gpu, or mixed"
            ),
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads and runs Qwen2.5 0.5B; invoke explicitly in release mode"]
async fn run_qwen_generation_and_shutdown() {
    let result = run_end_to_end_test().await;
    if let Err(error) = result {
        panic!("Qwen end-to-end test failed: {error:#}");
    }
}

async fn run_end_to_end_test() -> anyhow::Result<()> {
    let inference_device = TestInferenceDevice::from_environment()?;
    inference_device.validate_compiled_backend()?;
    let socket_directory = tempfile::tempdir_in("/tmp")?;
    let request_socket = socket_directory.path().join("request.sock");
    let control_socket = socket_directory.path().join("control.sock");
    let log_path = socket_directory.path().join("server.log");
    let mut server = spawn_server(
        &request_socket,
        &control_socket,
        &log_path,
        inference_device,
    )?;

    let test_result = async {
        let request_channel =
            wait_for_server(&mut server, &request_socket, STARTUP_TIMEOUT).await?;
        tokio::time::timeout(
            GENERATION_TIMEOUT,
            validate_text_generation(request_channel, inference_device),
        )
        .await
        .map_err(|_| anyhow::anyhow!("generation did not complete within {GENERATION_TIMEOUT:?}"))?
    }
    .await;

    if let Err(error) = test_result {
        let cleanup_result = terminate_and_reap(&mut server).await;
        return Err(with_server_diagnostics(
            error.context("end-to-end request failed"),
            &log_path,
            cleanup_result.err(),
        ));
    }

    if let Err(error) = shutdown_and_wait(&mut server, &control_socket).await {
        let cleanup_result = terminate_and_reap(&mut server).await;
        return Err(with_server_diagnostics(
            error,
            &log_path,
            cleanup_result.err(),
        ));
    }

    let socket_cleanup_result = (|| {
        anyhow::ensure!(
            !request_socket.exists(),
            "request-handler socket remained after shutdown"
        );
        anyhow::ensure!(
            !control_socket.exists(),
            "control socket remained after shutdown"
        );
        anyhow::Ok(())
    })();
    if let Err(error) = socket_cleanup_result {
        return Err(with_server_diagnostics(error, &log_path, None));
    }
    Ok(())
}

fn spawn_server(
    request_socket: &Path,
    control_socket: &Path,
    log_path: &Path,
    inference_device: TestInferenceDevice,
) -> anyhow::Result<Child> {
    let stdout = File::create(log_path)?;
    let stderr = stdout.try_clone()?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_mini-vllm-rs"));
    command
        .process_group(0)
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .arg("--model")
        .arg(MODEL)
        .arg("--inference-device")
        .arg(inference_device.to_string())
        .arg("--activation-dtype")
        .arg("f16")
        .arg("--kv-cache-config")
        .arg(format!(
            "kv_cache_type: KV_CACHE_TYPE_CONTIGUOUS target_kv_cache_size_bytes: {}",
            128 * 1024 * 1024
        ))
        .arg("--scheduler-config")
        .arg("max_batched_token_count: 64 max_active_request_count: 1")
        .arg("--request-orchestrator-config")
        .arg("routing_policy: REQUEST_ROUTING_POLICY_ROUND_ROBIN")
        .arg("--request-socket")
        .arg(request_socket)
        .arg("--control-socket")
        .arg(control_socket);
    if let Some(trace_directory) = std::env::var_os(TRACE_DIRECTORY_ENVIRONMENT_VARIABLE) {
        command.arg("--trace-directory").arg(trace_directory);
    }
    command.spawn().map_err(Into::into)
}

async fn validate_text_generation(
    channel: Channel,
    inference_device: TestInferenceDevice,
) -> anyhow::Result<()> {
    for (request_index, backend_id) in inference_device
        .benchmark_backend_ids()
        .iter()
        .copied()
        .enumerate()
    {
        validate_text_generation_on_backend(channel.clone(), request_index as u64, backend_id)
            .await?;
    }
    Ok(())
}

async fn validate_text_generation_on_backend(
    channel: Channel,
    request_id: u64,
    benchmark_backend_id: u64,
) -> anyhow::Result<()> {
    use request_handler::generate_text_event::Event;
    use request_handler::request_handler_service_client::RequestHandlerServiceClient;

    let mut client = RequestHandlerServiceClient::new(channel);
    eprintln!("End-to-end test input on backend {benchmark_backend_id}: {PROMPT:?}");
    let mut stream = client
        .generate_text(request_handler::GenerateText {
            request_id,
            prompt: PROMPT.to_owned(),
            max_new_tokens: 4,
            temperature: 0.0,
            total_variation_budget: 0.0,
            seed: None,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            stream_output: true,
            ignore_eos_tokens: true,
            benchmark_backend_id,
        })
        .await?
        .into_inner();
    let mut generated_text = String::new();
    let mut stats = None;
    let mut event_count = 0;
    let mut stats_event_index = None;

    while let Some(message) = stream.message().await? {
        event_count += 1;
        match message.event {
            Some(Event::Text(text)) => generated_text.push_str(&text),
            Some(Event::Stats(event_stats)) => {
                anyhow::ensure!(stats.is_none(), "received more than one statistics event");
                stats_event_index = Some(event_count);
                stats = Some(event_stats);
            }
            None => anyhow::bail!("received a generation event without a payload"),
        }
    }

    eprintln!("End-to-end test output from backend {benchmark_backend_id}: {generated_text:?}");
    anyhow::ensure!(!generated_text.is_empty(), "generation returned no text");
    let stats = stats.ok_or_else(|| anyhow::anyhow!("generation returned no statistics"))?;
    anyhow::ensure!(
        stats_event_index == Some(event_count),
        "statistics were not the final generation event"
    );
    anyhow::ensure!(stats.input_token_count > 0, "input token count was zero");
    anyhow::ensure!(
        stats.output_token_count == 4,
        "expected 4 output tokens, received {}",
        stats.output_token_count
    );
    anyhow::ensure!(
        stats.finish_reason == model_runner::TextGenerationFinishReason::MaxNewTokensReached as i32,
        "expected maximum-new-token-count finish reason, received {}",
        stats.finish_reason
    );
    let timing = stats
        .timing
        .ok_or_else(|| anyhow::anyhow!("generation returned no timing statistics"))?;
    anyhow::ensure!(
        timing.time_to_first_token_microseconds > 0,
        "time to first token was zero"
    );
    anyhow::ensure!(
        timing.time_to_last_token_microseconds >= timing.time_to_first_token_microseconds,
        "time to last token was shorter than time to first token"
    );
    anyhow::ensure!(
        timing.request_completion_latency_microseconds >= timing.time_to_last_token_microseconds,
        "request completion latency was shorter than time to last token"
    );
    anyhow::ensure!(
        stats.draft_token_stats.is_none(),
        "target-only generation returned draft-token statistics"
    );
    let target_prefix_cache_stats = stats
        .target_prefix_cache_stats
        .ok_or_else(|| anyhow::anyhow!("generation returned no target prefix-cache statistics"))?;
    anyhow::ensure!(
        target_prefix_cache_stats.restored_token_count == 0,
        "first request unexpectedly restored target prefix-cache tokens"
    );
    anyhow::ensure!(
        stats.draft_prefix_cache_stats.is_none(),
        "target-only generation returned draft prefix-cache statistics"
    );
    anyhow::ensure!(
        timing.prefill_duration_microseconds > 0,
        "prefill duration was zero"
    );
    anyhow::ensure!(
        timing.decode_duration_microseconds > 0,
        "decode duration was zero"
    );
    Ok(())
}

async fn shutdown_and_wait(server: &mut Child, control_socket: &Path) -> anyhow::Result<()> {
    use main_process::main_process_service_client::MainProcessServiceClient;

    if server.try_wait()?.is_none() {
        let shutdown_result = async {
            let channel = wait_for_server(server, control_socket, SHUTDOWN_TIMEOUT).await?;
            let mut client = MainProcessServiceClient::new(channel);
            client.shutdown(main_process::Shutdown {}).await?;
            anyhow::Ok(())
        }
        .await;
        if let Err(error) = shutdown_result {
            return Err(error.context("failed to shut down the server through its control RPC"));
        }
    }

    let deadline = tokio::time::Instant::now() + SHUTDOWN_TIMEOUT;
    loop {
        if let Some(status) = server.try_wait()? {
            anyhow::ensure!(status.success(), "server exited unsuccessfully: {status}");
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("server did not exit within {SHUTDOWN_TIMEOUT:?}");
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

async fn terminate_and_reap(server: &mut Child) -> anyhow::Result<()> {
    if server.try_wait()?.is_some() {
        return Ok(());
    }

    signal_process_group(server.id(), libc::SIGTERM)?;
    let deadline = tokio::time::Instant::now() + SHUTDOWN_TIMEOUT;
    loop {
        if server.try_wait()?.is_some() {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            signal_process_group(server.id(), libc::SIGKILL)?;
            server.wait()?;
            return Ok(());
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

fn signal_process_group(process_id: u32, signal: i32) -> anyhow::Result<()> {
    let process_group = i32::try_from(process_id)
        .map_err(|_| anyhow::anyhow!("process ID {process_id} does not fit in i32"))?;
    // The server is started as its own process-group leader. Its model-runner and
    // request-handler children inherit that group, so one signal reaches the complete tree.
    let result = unsafe { libc::kill(-process_group, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error.into())
    }
}

fn with_server_diagnostics(
    error: anyhow::Error,
    log_path: &Path,
    cleanup_error: Option<anyhow::Error>,
) -> anyhow::Error {
    let logs = fs::read_to_string(log_path)
        .unwrap_or_else(|log_error| format!("<failed to read server log: {log_error}>"));
    let cleanup = cleanup_error
        .map(|cleanup_error| format!("\nprocess cleanup also failed: {cleanup_error:#}"))
        .unwrap_or_default();
    anyhow::anyhow!(
        "{error:#}{cleanup}\nrequested test backend: {}\nserver log ({}):\n{}",
        std::env::var(INFERENCE_DEVICE_ENVIRONMENT_VARIABLE)
            .unwrap_or_else(|_| TestInferenceDevice::Cpu.to_string()),
        log_path.display(),
        logs
    )
}

async fn wait_for_server(
    server: &mut Child,
    socket_path: &Path,
    timeout: Duration,
) -> anyhow::Result<Channel> {
    let deadline = tokio::time::Instant::now() + timeout;
    let mut last_error = None;
    while tokio::time::Instant::now() < deadline {
        if let Some(status) = server.try_wait()? {
            anyhow::bail!("server exited before becoming ready: {status}");
        }
        match connect(socket_path).await {
            Ok(channel) => return Ok(channel),
            Err(error) => last_error = Some(error),
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    Err(last_error.unwrap_or_else(|| anyhow::anyhow!("server did not become ready")))
}

async fn connect(socket_path: &Path) -> anyhow::Result<Channel> {
    let connector_path = socket_path.to_owned();
    Endpoint::from_static("http://localhost")
        .connect_timeout(Duration::from_secs(1))
        .connect_with_connector(service_fn(move |_| {
            let connector_path = connector_path.clone();
            async move { UnixStream::connect(connector_path).await.map(TokioIo::new) }
        }))
        .await
        .map_err(Into::into)
}
