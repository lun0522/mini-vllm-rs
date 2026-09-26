use hyper_util::rt::TokioIo;
use std::path::Path;
use std::process::Child;
use std::process::Command;
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
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_millis(100);
const TRACE_DIRECTORY_ENVIRONMENT_VARIABLE: &str = "MINI_VLLM_TEST_TRACE_DIRECTORY";

#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads and runs Qwen2.5 0.5B; invoke explicitly in release mode"]
async fn qwen_cpu_generation_and_shutdown() {
    let result = run_end_to_end_test().await;
    if let Err(error) = result {
        panic!("Qwen end-to-end test failed: {error:#}");
    }
}

async fn run_end_to_end_test() -> anyhow::Result<()> {
    let socket_directory = tempfile::tempdir_in("/tmp")?;
    let request_socket = socket_directory.path().join("request.sock");
    let control_socket = socket_directory.path().join("control.sock");
    let mut server = spawn_server(&request_socket, &control_socket)?;

    let test_result = async {
        let request_channel =
            wait_for_server(&mut server, &request_socket, STARTUP_TIMEOUT).await?;
        validate_text_generation(request_channel).await
    }
    .await;

    let shutdown_result = shutdown_and_wait(&mut server, &control_socket).await;
    test_result?;
    shutdown_result?;

    anyhow::ensure!(
        !request_socket.exists(),
        "request-handler socket remained after shutdown"
    );
    anyhow::ensure!(
        !control_socket.exists(),
        "control socket remained after shutdown"
    );
    Ok(())
}

fn spawn_server(request_socket: &Path, control_socket: &Path) -> anyhow::Result<Child> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mini-vllm-rs"));
    command
        .arg("--model")
        .arg(MODEL)
        .arg("--inference-device")
        .arg("cpu")
        .arg("--kv-cache-type")
        .arg("contiguous")
        .arg("--target-kv-cache-size-bytes")
        .arg((128 * 1024 * 1024).to_string())
        .arg("--max-batched-token-count")
        .arg("64")
        .arg("--max-active-request-count")
        .arg("1")
        .arg("--request-socket")
        .arg(request_socket)
        .arg("--control-socket")
        .arg(control_socket);
    if let Some(trace_directory) = std::env::var_os(TRACE_DIRECTORY_ENVIRONMENT_VARIABLE) {
        command.arg("--trace-directory").arg(trace_directory);
    }
    command.spawn().map_err(Into::into)
}

async fn validate_text_generation(channel: Channel) -> anyhow::Result<()> {
    use request_handler::generate_text_event::Event;
    use request_handler::request_handler_service_client::RequestHandlerServiceClient;

    let mut client = RequestHandlerServiceClient::new(channel);
    eprintln!("End-to-end test input: {PROMPT:?}");
    let mut stream = client
        .generate_text(request_handler::GenerateText {
            request_id: 0,
            prompt: PROMPT.to_owned(),
            max_new_tokens: 4,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            stream_output: true,
            ignore_eos_tokens: true,
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

    eprintln!("End-to-end test output: {generated_text:?}");
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
    let latency = stats
        .token_generation_latency
        .ok_or_else(|| anyhow::anyhow!("generation returned no latency statistics"))?;
    anyhow::ensure!(
        latency.time_to_first_token_microseconds > 0,
        "time to first token was zero"
    );
    anyhow::ensure!(
        latency.end_to_end_latency_microseconds >= latency.time_to_first_token_microseconds,
        "end-to-end latency was shorter than time to first token"
    );
    anyhow::ensure!(
        stats.draft_token_acceptance_rate.is_none(),
        "target-only generation returned draft-token statistics"
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
            server.kill()?;
            server.wait()?;
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
            server.kill()?;
            server.wait()?;
            anyhow::bail!("server did not exit within {SHUTDOWN_TIMEOUT:?}");
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
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
