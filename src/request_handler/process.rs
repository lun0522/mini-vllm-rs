use crate::proto::request_handler::request_handler_service_client::RequestHandlerServiceClient;
use crate::proto::request_handler::GenerateText;
use crate::proto::request_handler::Shutdown;
use crate::proto::GenerateTextEvent;
use crate::request_handler::server;
use crate::utils::child_process::ChildProcess;
use crate::utils::domain_socket;
use anyhow::Context;
use anyhow::Result;
use std::io::ErrorKind;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use tonic::transport::Channel;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Owns the request handler process, RPC client, and public socket path.
pub(crate) struct RequestHandlerProcess {
    rpc_client: RequestHandlerServiceClient<Channel>,
    child_process: ChildProcess,
    socket_path: PathBuf,
}

impl RequestHandlerProcess {
    pub(crate) async fn start(
        model_runner_socket_path: &Path,
        tokenizer_path: &Path,
        draft_tokenizer_path: Option<&Path>,
        request_handler_socket_path: PathBuf,
    ) -> Result<Self> {
        if request_handler_socket_path
            .try_exists()
            .context("failed to inspect the request handler socket path")?
        {
            anyhow::bail!(
                "request handler socket '{}' already exists",
                request_handler_socket_path.display()
            );
        }

        let mut child_process = spawn(
            model_runner_socket_path,
            tokenizer_path,
            draft_tokenizer_path,
            request_handler_socket_path.as_path(),
        )?;
        let channel = match domain_socket::wait_for_server(
            &mut child_process,
            &request_handler_socket_path,
            STARTUP_TIMEOUT,
        )
        .await
        {
            Ok(channel) => channel,
            Err(error) => {
                let _ = child_process.stop();
                let _ = remove_socket(&request_handler_socket_path);
                return Err(error);
            }
        };
        Ok(Self {
            rpc_client: RequestHandlerServiceClient::new(channel),
            child_process,
            socket_path: request_handler_socket_path,
        })
    }

    pub(crate) async fn generate_text(
        &self,
        request: GenerateText,
    ) -> Result<tonic::Streaming<GenerateTextEvent>> {
        let mut rpc_client = self.rpc_client.clone();
        Ok(rpc_client
            .generate_text(request)
            .await
            .context("request handler generation request failed")?
            .into_inner())
    }

    pub(crate) fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub(crate) async fn shutdown(mut self) -> Result<()> {
        log::info!("Shutting down the request handler process");
        let shutdown_result = self
            .rpc_client
            .shutdown(Shutdown {})
            .await
            .context("failed to shut down the request handler");
        if let Err(error) = shutdown_result {
            self.child_process.stop()?;
            remove_socket(&self.socket_path)?;
            return Err(error);
        }

        let wait_result = self.child_process.wait();
        let remove_result = remove_socket(&self.socket_path);
        wait_result?;
        remove_result
    }
}

impl Drop for RequestHandlerProcess {
    fn drop(&mut self) {
        if let Err(error) = self.child_process.stop() {
            log::error!("Failed to stop the request handler during cleanup: {error:#}");
        }
        if let Err(error) = remove_socket(&self.socket_path) {
            log::error!("Failed to remove the request handler socket: {error:#}");
        }
    }
}

fn spawn(
    model_runner_socket_path: &Path,
    tokenizer_path: &Path,
    draft_tokenizer_path: Option<&Path>,
    socket_path: &Path,
) -> Result<ChildProcess> {
    let executable = std::env::current_exe().context("failed to locate the current executable")?;
    let mut command = Command::new(executable);
    command
        .process_group(0)
        .env(server::PROCESS_ENVIRONMENT_VARIABLE, "1")
        .arg("--model-runner-socket-path")
        .arg(model_runner_socket_path)
        .arg("--tokenizer-path")
        .arg(tokenizer_path);
    if let Some(draft_tokenizer_path) = draft_tokenizer_path {
        command
            .arg("--draft-tokenizer-path")
            .arg(draft_tokenizer_path);
    }
    let child = command
        .arg("--request-handler-socket-path")
        .arg(socket_path)
        .spawn()
        .context("failed to start the request handler process")?;
    Ok(ChildProcess::new(child, "request handler".to_owned()))
}

fn remove_socket(socket_path: &Path) -> Result<()> {
    match std::fs::remove_file(socket_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).context("failed to remove the request handler socket"),
    }
}
