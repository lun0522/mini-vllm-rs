use crate::utils::child_process::ChildProcess;
use anyhow::Context;
use anyhow::Result;
use hyper_util::rt::TokioIo;
use std::io::ErrorKind;
use std::path::Path;
use std::time::Duration;
use tokio::net::UnixStream;
use tonic::transport::Channel;
use tonic::transport::Endpoint;
use tower::service_fn;

const CONNECTION_TIMEOUT: Duration = Duration::from_secs(1);
const CONNECTION_RETRY_DELAY: Duration = Duration::from_millis(100);

pub(crate) fn ensure_available(socket_path: &Path, socket_name: &str) -> Result<()> {
    if socket_path
        .try_exists()
        .with_context(|| format!("failed to inspect the {socket_name} path"))?
    {
        anyhow::bail!("{socket_name} '{}' already exists", socket_path.display());
    }
    Ok(())
}

pub(crate) fn remove(socket_path: &Path, socket_name: &str) -> Result<()> {
    match std::fs::remove_file(socket_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("failed to remove the {socket_name}")),
    }
}

pub(crate) async fn connect(socket_path: &Path) -> Result<Channel> {
    let connector_path = socket_path.to_owned();
    Endpoint::from_static("http://localhost")
        .connect_timeout(CONNECTION_TIMEOUT)
        .connect_with_connector(service_fn(move |_| {
            let connector_path = connector_path.clone();
            async move { UnixStream::connect(connector_path).await.map(TokioIo::new) }
        }))
        .await
        .with_context(|| {
            format!(
                "failed to connect to Unix domain socket '{}'",
                socket_path.display()
            )
        })
}

pub(crate) async fn wait_for_server(
    child_process: &mut ChildProcess,
    socket_path: &Path,
    startup_timeout: Duration,
) -> Result<Channel> {
    let startup_started = tokio::time::Instant::now();
    let mut last_connection_error = None;

    while startup_started.elapsed() < startup_timeout {
        if let Some(status) = child_process.try_wait()? {
            anyhow::bail!(
                "{} process exited before accepting connections: {status}",
                child_process.name()
            );
        }

        match connect(socket_path).await {
            Ok(channel) => return Ok(channel),
            Err(error) => last_connection_error = Some(error.to_string()),
        }
        tokio::time::sleep(CONNECTION_RETRY_DELAY).await;
    }

    anyhow::bail!(
        "timed out after {startup_timeout:?} waiting for the {} Unix domain socket. Last \
         connection error: {}",
        child_process.name(),
        last_connection_error.as_deref().unwrap_or("none")
    )
}
