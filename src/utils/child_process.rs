use anyhow::Context;
use anyhow::Result;
use std::process::Child;
use std::process::ExitStatus;

/// Owns a child process and forcibly stops it if graceful shutdown is skipped.
pub(crate) struct ChildProcess {
    child: Child,
    name: String,
}

impl ChildProcess {
    pub(crate) fn new(child: Child, name: String) -> Self {
        Self { child, name }
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        self.child
            .try_wait()
            .with_context(|| format!("failed to inspect the {} process", self.name))
    }

    pub(crate) fn stop(&mut self) -> Result<()> {
        if self.try_wait()?.is_none() {
            self.child
                .kill()
                .with_context(|| format!("failed to stop the {} process", self.name))?;
            self.child
                .wait()
                .with_context(|| format!("failed to wait for the {} process", self.name))?;
        }
        Ok(())
    }

    pub(crate) fn wait(&mut self) -> Result<()> {
        let status = self
            .child
            .wait()
            .with_context(|| format!("failed to wait for the {} process", self.name))?;
        if !status.success() {
            anyhow::bail!("{} process exited unsuccessfully: {status}", self.name);
        }
        Ok(())
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        if let Err(error) = self.stop() {
            log::error!(
                "Failed to stop the {} process during cleanup: {error:#}",
                self.name
            );
        }
    }
}
