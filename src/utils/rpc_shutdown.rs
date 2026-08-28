use std::sync::Mutex;
use tokio::sync::oneshot;
use tonic::Status;

pub(crate) enum RpcShutdownError {
    SenderLockPoisoned,
    AlreadyRequested,
    ReceiverDropped,
}

impl From<RpcShutdownError> for Status {
    fn from(error: RpcShutdownError) -> Self {
        match error {
            RpcShutdownError::SenderLockPoisoned => {
                Self::internal("shutdown sender lock is poisoned")
            }
            RpcShutdownError::AlreadyRequested => {
                Self::failed_precondition("shutdown was already requested")
            }
            RpcShutdownError::ReceiverDropped => Self::internal("shutdown receiver was dropped"),
        }
    }
}

/// Allows an RPC handler to request graceful tonic server shutdown once.
pub(crate) struct RpcShutdown {
    sender: Mutex<Option<oneshot::Sender<()>>>,
}

impl RpcShutdown {
    pub(crate) fn channel() -> (Self, oneshot::Receiver<()>) {
        let (sender, receiver) = oneshot::channel();
        (
            Self {
                sender: Mutex::new(Some(sender)),
            },
            receiver,
        )
    }

    pub(crate) fn trigger(&self) -> Result<(), RpcShutdownError> {
        let sender = self
            .sender
            .lock()
            .map_err(|_| RpcShutdownError::SenderLockPoisoned)?
            .take()
            .ok_or(RpcShutdownError::AlreadyRequested)?;
        sender
            .send(())
            .map_err(|_| RpcShutdownError::ReceiverDropped)
    }
}
