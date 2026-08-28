pub(crate) mod model_runner {
    include!(concat!(env!("OUT_DIR"), "/model_runner.rs"));
}

pub(crate) mod request_handler {
    include!(concat!(env!("OUT_DIR"), "/request_handler.rs"));
}

pub(crate) use model_runner::*;
