pub(crate) mod model_runner {
    include!(concat!(env!("OUT_DIR"), "/model_runner.rs"));
}

pub(crate) mod request_handler {
    include!(concat!(env!("OUT_DIR"), "/request_handler.rs"));
}

pub(crate) mod main_process {
    include!(concat!(env!("OUT_DIR"), "/main_process.rs"));
}

pub(crate) mod inference_config;
