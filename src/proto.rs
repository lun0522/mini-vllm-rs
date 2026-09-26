use crate::utils::textproto::parse_textproto;
use std::str::FromStr;

pub(crate) mod model_runner {
    include!(concat!(env!("OUT_DIR"), "/model_runner.rs"));
}

pub(crate) mod request_handler {
    include!(concat!(env!("OUT_DIR"), "/request_handler.rs"));
}

pub(crate) mod main_process {
    include!(concat!(env!("OUT_DIR"), "/main_process.rs"));
}

pub(crate) mod inference_config {
    include!(concat!(env!("OUT_DIR"), "/inference_config.rs"));
}

impl FromStr for inference_config::ModelConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.ModelConfig")
    }
}

impl FromStr for inference_config::DraftModelConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.DraftModelConfig")
    }
}

impl FromStr for inference_config::DraftModelRunnerConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "inference_config.DraftModelRunnerConfig")
    }
}
