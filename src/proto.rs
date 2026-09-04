pub(crate) mod model_runner {
    include!(concat!(env!("OUT_DIR"), "/model_runner.rs"));
}

pub(crate) mod request_handler {
    include!(concat!(env!("OUT_DIR"), "/request_handler.rs"));
}

pub(crate) mod main_process {
    include!(concat!(env!("OUT_DIR"), "/main_process.rs"));
}

pub(crate) mod model_config {
    include!(concat!(env!("OUT_DIR"), "/model_config.rs"));
}

impl model_config::ModelConfig {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        if self.model_id.is_empty()
            || self.model_filename.is_empty()
            || self.tokenizer_id.is_empty()
        {
            anyhow::bail!("model_id, model_filename, and tokenizer_id must not be empty");
        }
        if !self.model_filename.to_ascii_lowercase().ends_with(".gguf") {
            anyhow::bail!("model filename must identify a .gguf file");
        }
        Ok(())
    }
}
