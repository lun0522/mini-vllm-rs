use crate::proto::ModelConfig;
use crate::utils::textproto::parse_textproto;
use argh::FromArgs;
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

const DEFAULT_DRAFT_TOKEN_COUNT: usize = 4;

/// Runs text generation with a model from Hugging Face.
#[derive(FromArgs)]
pub(crate) struct MainProcessArgs {
    /// textproto configuration for the target GGUF model
    #[argh(option, default = "default_model_config()")]
    pub(crate) model: ModelConfig,
    /// textproto configuration for the speculative-decoding draft model
    #[argh(option)]
    pub(crate) draft_model: Option<ModelConfig>,
    /// number of tokens proposed by the draft model per speculative decoding step
    #[argh(option, default = "DEFAULT_DRAFT_TOKEN_COUNT")]
    pub(crate) draft_token_count: usize,
    /// unix domain socket exposed to local inference clients
    #[argh(option, default = "default_request_socket()")]
    pub(crate) request_socket: PathBuf,
    /// submit the built-in example request after startup
    #[argh(switch)]
    pub(crate) run_example: bool,
}

impl fmt::Display for MainProcessArgs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "Model: {}", self.model.model_id)?;
        writeln!(formatter, "GGUF file: {}", self.model.model_filename)?;
        writeln!(formatter, "Tokenizer: {}", self.model.tokenizer_id)?;
        writeln!(formatter, "Revision: {}", self.model.model_revision)?;
        if let Some(model) = &self.draft_model {
            writeln!(formatter, "Draft model: {}", model.model_id)?;
            writeln!(formatter, "Draft GGUF file: {}", model.model_filename)?;
            writeln!(formatter, "Draft tokenizer: {}", model.tokenizer_id)?;
            writeln!(formatter, "Draft revision: {}", model.model_revision)?;
            writeln!(formatter, "Draft token count: {}", self.draft_token_count)?;
        } else {
            writeln!(formatter, "Draft model: disabled")?;
        }
        writeln!(
            formatter,
            "Request socket: {}",
            self.request_socket.display()
        )?;
        write!(formatter, "Run example: {}", self.run_example)
    }
}

impl FromStr for ModelConfig {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_textproto(value, "model_config.ModelConfig")
    }
}

pub(crate) fn parse() -> MainProcessArgs {
    normalize(argh::from_env())
}

fn default_model_config() -> ModelConfig {
    ModelConfig {
        model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF".to_owned(),
        model_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_owned(),
        tokenizer_id: "Qwen/Qwen2.5-7B-Instruct".to_owned(),
        model_revision: "main".to_owned(),
    }
}

fn default_request_socket() -> PathBuf {
    PathBuf::from("/tmp/mini-vllm-rs.sock")
}

fn normalize(mut args: MainProcessArgs) -> MainProcessArgs {
    if args.model.model_revision.is_empty() {
        args.model.model_revision = "main".to_owned();
    }
    if let Some(draft_model) = args.draft_model.as_mut() {
        if draft_model.model_revision.is_empty() {
            draft_model.model_revision = "main".to_owned();
        }
        if args.draft_token_count == 0 {
            args.draft_token_count = DEFAULT_DRAFT_TOKEN_COUNT;
        }
    }
    args
}
