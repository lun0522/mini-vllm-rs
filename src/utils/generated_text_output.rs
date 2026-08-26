use anyhow::Context;
use anyhow::Result;
use log::info;
use std::io::Write;

/// Receives decoded model fragments and presents the completed generation.
pub(crate) trait GeneratedTextOutput {
    /// Prepares the output before the first generated fragment arrives.
    fn start(&self);

    /// Handles the next decoded fragment from the model.
    fn push_fragment(&mut self, fragment: &str) -> Result<()>;

    /// Completes the output before subsequent log messages are emitted.
    fn finish(&self) -> Result<()>;
}

/// Creates either an incremental stdout writer or a buffered log writer.
pub(crate) fn create_generated_text_output(enable_streaming: bool) -> Box<dyn GeneratedTextOutput> {
    if enable_streaming {
        Box::new(StreamingGeneratedTextOutput)
    } else {
        Box::new(BufferedGeneratedTextOutput::new())
    }
}

struct StreamingGeneratedTextOutput;

impl GeneratedTextOutput for StreamingGeneratedTextOutput {
    fn start(&self) {
        info!("Output:");
    }

    fn push_fragment(&mut self, fragment: &str) -> Result<()> {
        stream_generated_text(fragment)
    }

    fn finish(&self) -> Result<()> {
        finish_generated_text_stream()
    }
}

struct BufferedGeneratedTextOutput {
    buffer: String,
}

impl BufferedGeneratedTextOutput {
    fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }
}

impl GeneratedTextOutput for BufferedGeneratedTextOutput {
    fn start(&self) {}

    fn push_fragment(&mut self, fragment: &str) -> Result<()> {
        self.buffer.push_str(fragment);
        Ok(())
    }

    fn finish(&self) -> Result<()> {
        info!("Output:\n{}", self.buffer);
        Ok(())
    }
}

fn stream_generated_text(text: &str) -> Result<()> {
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    write!(stdout, "{text}").context("failed to stream generated text")?;
    stdout.flush().context("failed to flush generated text")
}

fn finish_generated_text_stream() -> Result<()> {
    stream_generated_text("\n")
}
