# mini-vllm-rs

A small Rust project that will grow into a continuous-batching LLM server. The
first milestone downloads a compact model from Hugging Face and performs basic
autoregressive inference with [Candle](https://github.com/huggingface/candle).

The example uses `Qwen/Qwen2.5-0.5B-Instruct`, a compact instruction-tuned text
model whose Qwen2 architecture is supported directly by `candle-transformers`.
Model files are downloaded on the first run and reused from the Hugging Face
cache afterward.

## Run

CPU (portable, but slower):

```shell
cargo run --release
```

Apple Silicon with Metal acceleration:

```shell
cargo run --release --features metal
```

The inference settings are currently initialized in `main()`. They select the
model to explain continuous batching in detail, using Qwen's chat template,
greedy decoding with a repetition penalty, and a limit of 1,024 new tokens.
Generation stops early when the model emits a Qwen end token. The program logs
the effective model, revision, device, data type, and prompt, streams generated
text to the console token by token, then logs elapsed time and throughput. Set
`stream_output` to `false` in the inference settings to buffer and log the
complete response only after generation finishes.
Informational logs are enabled by default and can be filtered with `RUST_LOG`
(for example, `RUST_LOG=warn cargo run --release`). Expect the first run to
download roughly 1 GB of model data.

## Project direction

Inference currently lives in a single Rust binary. Server-side scheduling and
continuous batching can be added in Rust as the project grows; Python clients
and load generators can live alongside it without changing the Cargo layout.
