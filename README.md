# mini-vllm-rs

A small Rust project that will grow into a continuous-batching LLM server. The
first milestone downloads a compact model from Hugging Face and performs basic
autoregressive inference with [Candle](https://github.com/huggingface/candle).

The example uses `HuggingFaceTB/SmolLM2-360M-Instruct`, a compact
instruction-tuned text model whose Llama architecture is supported directly by
`candle-transformers`.
Model files are downloaded on the first run and reused from the Hugging Face
cache afterward. The initialized model, tokenizer, device, and data type are
owned by a reusable `LoadedModel`, so additional inference calls do not reload
the checkpoint. Model-specific Candle code is isolated behind a common causal
language-model backend interface. Both Llama models such as SmolLM2 and Qwen2
models such as `Qwen/Qwen2.5-0.5B-Instruct` use that interface without changing
inference control flow.

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
model to explain continuous batching in detail, using a ChatML-style prompt,
greedy decoding with a repetition penalty, and a limit of 1,024 new tokens.
Generation stops early when the model emits a chat end token. The program logs
the effective model, revision, device, data type, and prompt, streams generated
text to the console token by token, then logs elapsed time and throughput. Set
`stream_output` to `false` in the inference settings to buffer and log the
complete response only after generation finishes.
Informational logs are enabled by default and can be filtered with `RUST_LOG`
(for example, `RUST_LOG=warn cargo run --release`). Expect the first run to
download less than 1 GB of model data.

## Model licenses

This project's source code is licensed under the MIT License. Model weights are
downloaded separately and remain subject to their respective licenses. Using or
distributing this software does not grant rights to any model weights. Review
and comply with the license and usage terms for each model before using it.

The models currently supported by the example are licensed under Apache 2.0:

- [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

If model weights are bundled or redistributed, include the model's license and
any required attribution or notices with the distribution. Models added in the
future may use different licenses or require accepting additional usage terms.

## Project direction

Inference currently lives in a single Rust binary. Server-side scheduling and
continuous batching can be added in Rust as the project grows; Python clients
and load generators can live alongside it without changing the Cargo layout.
