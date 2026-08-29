# mini-vllm-rs

`mini-vllm-rs` is an educational project for building a fast, lightweight LLM
inference engine in Rust with [Candle](https://github.com/huggingface/candle).
It favors clear implementations of modern serving techniques over production
completeness. The roadmap shows what the project supports and where it is
heading. The project currently supports macOS only.

Status: ✅ done · 🚧 in progress · ⬜ not started

- ✅ Core inference serving.
  - ✅ End-to-end quantized GGUF inference for Qwen2 and Llama on a single
    inference thread.
  - ✅ Streaming or buffered output with generation statistics.
  - ✅ Separate request-handling and model-inference processes.
- 🚧 Paged attention.
  - ✅ Preallocated, engine-owned KV caches passed into model forward calls.
  - 🚧 Fixed-size KV-cache blocks with per-request allocation and block tables.
  - ⬜ Attention over paged caches without rebuilding contiguous tensors.
- ⬜ Continuous batching.
  - ⬜ Per-request state with dynamic admission, scheduling, and cancellation.
  - ⬜ Batched prefill and decode with chunked prefill support.
- ⬜ Speculative decoding.
  - ✅ Draft-model loading with tokenizer compatibility and vocabulary coverage
    validation.
  - ⬜ Draft proposal and target verification with cache commit and rollback.
- ⬜ Performance evaluation.
  - ⬜ Measure latency, throughput, and KV-cache memory usage.
  - ⬜ Compare baseline, continuously batched, and speculative execution.

## Process architecture

- The main process downloads model artifacts and manages child-process
  lifecycles.
- The request-handler process accepts local tonic requests and proxies response
  streams.
- The model-runner process owns the loaded models, inference device, and mutable
  inference state on a dedicated thread.
- Processes communicate through Protocol Buffers over Unix domain sockets.
- See [Model runner architecture](src/model_runner/README.md) for request-flow
  diagrams and [Model loaders](src/model_loaders/README.md) for model and cache
  details.

## Run

Run the built-in example on CPU:

```shell
cargo run --release -- --run-example
```

Run it with Metal acceleration on Apple Silicon:

```shell
cargo run --release --features metal -- --run-example
```

Run the example with a different GGUF model:

```shell
cargo run --release -- \
  --run-example \
  --model 'model_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" model_filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"'
```

Arguments:

- `--run-example` submits the built-in request after startup. Without it, the
  server waits for requests on `/tmp/mini-vllm-rs.sock` until Ctrl-C.
- `--model '<textproto>'` selects the target model. Set `model_id`,
  `model_filename`, and `tokenizer_id`; optionally set `model_revision`, which
  defaults to `main`.
- `--draft-model '<textproto>'` loads a tokenizer-compatible draft model for
  future speculative decoding.
- `--draft-token-count <count>` sets the future proposal length and defaults to
  `4`.
- `--request-socket <path>` changes the request-handler Unix socket path.

Notes:

- The default model is "Qwen2.5 7B Instruct Q4_K_M" and is approximately 4.7 GB.
- Downloads are reused from the Hugging Face cache.
- Set `RUST_LOG` to change log filtering, for example `RUST_LOG=warn`.

## Troubleshooting

### A gated tokenizer returns HTTP 401

The Llama example downloads its GGUF weights from the public
`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` repository, but downloads
`tokenizer.json` from the gated `meta-llama/Meta-Llama-3.1-8B-Instruct`
repository. Without approved access and local authentication, startup fails
with an error similar to:

```text
tokenizer model 'meta-llama/Meta-Llama-3.1-8B-Instruct' does not provide
tokenizer.json, or the file could not be downloaded: status code 401
```

To fix it:

1. Sign in to Hugging Face and open
   [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
2. Accept the model terms and wait for access approval if it is not immediate.
3. Create a Hugging Face user token with read access.
4. Save the token where this project's `hf-hub` client can read it:

   ```shell
   hf auth login
   ```

5. Confirm that the CLI uses the approved account, then rerun the original
   command:

   ```shell
   hf auth whoami
   ```

If the `hf` command is unavailable, install the
[official Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

## Model licenses

This project's source code is licensed under the MIT License. Model weights are
downloaded separately and remain subject to their respective licenses. Using or
distributing this software does not grant rights to any model weights. Review
and comply with the license and usage terms for each model before using it.

The Qwen model is licensed under Apache 2.0:

- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

Meta Llama 3.1 uses the [Llama 3.1 Community
License](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct):

- [`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

If model weights are bundled or redistributed, include the model's license and
any required attribution or notices with the distribution. Models added in the
future may use different licenses or require accepting additional usage terms.
