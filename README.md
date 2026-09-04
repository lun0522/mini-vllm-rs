# mini-vllm-rs

`mini-vllm-rs` is an educational project for building a fast, lightweight LLM
inference engine in Rust with [Candle](https://github.com/huggingface/candle).
It favors clear implementations of modern serving techniques over production
completeness. The roadmap shows what the project supports and where it is
heading. The project currently supports macOS only.

Status: ✅ done · 🚧 in progress · ⬜ not started · ❌ out of scope

- ✅ Core inference serving.
  - ✅ End-to-end quantized GGUF inference for Qwen2 and Llama on a single
    inference thread.
  - ✅ Streaming or buffered output with generation statistics.
  - ✅ Separate request-handling and model-inference processes.
- ✅ Paged attention.
  - ✅ Preallocated, engine-owned KV caches passed into model forward calls.
  - ✅ Fixed-size KV-cache pages with per-layer allocation and block tables.
  - ❌ Attention over paged caches without rebuilding contiguous tensors.
- ✅ Speculative decoding.
  - ✅ Draft-model loading with tokenizer compatibility and vocabulary coverage
    validation.
  - ✅ Draft proposal and batched target verification with cache commit and
    rollback.
  - ❌ Randomized sampling with distribution-preserving probabilistic draft
    verification.
- ⬜ Prefix caching.
  - ⬜ Reuse KV-cache pages for prompt prefixes shared across requests.
- ⬜ Continuous batching.
  - ⬜ Per-request state with dynamic admission, scheduling, and cancellation.
  - ⬜ Batched prefill and decode with chunked prefill support.
- ⬜ Performance evaluation.
  - ⬜ Measure latency, throughput, and KV-cache memory usage.
  - ⬜ Compare baseline, continuously batched, and speculative execution.

## Process architecture

- The main process downloads model artifacts, manages child-process lifecycles,
  and accepts shutdown requests on its control socket.
- The request-handler process accepts local tonic requests and proxies response
  streams.
- The model-runner process owns the loaded models, inference device, and mutable
  inference state on a dedicated thread.
- Processes communicate through Protocol Buffers over Unix domain sockets.
- See [Main process architecture](src/main_process/README.md) for startup and
  coordinated-shutdown details,
  [Request handler architecture](src/request_handler/README.md) for text
  preprocessing and event-flow details,
  [Model runner architecture](src/model_runner/README.md) for inference
  request-flow diagrams, and [Model loaders](src/model_loaders/README.md) for
  model and cache details.

The main process is a supervisor rather than a serving stage, so there are two
serving processes today. Unlike
[vLLM](https://docs.vllm.ai/en/latest/design/arch_overview/), the model runner
keeps scheduling, KV-cache management, and device execution together. A third
serving process would add an IPC exchange to every inference step without a
clear benefit while there is only one device worker.

The following architectural changes are on the way:

- Move chat formatting, tokenization, and incremental decoding into bounded
  worker threads in the request-handler process, and use token-level RPCs with
  the model runner.
- Add a scheduler component inside the model-runner process and implement
  continuous batching with one local Metal worker.
- Define a narrow model-worker interface before adding more execution backends.
- Add routing across local CPU and Metal workers for heterogeneous-device
  experiments.
- Add independent model replicas and per-model workers for data parallelism and
  serving multiple small models.
- Move workers into separate processes only when multiple devices, model
  unloading, or fault isolation justify the additional boundary.

## Run

Run the server on CPU:

```shell
cargo run --release
```

Optionally, set the `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS` environment
variables for this command to control the number of CPU worker threads:

```shell
CANDLE_NUM_THREADS=8 RAYON_NUM_THREADS=8 cargo run --release
```

`CANDLE_NUM_THREADS` controls Candle's dedicated worker pool, including
quantized matrix multiplication, while `RAYON_NUM_THREADS` controls
Rayon-based operations. On Apple Silicon, Candle defaults to the number of
performance-core logical CPUs, so the efficiency cores are not used by
default. Adjust both values for the machine's CPU; using more threads does not
necessarily improve throughput for every model or workload.

Run it with Metal acceleration on Apple Silicon:

```shell
cargo run --release --features metal
```

Run the "Qwen2.5 7B Instruct Q4_K_M" target model with the "Qwen2.5 0.5B
Instruct Q4_K_M" draft model for speculative decoding (both use the same
tokenizer so their token IDs remain compatible):

```shell
cargo run --release --features metal -- \
  --model 'model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF" model_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf" tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"' \
  --draft-model 'model_id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF" model_filename: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"' \
  --draft-token-count 4
```

Run the "Llama 3.1 8B Instruct Q4_K_M" target model with the "Llama 3.2 1B
Instruct Q4_K_M" draft model for speculative decoding (both use the same
tokenizer so their token IDs remain compatible) and a paged KV cache containing
32 tokens per page:

```shell
cargo run --release --features metal -- \
  --kv-cache-type paged:32 \
  --model 'model_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" model_filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"' \
  --draft-model 'model_id: "bartowski/Llama-3.2-1B-Instruct-GGUF" model_filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf" tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"' \
  --draft-token-count 4
```

Arguments:

- `--model '<textproto>'` selects the target model. Set `model_id`,
  `model_filename`, and `tokenizer_id`; optionally set `model_revision`, which
  defaults to `main`.
- `--draft-model '<textproto>'` loads a tokenizer-compatible draft model for
  speculative decoding.
- `--draft-token-count <count>` sets the proposal length and defaults to `4`.
- `--kv-cache-type <type>` selects `contiguous`, `paged[:tokens-per-page]`, or
  `paged-prefix[:tokens-per-page]` KV-cache storage and defaults to
  `contiguous`. Paged caches contain 16 tokens per page when the count is
  omitted. The `paged-prefix` value currently enables configuration plumbing;
  prefix reuse will be added separately.
- `--target-kv-cache-size-bytes <bytes>` sets the target model's total KV-cache
  allocation and defaults to 2 GiB. A draft model is allocated enough KV-cache
  memory to hold the same number of tokens.
- `--request-socket <path>` changes the request-handler Unix socket path.
- `--control-socket <path>` changes the main-process control socket path. It
  defaults to `/tmp/mini-vllm-main-process.sock` and exposes the `Shutdown` RPC
  defined in `proto/main_process.proto`.

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

### Metal produces repetitive or nonsensical output after loading a draft model

Loading a draft model increases unified-memory usage even before speculative
decoding uses it: both models' weights and their preallocated KV caches remain
resident, alongside activations and Metal working buffers. Under severe memory
pressure, Metal may report an allocation or command-buffer error, but an error
is not guaranteed. Buffer allocation can succeed before the working set is
fully exercised, and inference may instead produce numerically corrupted logits
that appear as repetitive punctuation, a repeated token, or otherwise
nonsensical text.

If target-model output is correct without `--draft-model` but becomes corrupted
when the draft model is loaded, treat memory pressure as the first suspect. Try
omitting the draft model or using smaller target and draft models. You can also
open macOS Activity Monitor, select the Memory tab, and watch the Memory
Pressure graph and Swap Used while loading the models and generating text. A
yellow or red graph, or rapidly increasing swap usage, supports the memory-
pressure diagnosis. Lowering `--target-kv-cache-size-bytes` can confirm the
diagnosis. Changing the number of tokens per page does not materially reduce
the requested KV-cache budget.

## Model licenses

This project's source code is licensed under the MIT License. Model weights are
downloaded separately and remain subject to their respective licenses. Using or
distributing this software does not grant rights to any model weights. Review
and comply with the license and usage terms for each model before using it.

The Qwen target and draft models are licensed under Apache 2.0:

- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

The Meta Llama target and draft models use their respective community licenses:

- [`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
- [`bartowski/Llama-3.2-1B-Instruct-GGUF`](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF)

If model weights are bundled or redistributed, include the model's license and
any required attribution or notices with the distribution. Models added in the
future may use different licenses or require accepting additional usage terms.
