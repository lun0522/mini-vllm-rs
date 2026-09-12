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
- ✅ Paged KV-cache management.
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
- ✅ Prefix caching.
  - ✅ Reuse complete KV-cache pages for prefixes shared across requests.
  - ✅ Evict inactive cached prefixes when physical pages are exhausted.
- 🚧 Continuous batching.
  - ✅ Per-request state with dynamic admission, scheduling, and cancellation.
  - ✅ Interleaved execution with token-budgeted chunked prefill.
  - ⬜ Reserve paged KV-cache capacity when admitting and scheduling requests.
  - ⬜ Batched model forwards for prefill and decode.
- ⬜ Performance evaluation.
  - ⬜ Measure latency, throughput, and KV-cache memory usage.
  - ⬜ Compare baseline, continuously batched, and speculative execution.

## Design choices

This project borrows serving ideas from
[vLLM](https://docs.vllm.ai/en/latest/design/arch_overview/),
[SGLang](https://github.com/sgl-project/sglang), and
[mistral.rs](https://docs.mistralrs.dev/developer/architecture/), but does not
reproduce them. It targets one local machine with one CPU or Metal inference
worker rather than tensor, pipeline, or data parallel execution across many
devices. That assumption favors explicit ownership and inspectable control flow
over distributed-worker infrastructure.

### Why three processes?

- **Main process:** downloads artifacts and supervises startup and shutdown. It
  stays off the request data path so lifecycle work does not interfere with
  generation.
- **Request handler:** accepts requests and owns tokenization, detokenization,
  and response streaming. Its worker pool keeps CPU preprocessing away from the
  inference loop.
- **Model runner:** owns model weights, scheduling, KV caches, and device
  execution. Keeping these together avoids IPC on every generation iteration
  and gives mutable device state one clear owner.

The two serving processes exchange requests, token IDs, and statistics over
local Protocol Buffers and Unix sockets—never model tensors. This keeps CPU
tokenization away from the inference loop and gives model memory one clear
owner, while process exit provides a simple way to reclaim model resources.

- [mistral.rs uses one engine thread per model in a single
  binary](https://docs.mistralrs.dev/developer/architecture/#engine-threads),
  minimizing process and IPC overhead. This project accepts a small local IPC
  cost in exchange for explicit readiness, shutdown, and resource ownership.
- [vLLM separates API servers, engine cores, and GPU
  workers](https://docs.vllm.ai/en/latest/design/arch_overview/#v1-process-architecture)
  and adds coordination for multi-GPU and distributed execution.
- [SGLang separates tokenizer, scheduler, and detokenizer
  managers](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/managers)
  and can scale tokenizer and device workers independently.

The names do not map one-to-one: this project's request handler combines
tokenization and detokenization, while its model runner combines scheduling and
device execution. That is enough isolation for a local deployment without
introducing a worker boundary into each generation iteration. See the
architecture READMEs for the
[main process](src/main_process/README.md),
[request handler](src/request_handler/README.md),
[model runner](src/model_runner/README.md), and [models](src/models/README.md).

### What does `max_batched_token_count` mean?

`max_batched_token_count` is a scheduler-work budget:

- One prefill token costs one unit.
- One decode iteration costs one unit.
- A speculative decode iteration may still propose up to the shared
  `draft_token_count`, capped by the request's remaining output allowance.

Every request shares one target, optional draft, and speculative configuration,
so a decode iteration is a stable fairness unit even when the draft model
proposes several tokens. Restricting a budgeted decode iteration to one proposal
would discard much of the benefit of speculative decoding.

The tradeoff is that this setting bounds scheduler work, not exact model FLOPs,
target-plus-draft work, or verification tensor width. Separate limits can be
added if batched execution needs them. vLLM instead distinguishes scheduled
tokens from capacity for speculative slots in its
[scheduler configuration](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/).

### Planned architectural changes

- Make scheduler admission aware of available paged KV-cache capacity.
- Batch scheduled prefill and decode work into model forward calls on one local
  Metal worker.
- Add routing across local CPU and Metal workers for heterogeneous-device
  experiments.
- Add independent model replicas and per-model workers for data parallelism and
  serving multiple small models.
- Implement model kernels with NVIDIA's `cutile-rs`, including paged attention
  that operates directly on cache pages without concatenating tensors, and
  verify the backend on Linux with NVIDIA GPUs.
- Experiment with lightweight speculative decoding algorithms such as n-gram
  speculation.
- Support multimodal models small enough for local execution.

## Run

Run the server on GPU using Metal:

```shell
cargo run --release
```

Run the server on CPU:

```shell
cargo run --release -- --inference-device cpu
```

Input preprocessing uses four request-handler worker threads by default. Use
`--input-preprocessing-thread-count` to change the pool size.

The scheduler can be configured with
`--max-batched-token-count`, `--max-active-request-count`, and
`--scheduling-policy`. It interleaves request execution while model forwards
remain single-request operations.

Optionally, set the `CANDLE_NUM_THREADS` and `RAYON_NUM_THREADS` environment
variables for CPU inference to control the number of CPU worker threads.
`CANDLE_NUM_THREADS` controls Candle's dedicated worker pool, including
quantized matrix multiplication, while `RAYON_NUM_THREADS` controls
Rayon-based operations. On Apple Silicon, Candle defaults to the number of
performance-core logical CPUs, so the efficiency cores are not used by
default. Adjust both values for the machine's CPU; using more threads does not
necessarily improve throughput for every model or workload.

Run the "Qwen2.5 7B Instruct Q4_K_M" target model with the "Qwen2.5 0.5B
Instruct Q4_K_M" draft model for speculative decoding (both use the same
tokenizer so their token IDs remain compatible):

```shell
cargo run --release -- \
  --model 'model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF" model_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf" tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"' \
  --draft-model 'model_id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF" model_filename: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"' \
  --draft-token-count 4
```

Run the "Llama 3.1 8B Instruct Q4_K_M" target model with the "Llama 3.2 1B
Instruct Q4_K_M" draft model for speculative decoding (both use the same
tokenizer so their token IDs remain compatible) and a paged KV cache containing
32 tokens per page:

```shell
cargo run --release -- \
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
- `--inference-device <device>` selects `gpu` or `cpu` and defaults to `gpu`.
- `--kv-cache-type <type>` selects `contiguous`, `paged[:tokens-per-page]`, or
  `paged-prefix[:tokens-per-page]` KV-cache storage and defaults to
  `contiguous`. Paged caches contain 16 tokens per page when the count is
  omitted. `paged-prefix` retains and restores complete shared prefixes and
  evicts least-recently-used inactive prefixes when more pages are needed.
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
