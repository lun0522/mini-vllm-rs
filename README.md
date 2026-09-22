# mini-vllm-rs

`mini-vllm-rs` is an educational project for building a fast, lightweight LLM
inference engine in Rust with [Candle](https://github.com/huggingface/candle).
It favors clear implementations of modern serving techniques over production
completeness. The project currently supports macOS only.

Supported features:

- **Quantized GGUF inference** for Qwen2 and Llama, with streaming or buffered
  output and generation statistics.
- **Continuous batching** with dynamic admission, cancellation, chunked prefill,
  and batched prefill, decode, and speculative verification.
- **Speculative decoding** with a tokenizer-compatible draft model, batched
  target verification, and cache commit or rollback.
- **Paged KV-cache management** and **prefix caching** on both CPU and Metal.
  **Paged attention** is currently CPU-only; Metal reconstructs contiguous
  tensors before attention.
- Separate request-handling and model-inference processes.

If you are interested in the project's design, see the
[architecture and process model](ARCHITECTURE.md). It explains the three-process
design, CPU and GPU inference backends, and how the project differs from
mistral.rs, vLLM, and SGLang.

## Benchmarks

Benchmarking is driven by the Python-based
[`mini-vllm-eval`](https://github.com/lun0522/mini-vllm-eval) project. Keeping
the flexible benchmarking harness separate lets `mini-vllm-rs` remain entirely
in Rust for inference performance while Python handles experiment setup,
measurement, and comparison.

- [Metal GEMV threshold](benchmarks/metal_gemv_threshold.md): Measures the
  initial continuous-batching implementation and the effect of the Metal GEMV
  threshold.
- [CPU paged attention](benchmarks/cpu_paged_attention.md): Compares contiguous
  and paged CPU attention using repeated KV, grouped Q, and page-wise V matmul.

## Run

### Start the server

Run the server with the default Qwen2.5 7B Instruct Q4_K_M model on Metal:

```shell
cargo run --release
```

### Server CLI arguments

- `--model '<textproto>'` selects the target model. Set `model_id`,
  `model_filename`, and `tokenizer_id`; optionally set `model_revision`, which
  defaults to `main`.
- `--draft-model '<textproto>'` loads a tokenizer-compatible draft model for
  speculative decoding.
- `--draft-token-count <count>` sets the maximum proposal length and defaults to
  `4`.
- `--inference-device <device>` selects `gpu`, `cpu`, or `mixed` and defaults to
  `gpu`. Mixed mode runs one CPU and one GPU backend concurrently.
- `--activation-dtype <dtype>` selects `f16` or `f32` and defaults to `f32`. On
  macOS, Candle's Metal quantized matmul does not support F16 activations, so
  GPU inference falls back to `f32` with a warning. In mixed mode on macOS, an
  `f16` request uses `f16` on the CPU backend and `f32` on the Metal backend.
- `--kv-cache-type <type>` selects `contiguous`, `paged[:tokens-per-page]`, or
  `paged-prefix[:tokens-per-page]` KV-cache storage and defaults to
  `contiguous`. Paged caches contain 16 tokens per page when the count is
  omitted. `paged-prefix` retains and restores complete shared prefixes and
  evicts least-recently-used inactive prefixes when more pages are needed.
- `--target-kv-cache-size-bytes <bytes>` sets the target model's total KV-cache
  allocation and defaults to 2 GiB. A draft model is allocated enough KV-cache
  memory to hold the same number of tokens.
- `--max-batched-token-count <count>` sets the scheduling work budget and
  defaults to `512`. See
  [Scheduling work budget](ARCHITECTURE.md#scheduling-work-budget).
- `--max-active-request-count <count>` limits the number of requests holding
  active inference state and defaults to `4`. Contiguous KV-cache storage
  limits this to `1`.
- `--scheduling-policy <policy>` selects `first-come-first-served` or
  `shortest-prefill-first` and defaults to `first-come-first-served`.
- `--input-preprocessing-thread-count <count>` sets the request-handler
  preprocessing pool size and defaults to `4`.
- `--request-socket <path>` changes the public request-handler Unix socket and
  defaults to `/tmp/mini-vllm-request-handler.sock`.
- `--control-socket <path>` changes the main-process control socket path. It
  defaults to `/tmp/mini-vllm-main-process.sock` and exposes the `Shutdown` RPC
  defined in `proto/main_process.proto`.

#### Examples

Run the default target with a tokenizer-compatible Qwen2.5 0.5B draft model:

```shell
cargo run --release -- \
  --model 'model_id: "bartowski/Qwen2.5-7B-Instruct-GGUF" model_filename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf" tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"' \
  --draft-model 'model_id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF" model_filename: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" tokenizer_id: "Qwen/Qwen2.5-7B-Instruct"' \
  --draft-token-count 4
```

Run Llama 3.1 8B with a tokenizer-compatible Llama 3.2 1B draft model and
32-token KV-cache pages:

```shell
cargo run --release -- \
  --kv-cache-type paged:32 \
  --model 'model_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" model_filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"' \
  --draft-model 'model_id: "bartowski/Llama-3.2-1B-Instruct-GGUF" model_filename: "Llama-3.2-1B-Instruct-Q4_K_M.gguf" tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"' \
  --draft-token-count 4
```

Run Qwen2.5 0.5B on both CPU and Metal, with requests distributed between the
two inference backends:

```shell
cargo run --release -- \
  --inference-device mixed \
  --model 'model_id: "bartowski/Qwen2.5-0.5B-Instruct-GGUF" model_filename: "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf" tokenizer_id: "Qwen/Qwen2.5-0.5B-Instruct"'
```

### Performance tracing

`--trace-directory` enables Chrome trace export for the model-runner process
and writes `model-runner-<pid>.json` in the specified directory. For example,
when running a benchmark from `mini-vllm-eval`:

```shell
python3 main.py --benchmark simple_generation \
  --trace-directory /tmp/mini-vllm-traces
```

Open the generated JSON file in [Perfetto](https://ui.perfetto.dev).

### Server environment variables

- `RUST_LOG` controls log filtering, for example `RUST_LOG=warn`.
- `CANDLE_NUM_THREADS` controls Candle's CPU worker pool, including quantized
  matrix multiplication.
- `RAYON_NUM_THREADS` controls Rayon-based CPU operations.

On Apple Silicon, Candle defaults to the number of performance-core logical
CPUs, so efficiency cores are not used by default. Tune both thread counts for
the workload; more threads do not always improve throughput.

The remaining variables are evaluated at compile time. Set them when invoking
Cargo.

#### CPU attention

These defaults follow the results in the
[CPU paged-attention benchmark](benchmarks/cpu_paged_attention.md):

- `MINI_VLLM_ENABLE_CPU_GROUPED_QUERY_MATMUL` groups query heads sharing a KV
  head instead of explicitly replicating K and V heads. It defaults to `true`.
- `MINI_VLLM_ENABLE_CPU_PAGED_ATTENTION` computes query-key scores directly
  from KV-cache pages instead of rebuilding a contiguous K tensor. It defaults
  to `false` because contiguous attention is faster in the current benchmark.
- `MINI_VLLM_ENABLE_CPU_PAGEWISE_VALUE_MATMUL` multiplies attention weights by
  each V page separately instead of concatenating the pages. It defaults to
  `false` because concatenated V is faster in the current benchmark.

For example:

```shell
MINI_VLLM_ENABLE_CPU_PAGED_ATTENTION=false \
MINI_VLLM_ENABLE_CPU_GROUPED_QUERY_MATMUL=false \
cargo run --release -- --inference-device cpu --kv-cache-type paged:16
```

#### Metal GEMV

The default threshold follows the crossover measured in the
[Metal GEMV threshold benchmark](benchmarks/metal_gemv_threshold.md):

- `MINI_VLLM_METAL_GEMV_MAX_ROWS` uses separate GEMV operations for Metal
  inputs up to the configured row count. It defaults to `4`; set it to `0` to
  disable the optimization.

### Send a client request

Install [`grpcurl`](https://github.com/fullstorydev/grpcurl) with Homebrew:

```shell
brew install grpcurl
```

Leave the server running in its terminal. In a second terminal, change to the
`mini-vllm-rs` project directory so `-import-path proto` can find the protocol
definitions, then send a protobuf text request over the Unix socket:

```shell
cd /path/to/mini-vllm-rs

grpcurl \
  -plaintext \
  -authority localhost \
  -import-path proto \
  -proto request_handler.proto \
  -format text \
  -d 'prompt: "Explain paged attention briefly." max_new_tokens: 64 repeat_penalty: 1.0 repeat_last_n: 64 stream_output: true' \
  unix:///tmp/mini-vllm-request-handler.sock \
  request_handler.RequestHandlerService/GenerateText
```

The server does not expose gRPC reflection, so the protocol definition is
required. When running the command from another directory, replace
`-import-path proto` with an absolute path such as
`-import-path /path/to/mini-vllm-rs/proto`.

The request and streaming response schemas are defined in
[`proto/request_handler.proto`](proto/request_handler.proto). For a Python
client and benchmark orchestration, see
[`mini-vllm-eval`](https://github.com/lun0522/mini-vllm-eval).

After sending requests, return to the server terminal and press Ctrl-C to shut
it down gracefully.

## End-to-end test

The ignored end-to-end test runs Qwen2.5 0.5B on CPU through the complete
three-process serving path. It downloads the model on first use and reuses the
Hugging Face cache afterward:

```shell
cargo test --release --test end_to_end -- --ignored --nocapture
```

Set `MINI_VLLM_TEST_TRACE_DIRECTORY` to collect a Chrome trace during the test:

```shell
MINI_VLLM_TEST_TRACE_DIRECTORY=/path/to/traces \
cargo test --release --test end_to_end -- --ignored --nocapture
```

The test forwards the directory to the server's `--trace-directory` option.

## Additional documentation

- [Troubleshooting](TROUBLESHOOTING.md)
- [Model licenses](MODEL_LICENSES.md)
