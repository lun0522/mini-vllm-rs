# mini-vllm-rs

A small Rust project that will grow into a continuous-batching LLM server. The
first milestone downloads a compact model from Hugging Face and performs basic
autoregressive inference with [Candle](https://github.com/huggingface/candle).

The example uses `HuggingFaceTB/SmolLM2-360M-Instruct`, a compact
instruction-tuned text model whose Llama architecture is supported directly by
`candle-transformers`.
Model files are downloaded on the first run and reused from the Hugging Face
cache afterward. A `ModelDownloader` owns that disk-acquisition step, while
`LoadedModel` only loads the resulting local files. The initialized model,
tokenizer, device, and data type are owned by a reusable `LoadedModel`, so
additional inference calls do not reload the checkpoint. Model-specific Candle
code is isolated behind a common causal language-model backend interface. Both
Llama models such as SmolLM2 and Qwen2 models such as
`Qwen/Qwen2.5-0.5B-Instruct` use that interface without changing inference
control flow. The main process downloads the model and starts separate request
handler and model runner processes. Commands cross process boundaries as
Protocol Buffers messages over tonic and Unix domain sockets, so the model
runner does not need internet access.

## Process architecture

During inference, the program uses three operating-system processes. All run
the same executable, but internal environment markers select the request-handler
and model-runner modes for the child processes.

```text
main process
 ├─ parses the public model ID and revision arguments
 ├─ downloads model artifacts or reuses the Hugging Face cache
 ├─ starts the model runner and request handler processes
 ├─ optionally submits the built-in example request
 ├─ waits for Ctrl-C
 └─ shuts down both child processes in reverse dependency order

request handler process
 ├─ listens for local tonic requests on the public Unix domain socket
 ├─ accepts concurrent GenerateText requests
 ├─ forwards inference requests to the model runner
 └─ exits after receiving its Shutdown RPC

model runner process
 ├─ parses local model-file and socket arguments
 ├─ loads the model from disk without accessing the internet
 ├─ starts the tonic Unix domain socket server
 ├─ queues inference requests from tonic handlers
 ├─ runs a dedicated inference thread that owns the loaded model
 └─ exits after receiving the Shutdown command
```

The main process owns downloading and both child-process lifecycles. The request
handler owns the client-facing local endpoint, while the model runner owns the
initialized model, tokenizer, inference device, and mutable inference state on
its inference thread. This boundary allows multiple inference commands to reuse
one loaded model. The request queue can later feed a continuous-batching
scheduler or multiple device-specific workers without changing the RPC layer.

## Run

CPU (portable, but slower):

```shell
cargo run --release
```

Apple Silicon with Metal acceleration:

```shell
cargo run --release --features metal
```

The default model is `Qwen/Qwen2.5-0.5B-Instruct`. Select another supported
model with `--model-id`:

```shell
cargo run --release -- --model-id HuggingFaceTB/SmolLM2-360M-Instruct
```

Select a branch, tag, or commit with `--model-revision`; it defaults to `main`:

```shell
cargo run --release -- --model-revision <revision>
```

The request handler listens on `/tmp/mini-vllm-rs.sock` by default. Select a
different local Unix domain socket with `--request-socket`:

```shell
cargo run --release -- --request-socket /tmp/my-mini-vllm.sock
```

The processes remain active until Ctrl-C. To submit the built-in example after
startup, pass `--run-example`:

```shell
cargo run --release -- --run-example
```

The built-in example settings are currently initialized in `main()`. They
ask the model to explain continuous batching in detail, using a ChatML-style
prompt, greedy decoding with a repetition penalty, and a limit of 1,024 new tokens.
Generation stops early when the model emits a chat end token. The program logs
the effective model, revision, device, data type, and prompt, streams generated
text to the console token by token, then logs elapsed time and throughput. Set
`stream_output` to `false` in the inference settings to buffer and log the
complete response only after generation finishes.
Informational logs are enabled by default and can be filtered with `RUST_LOG`
(for example, set `RUST_LOG=warn`). Expect the first run to download less than
1 GB of model data.

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

Inference runs in a dedicated worker process spawned from the Rust binary.
Server-side scheduling and continuous batching can be added in Rust as the
project grows; Python clients and load generators can live alongside it without
changing the Cargo layout.
