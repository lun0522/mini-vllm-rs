# mini-vllm-rs

A small Rust project that will grow into a continuous-batching LLM server. The
first milestone downloads a quantized GGUF model from Hugging Face and performs
basic autoregressive inference with [Candle](https://github.com/huggingface/candle).

Only quantized GGUF checkpoints are supported. The default is the Q4_K_M
variant of Qwen2.5 7B Instruct. The selected GGUF and its `tokenizer.json` are
downloaded on the first run and reused from the Hugging Face cache afterward.
A `ModelDownloader` owns that disk-acquisition step, while `LoadedModel` only
loads the resulting local artifacts. GGUF metadata selects the Qwen2 or Llama
quantized Candle backend and the matching chat prompt format. The main process
downloads the artifacts and starts separate request-handler and model-runner
processes. Commands cross process boundaries as
Protocol Buffers messages over tonic and Unix domain sockets, so the model
runner does not need internet access.

## Process architecture

During inference, the program uses three operating-system processes. All run
the same executable, but internal environment markers select the request-handler
and model-runner modes for the child processes.

```text
main process
 ├─ parses the GGUF, tokenizer, and revision arguments
 ├─ downloads the GGUF and tokenizer or reuses the Hugging Face cache
 ├─ starts the model runner and request handler processes
 ├─ optionally submits the built-in example request
 ├─ waits for Ctrl-C
 └─ shuts down both child processes in reverse dependency order

request handler process
 ├─ listens for local tonic requests on the public Unix domain socket
 ├─ accepts concurrent GenerateText requests
 ├─ forwards inference requests to the model runner
 ├─ proxies generated-text and statistics streams back to requesters
 └─ exits after receiving its Shutdown RPC

model runner process
 ├─ parses local GGUF, tokenizer, and socket arguments
 ├─ loads the quantized model from disk without accessing the internet
 ├─ starts the tonic Unix domain socket server
 ├─ queues inference requests from tonic handlers
 ├─ runs a dedicated inference thread that owns the loaded model
 ├─ streams generated text followed by final generation statistics
 └─ exits after receiving the Shutdown command
```

The main process owns downloading and both child-process lifecycles. The request
handler owns the client-facing local endpoint, while the model runner owns the
initialized quantized model, tokenizer, inference device, and mutable inference state on
its inference thread. This boundary allows multiple inference commands to reuse
one loaded model. The request queue can later feed a continuous-batching
scheduler or multiple device-specific workers without changing the RPC layer.
Generated text follows the reverse path over tonic: the model runner emits
events to the request handler, which proxies them to the requesting client.
Successful streams end with token counts and prefill/decode durations in
milliseconds. The processes between the model and client do not print or
aggregate streamed text.

## Run

CPU (portable, but slower):

```shell
cargo run --release
```

Apple Silicon with Metal acceleration:

```shell
cargo run --release --features metal
```

The defaults select `Qwen2.5-7B-Instruct-Q4_K_M.gguf` from
[`bartowski/Qwen2.5-7B-Instruct-GGUF`](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF)
and its tokenizer from `Qwen/Qwen2.5-7B-Instruct`. A GGUF repository can contain
many quantizations, so each `--model` textproto identifies the repository,
GGUF filename, tokenizer repository, and optional revision. Select the Llama
3.1 Q4_K_M model with:

```shell
cargo run --release -- \
  --model 'model_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF" model_filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" tokenizer_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"'
```

The tokenizer repository may be gated even when the converted GGUF repository
is public. Set `HF_TOKEN` to a Hugging Face access token after accepting any
applicable model terms.

Set `model_revision: "<revision>"` in the textproto to select a branch, tag, or
commit. It defaults to `main` when omitted.

The optional `--draft-model` accepts the same textproto structure and prepares a
second GGUF model for future speculative decoding. `--draft-token-count`
controls how many tokens it will propose per step and defaults to `4`:

```shell
cargo run --release -- \
  --draft-model 'model_id: "<gguf-repository>" model_filename: "<model.gguf>" tokenizer_id: "<tokenizer-repository>"' \
  --draft-token-count 4
```

The draft GGUF and tokenizer are downloaded by the main process and loaded by
the model runner. Speculative decoding itself is not implemented yet.

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
architecture-appropriate chat prompt, greedy decoding with a repetition penalty,
and a limit of 1,024 new tokens.
Generation stops early when the model emits a chat end token. The program logs
the effective model, GGUF file, revision, device, and prompt. The model runner
streams generated-text events through the request handler, and the main process
uses `GeneratedTextOutput` to print the example token by token. The final event
reports input/output token counts and prefill/decode durations. Set
`stream_output` to `false` in the inference settings to make the model runner
buffer the response and send it as one text event before the statistics event.
Informational logs are enabled by default and can be filtered with `RUST_LOG`
(for example, set `RUST_LOG=warn`). The default Q4_K_M GGUF is approximately
4.7 GB, in addition to the tokenizer.

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

## Project direction

Inference runs in a dedicated worker process spawned from the Rust binary.
Server-side scheduling and continuous batching can be added in Rust as the
project grows; Python clients and load generators can live alongside it without
changing the Cargo layout.
