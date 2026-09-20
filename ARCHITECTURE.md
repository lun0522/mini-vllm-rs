# Architecture

The name `mini-vllm-rs` reflects the project's goal: a compact, Rust-native
implementation of modern LLM serving techniques popularized by systems such as
vLLM. It is an independent project, not a reduced version or port of vLLM.

The project is designed for one local machine. It can run one CPU worker, one
GPU worker, or one of each, while using three operating-system processes:

```text
client -> request handler -> model runner
                            (CPU and/or GPU threads)
              ^                  ^
              |                  |
              +---- main process-+
                   supervises
```

1. The **main process** downloads and validates artifacts, starts the serving
   processes in dependency order, exposes the shutdown control service, and
   supervises cleanup. It leaves the request data path after startup.
2. The **request-handler process** owns the public inference service,
   tokenization, detokenization, and response streaming. A preprocessing thread
   pool prevents CPU-side prompt preparation from blocking other requests.
3. The **model-runner process** owns the model weights, schedulers, request
   generation state, KV caches, and device execution. It contains one inference
   backend in CPU-only or GPU-only mode. Mixed mode creates two independent
   backends on separate threads, one for CPU and one for GPU. Each backend has
   its own scheduler, loaded target and optional draft model, and KV caches.
   Whole requests are routed round-robin and continuously batched within their
   assigned backend; a request is not divided across devices.

The serving processes exchange requests, token IDs, and statistics (rather than
model tensors) using Protocol Buffers over local Unix sockets. The model runner
starts first; its socket is its readiness signal. The request handler then
connects to it before exposing its own socket. Shutdown proceeds in the reverse
order.

This arrangement gives each backend's model and device memory a clear owner,
keeps text processing away from the inference loops, and makes process exit a
reliable resource-cleanup boundary. At the same time, each scheduler stays with
its device execution, so there is no inter-process round trip for every
generation iteration.

## Scheduling work budget

`max_batched_token_count` limits the work selected in one scheduling iteration:

- Each prefill token costs one unit.
- Each decode iteration costs one unit.
- A speculative decode iteration costs one unit even though it may propose up
  to `draft_token_count` tokens, capped by the request's remaining output
  allowance.

All requests assigned to a backend share its target model, optional draft
model, and speculative-decoding configuration. Treating one decode iteration as
the scheduling unit therefore provides stable fairness while preserving the
benefit of verifying several draft tokens together.

This setting bounds scheduler work, not exact model FLOPs, combined target and
draft work, or verification tensor width. More limits could be introduced if
those distinctions become important. vLLM instead distinguishes scheduled
tokens from capacity reserved for speculative slots in its
[scheduler configuration](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/).

## Comparison with other inference engines

Process counts below describe the common serving configuration. Each project
also supports modes that alter the topology.

### mistral.rs

[mistral.rs](https://docs.mistralrs.dev/developer/architecture/) organizes the
runtime into server, engine, and pipeline layers. It starts one engine
**thread** per loaded model; requests reach that thread over a channel, and its
scheduler batches concurrent requests. This keeps the serving stack compact and
makes each model an independent concurrency unit without requiring a process
boundary between request handling and inference.

That choice fits mistral.rs's goal of providing both an embeddable Rust library
and a standalone server with broad model and API support. `mini-vllm-rs` instead
uses process boundaries so orchestration, text processing, and inference can in
be started and stopped independently and could support independent restart in
the future. Like mistral.rs, its request and inference paths remain native Rust.

### vLLM

[vLLM V1](https://docs.vllm.ai/en/latest/design/arch_overview/#v1-process-architecture)
uses a multi-process topology aimed at high-throughput, scalable accelerator
serving:

- **API-server processes** handle HTTP, input processing, and output streaming.
- **Engine-core processes**, normally one per data-parallel rank, schedule work,
  manage KV caches, and coordinate model execution.
- **GPU-worker processes**, one per GPU, load weights and execute the model.
- An optional **data-parallel coordinator process** handles load balancing and
  synchronization when data parallelism is enabled.

A default single-GPU deployment therefore has one API server, one engine core,
and one GPU worker. The counts grow with the parallel configuration.

Those boundaries let vLLM scale HTTP/input processing, scheduling, and model
execution independently and map workers cleanly onto accelerator ranks.
`mini-vllm-rs` has no distributed ranks to coordinate, so each of its inference
backends combines the *engine-core* and *device-worker* roles in one thread.
Mixed mode runs its CPU and GPU backends on different threads within the same
model-runner process. The project is not intended to scale beyond these two
concurrent local backends, so separate backend processes would add RPC overhead
without providing a needed distributed-execution boundary.

### SGLang

[SGLang's serving runtime](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/managers)
uses another multi-process topology:

- A **tokenizer-manager process** handles requests and input tokenization.
- **Scheduler processes** own model execution; their number depends on the
  tensor- and data-parallel configuration.
- A **detokenizer-manager process** converts generated token IDs into output.
- Optional processes support features such as data-parallel attention, overlap
  scheduling, and disaggregated prefill and decode.

This topology supports SGLang's emphasis on high-throughput serving and flexible
parallel or disaggregated deployments. `mini-vllm-rs` uses the same broad
separation between text handling and model execution, but combines tokenization
and detokenization in one request-handler process and combines scheduling with
device execution in one model-runner process.

For local execution, these three total processes retain the useful ownership
boundaries; mixed mode adds a second heterogeneous worker inside the
model-runner process without importing distributed-worker coordination. Unlike
the Python control paths in vLLM and SGLang, request routing and scheduling
remain in native Rust and avoid Python interpreter overhead.

## Detailed documentation

- [Main process](src/main_process/README.md)
- [Request handler](src/request_handler/README.md)
- [Model runner](src/model_runner/README.md)
- [KV cache](src/model_runner/server/kv_cache/README.md)
- [Models](src/models/README.md)

## Future directions

- Make scheduler admission aware of available paged KV-cache capacity.
- Verify the backend on Linux with NVIDIA GPUs and integrate
  `candle-flash-attn` for paged attention.
- Support multimodal models small enough for local execution.
