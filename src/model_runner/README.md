# Model runner architecture

The `model_runner` module separates the main-process lifecycle client from the
code that runs inside the model-runner process.

```mermaid
flowchart LR
    subgraph Main[Main process]
        Client["client.rs<br/>Starts and stops the worker"]
    end

    subgraph Worker[Model runner process]
        Server["server/mod.rs<br/>tonic service and request queues"]
        Cli["server/cli.rs<br/>Worker arguments and artifact paths"]
        KvCache["server/kv_cache/<br/>Engine-owned KV-cache implementations"]
        InferenceEngine["server/inference_engine.rs<br/>Request lifecycle orchestration and timing"]
        RequestManager["server/request_manager.rs<br/>Request execution state, timing, and responses"]
        Scheduler["server/scheduler.rs<br/>Queued and active scheduling metadata"]
        ModelRunner["server/model_runner.rs<br/>Request execution across model instances"]
        ModelInstance["server/model_instance.rs<br/>One loaded model and its KV-cache manager"]
        TextGeneration["server/text_generation.rs<br/>Resumable request generation state"]
    end

    Client -->|"Spawns with local paths and socket"| Cli
    Cli --> Server
    Server -->|"Dedicated thread and bounded request channel"| InferenceEngine
    InferenceEngine --> RequestManager
    InferenceEngine --> Scheduler
    InferenceEngine --> ModelRunner
    ModelRunner --> ModelInstance
    ModelInstance --> KvCache
    ModelRunner --> TextGeneration
```

- `client.rs` checks that the socket path is available, starts the worker, waits
  for the worker to bind the socket, and sends the shutdown command.
- `server/cli.rs` receives target and optional draft GGUF paths and the selected
  inference device from the main process.
- [`server/kv_cache/`](server/kv_cache/README.md) preallocates separate key/value pools for contiguous or
  paged storage. Paged mode uses configurable fixed-token-count pages and
  per-layer block tables, and reconstructs contiguous tensors for the existing
  attention operations. Its physical page pool owns tensor storage and free
  page IDs, while its active block tables only map the current sequence to
  those physical pages. Page ownership states keep cached pages allocated and
  track whether active requests are currently reading them.
- Prefix-enabled paged caches create a prefix-block index. It indexes only
  complete blocks and includes the preceding block in each identity so equal
  token blocks from different prompt contexts cannot share incompatible pages.
- `server/model_runner.rs` owns the target model instance and optional draft
  model instance and exposes request start, one-step execution, completion, and
  abortion operations.
- `server/mod.rs` starts the dedicated inference thread and forwards requests
  from its bounded channel to `InferenceEngine`.
- `server/inference_engine.rs` coordinates request storage, scheduling, model
  execution, response events, and elapsed-time tracking.
- `server/request_manager.rs` owns each request's payload, resumable execution
  state, response channel, and lifecycle timing while the scheduler retains
  only the metadata needed to make decisions.
- `server/scheduler.rs` applies the selected scheduling policy and enforces the
  configured active-request and token limits without owning generation state.
- `server/model_instance.rs` keeps each loaded model paired with its cache
  manager and passes request-aware forward contexts into model execution. The
  target cache uses the configured byte budget; the draft cache is sized to
  hold the same number of tokens.
- `server/text_generation.rs` keeps each request's generation progress and
  sampling state resumable between prefill and decode iterations. It supports
  ordinary greedy decode or speculative decode using draft proposals, batched
  target verification, cache rollback, and request-level acceptance statistics.
- Tokenization, tokenizer compatibility checks, and incremental decoding belong
  to the request-handler process. The model runner receives and returns token
  IDs.
- The worker binds its socket after loading the model, so the socket signals
  readiness.
- `client.rs` manages the worker lifecycle; it does not forward inference
  requests.

Once startup is complete, inference requests come from the request-handler
process rather than through `client.rs`:

```mermaid
sequenceDiagram
    participant Caller as Inference client
    participant Handler as Request handler process
    participant Rpc as model_runner/server/mod.rs
    participant Engine as inference_engine.rs
    participant Requests as request_manager.rs
    participant Scheduler as scheduler.rs
    participant Runner as model_runner.rs
    participant Decode as text_generation.rs
    participant Target as Target model / Candle
    participant Draft as Optional draft model / Candle

    Caller->>Handler: GenerateText request
    Handler->>Rpc: Forward GenerateText over tonic/UDS
    Rpc->>Engine: Queue InferenceRequest on the dedicated thread
    Engine->>Requests: Store request payload and response channel
    Engine->>Scheduler: Queue scheduling metadata
    Scheduler-->>Engine: Return newly admitted request IDs
    Engine->>Requests: Start request by ID
    Engine->>Runner: Initialize model and cache state
    Runner->>Runner: Restore each model's longest cached prompt prefix
    Runner-->>Engine: Return resumable execution state
    Engine->>Requests: Store resumable execution state
    Engine->>Scheduler: Request a token-budgeted scheduling decision
    Engine->>Requests: Borrow each selected resumable execution state
    Engine->>Runner: Execute one scheduled generation step
    Runner->>Decode: Generate text with loaded model(s)
    alt Draft model configured
        Decode->>Target: Prefill through second-to-last prompt token
        Decode->>Draft: Prefill through second-to-last prompt token
    else Target-only generation
        Decode->>Target: Prefill prompt and sample first token
    end
    loop Until stop token, limit, or cancellation
        alt Draft model configured
            Decode->>Draft: Generate proposals from pending prompt/output token
            Draft-->>Decode: Proposed token IDs
            Decode->>Target: Verify proposal batch in one forward pass
            Target-->>Decode: Logits for every proposal position
            Decode->>Decode: Accept matching prefix and choose replacement
            Decode->>Target: Truncate rejected cache suffix
            Decode->>Draft: Truncate rejected cache suffix
        else Target-only decode
            Decode->>Target: Forward the pending token
            Target-->>Decode: Next-token logits
        end
        Decode-->>Runner: Push generated token ID
        Runner-->>Engine: Push generated token ID
        Engine-->>Rpc: Queue GenerateTextEvent::TokenId
        Rpc-->>Handler: Stream token-ID event
        Handler->>Handler: Incrementally decode token ID
        Handler-->>Caller: Stream text event
    end
    Decode-->>Runner: Return TextGenerationStats
    Runner->>Runner: Index complete cached blocks and release active references
    Runner-->>Engine: Return completed generation
    Engine-->>Rpc: Queue GenerateTextEvent::Stats
    Rpc-->>Handler: Stream final statistics
    Handler-->>Caller: Proxy final statistics
```

- `server/mod.rs` rejects inputs that cannot fit the configured KV-cache
  capacity, limits the requested output length to the remaining capacity, then
  queues valid tonic requests on a bounded channel.
- `inference_engine.rs` executes scheduled requests, records request-level
  timing, streams generation events, and reports updated generation state to
  the scheduler.
- `model_runner.rs` owns the models and caches, restores reusable prefixes,
  delegates decoding to `text_generation.rs`, and finishes or clears cache
  state after each request.
- `text_generation.rs` runs token-budgeted prefill chunks and decode steps,
  samples tokens, checks cancellation and stop tokens, and records prefill, decode, and
  speculative acceptance statistics.
- The request handler decodes generated token IDs. It streams text fragments
  immediately unless `stream_output` is false, in which case it buffers them.
- A successful response ends with a `TextGenerationStats` event.
