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
        InferenceWorker["server/inference_worker.rs<br/>Inference thread and model ownership"]
        TextGeneration["server/text_generation.rs<br/>Autoregressive decoding loop"]
    end

    Client -->|"Spawns with local paths and socket"| Cli
    Cli --> Server
    InferenceWorker --> KvCache
    Server -->|"Bounded request channel"| InferenceWorker
    InferenceWorker --> TextGeneration
```

- `client.rs` checks that the socket path is available, starts the worker, waits
  for the worker to bind the socket, and sends the shutdown command.
- `server/cli.rs` receives target and optional draft GGUF paths from the main
  process.
- [`server/kv_cache/`](server/kv_cache/README.md) preallocates separate key/value pools for contiguous or
  paged storage. Paged mode uses configurable fixed-token-count pages and
  per-layer block tables, and reconstructs contiguous tensors for the existing
  attention operations. Its physical page pool owns tensor storage and free
  page IDs, while its active block tables only map the current sequence to
  those physical pages. Reference counts keep shared pages allocated until
  both active sequences and cached-prefix entries release them.
- Prefix-enabled paged caches create a prefix-block index. It indexes only
  complete blocks and includes the preceding block in each identity so equal
  token blocks from different prompt contexts cannot share incompatible pages.
- `server/inference_worker.rs` owns the target model, optional draft model,
  device, and corresponding KV caches on its dedicated thread. The
  target cache uses the configured byte budget; the draft cache is sized to
  hold the same number of tokens.
- `server/text_generation.rs` performs prompt prefill, ordinary greedy decode,
  or speculative decode using draft proposals, batched target verification,
  cache rollback, and request-level acceptance statistics.
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
    participant Worker as inference_worker.rs
    participant Decode as text_generation.rs
    participant Target as Target model / Candle
    participant Draft as Optional draft model / Candle

    Caller->>Handler: GenerateText request
    Handler->>Rpc: Forward GenerateText over tonic/UDS
    Rpc->>Worker: Queue InferenceRequest
    Worker->>Worker: Restore each model's longest cached prompt prefix
    Worker->>Decode: Generate text with loaded model(s)
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
        Decode-->>Worker: Push generated token ID
        Worker-->>Rpc: Queue GenerateTextEvent::TokenId
        Rpc-->>Handler: Stream token-ID event
        Handler->>Handler: Incrementally decode token ID
        Handler-->>Caller: Stream text event
    end
    Decode-->>Worker: Return TextGenerationStats
    Worker->>Worker: Index complete cached blocks and release active references
    Worker-->>Rpc: Queue GenerateTextEvent::Stats
    Rpc-->>Handler: Stream final statistics
    Handler-->>Caller: Proxy final statistics
```

- `server/mod.rs` rejects inputs that cannot fit the configured KV-cache
  capacity, limits the requested output length to the remaining capacity, then
  queues valid tonic requests on a bounded channel.
- `inference_worker.rs` owns the loaded model and processes requests on its
  dedicated thread. It restores reusable prefixes before generation, indexes
  complete cached blocks after success, and releases active references after
  either success or failure.
- `inference_worker.rs` delegates decoding to `text_generation.rs`.
- `text_generation.rs` runs prefill and decode over token IDs, samples tokens,
  checks cancellation and stop tokens, and records prefill, decode, and
  speculative acceptance statistics.
- The request handler decodes generated token IDs. It streams text fragments
  immediately unless `stream_output` is false, in which case it buffers them.
- A successful response ends with a `TextGenerationStats` event.
