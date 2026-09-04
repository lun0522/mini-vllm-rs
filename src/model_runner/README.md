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
        Tokenizer["server/tokenizer.rs<br/>Tokenizer loading and compatibility"]
        KvCache["server/kv_cache.rs<br/>Engine-owned KV-cache implementations"]
        InferenceWorker["server/inference_worker.rs<br/>Inference thread and model ownership"]
        TextGeneration["server/text_generation.rs<br/>Autoregressive decoding loop"]
    end

    Client -->|"Spawns with local paths and socket"| Cli
    Cli --> Server
    InferenceWorker --> Tokenizer
    InferenceWorker --> KvCache
    Server -->|"Bounded request channel"| InferenceWorker
    InferenceWorker --> TextGeneration
```

- `client.rs` creates the socket, starts the worker, waits for readiness, and
  sends the shutdown command.
- `server/cli.rs` receives target and optional draft GGUF paths from the main
  process.
- `server/tokenizer.rs` loads tokenizers and validates that target and draft
  vocabularies use identical token-to-ID mappings.
- `server/kv_cache.rs` preallocates separate key/value pools for contiguous or
  paged storage. Paged mode uses configurable fixed-token-count pages and
  per-layer block tables, and reconstructs contiguous tensors for the existing
  attention operations. Its physical page pool owns tensor storage and free
  page IDs, while its active block tables only map the current sequence to
  those physical pages.
- `server/inference_worker.rs` owns the target model, optional draft model,
  tokenizer, device, and corresponding KV caches on its dedicated thread. The
  target cache uses the configured byte budget; the draft cache is sized to
  hold the same number of tokens.
- `server/text_generation.rs` performs prompt prefill, ordinary greedy decode,
  or speculative decode using draft proposals, batched target verification,
  cache rollback, and request-level acceptance statistics.
- When a draft model is configured, the worker validates its tokenizer against
  the target tokenizer and then retains only the target tokenizer.
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
    Worker->>Decode: Generate text with loaded model(s)
    opt Draft model configured
        Decode->>Draft: Prefill prompt without sampling
    end
    Decode->>Target: Prefill prompt and sample first token
    loop Until stop token, limit, or cancellation
        alt Draft model configured
            Decode->>Draft: Generate draft proposals autoregressively
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
        Decode-->>Worker: Push decoded text fragment
        Worker-->>Rpc: Queue GenerateTextEvent::Text
        Rpc-->>Handler: Stream text event
        Handler-->>Caller: Proxy text event
    end
    Decode-->>Worker: Return TextGenerationStats
    Worker-->>Rpc: Queue GenerateTextEvent::Stats
    Rpc-->>Handler: Stream final statistics
    Handler-->>Caller: Proxy final statistics
```

- `server/mod.rs` receives tonic requests and queues them on a bounded channel.
- `inference_worker.rs` owns the loaded model and processes requests on its
  dedicated thread, clearing its reusable caches before each request.
- `inference_worker.rs` delegates decoding to `text_generation.rs`.
- `text_generation.rs` tokenizes prompts, samples and decodes tokens, checks
  cancellation and stop tokens, and records prefill, decode, and speculative
  acceptance statistics.
- Text fragments stream immediately unless `stream_output` is false, in which
  case they are buffered.
- A successful response ends with a `TextGenerationStats` event.
