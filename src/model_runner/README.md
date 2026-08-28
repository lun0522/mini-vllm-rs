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
        InferenceWorker["server/inference_worker.rs<br/>Inference thread and model ownership"]
        TextGeneration["server/text_generation.rs<br/>Autoregressive decoding loop"]
    end

    Client -->|"Spawns with local paths and socket"| Cli
    Cli --> Server
    Server -->|"Bounded request channel"| InferenceWorker
    InferenceWorker --> TextGeneration
```

- `client.rs` creates the socket, starts the worker, waits for readiness, and
  sends the shutdown command.
- `server/cli.rs` parses local model paths into `ModelArtifacts`.
- `server/inference_worker.rs` owns the model, tokenizer, device, and KV-cache
  state on its dedicated thread.
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
    participant Model as LoadedModel / Candle

    Caller->>Handler: GenerateText request
    Handler->>Rpc: Forward GenerateText over tonic/UDS
    Rpc->>Worker: Queue InferenceRequest
    Worker->>Decode: Generate text with the loaded model
    loop Until stop token, limit, or cancellation
        Decode->>Model: Forward tokens using KV cache
        Model-->>Decode: Next-token logits
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
  dedicated thread.
- `inference_worker.rs` delegates decoding to `text_generation.rs`.
- `text_generation.rs` tokenizes prompts, samples and decodes tokens, checks
  cancellation and stop tokens, and records timing.
- Text fragments stream immediately unless `stream_output` is false, in which
  case they are buffered.
- A successful response ends with a `TextGenerationStats` event.
