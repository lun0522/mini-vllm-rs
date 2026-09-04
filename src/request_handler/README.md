# Request handler architecture

The `request_handler` module separates the main-process lifecycle client from
the code that runs inside the request-handler process.

```mermaid
flowchart LR
    subgraph Main[Main process]
        Client["client.rs<br/>Starts and stops the request handler"]
    end

    subgraph Handler[Request handler process]
        Server["server/mod.rs<br/>Public tonic service"]
        Tokenizer["server/tokenizer.rs<br/>Chat formatting and tokenization"]
        Events["server/generation_event_processor.rs<br/>Token decoding and event conversion"]
    end

    Runner["Model runner process"]

    Client -->|"Spawns with tokenizer and socket paths"| Server
    Server --> Tokenizer
    Server -->|"Token-level RPC"| Runner
    Runner -->|"Token IDs and final statistics"| Events
    Events -->|"Text and final statistics"| Server
```

- `client.rs` starts the request-handler process, waits for its socket, exposes
  the public generation client, and manages shutdown and socket cleanup.
- `server/mod.rs` connects to the model runner before binding its public socket,
  so the socket indicates that the request handler is ready to serve requests.
- `server/tokenizer.rs` loads the target tokenizer, validates the optional draft
  tokenizer and both models' vocabulary sizes, formats chat prompts, tokenizes
  model input, and incrementally decodes generated token IDs.
- `server/generation_event_processor.rs` converts model-runner token events into
  public text events. It emits fragments immediately for streaming requests or
  buffers them into one response when streaming is disabled.
- The request handler owns text-oriented preprocessing and postprocessing; the
  model runner receives token IDs and returns token IDs plus final generation
  statistics.

The generation path is:

```mermaid
sequenceDiagram
    participant Caller as Inference client
    participant Handler as Request handler
    participant Tokenizer as Tokenizer
    participant Runner as Model runner
    participant Events as Event processor

    Caller->>Handler: GenerateText(prompt, parameters)
    Handler->>Tokenizer: Format and tokenize prompt
    Tokenizer-->>Handler: Input and stop token IDs
    Handler->>Runner: GenerateTextRequest(token IDs)
    loop Generated tokens
        Runner-->>Handler: Token ID
        Handler->>Events: Process token ID
        Events-->>Caller: Text fragment when available
    end
    Runner-->>Handler: Final statistics
    Handler->>Events: Process statistics
    opt Buffered output
        Events-->>Caller: Complete generated text
    end
    Events-->>Caller: Final statistics
```

A successful response always ends with a statistics event. Transport errors,
malformed model-runner events, and incremental decoding failures are forwarded
to the inference client as gRPC errors.
