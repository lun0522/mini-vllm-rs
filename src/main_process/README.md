# Main process architecture

The `main_process` module implements the supervisor that prepares model
artifacts, starts the serving processes, and coordinates their shutdown.

```mermaid
flowchart LR
    Cli["cli.rs<br/>Configuration parsing"] --> Supervisor["supervisor.rs<br/>Process orchestration"]
    Loaders["model_loaders<br/>Artifact download and validation"] --> Supervisor
    Supervisor -->|"Starts and owns"| Runner["Model runner process"]
    Supervisor -->|"Starts and owns"| Handler["Request handler process"]
    Control["server.rs<br/>Shutdown service"] --> Supervisor
```

- `cli.rs` parses model, KV-cache, and socket configuration and normalizes
  invalid or omitted values.
- `supervisor.rs` downloads the configured artifacts, starts the model runner
  and request handler in dependency order, waits for Ctrl-C or a control RPC,
  and shuts both child processes down.
- `server.rs` exposes the main process `Shutdown` RPC on
  `/tmp/mini-vllm-main-process.sock` by default. The control socket is removed
  when its server is dropped.
- Model downloading, configuration validation, architecture selection, and
  device-specific loading are described in [Model loaders](../model_loaders/README.md).

Startup follows the serving processes' dependency order:

```mermaid
sequenceDiagram
    participant Main as Main process
    participant Loaders as Model loaders
    participant Runner as Model runner
    participant Handler as Request handler
    participant Control as Control server

    Main->>Loaders: Download target and optional draft artifacts
    Loaders-->>Main: GGUF and tokenizer paths
    Main->>Runner: Start with model paths and cache configuration
    Runner-->>Main: Model-runner socket is ready
    Main->>Handler: Start with tokenizer and model-runner socket paths
    Handler->>Runner: Connect and validate model metadata
    Handler-->>Main: Request-handler socket is ready
    Main->>Control: Bind the main-process control socket
```

The model runner starts first because the request handler connects to it during
initialization. Each child socket is treated as its readiness signal, so the
next stage does not start until its dependency is accepting connections.

Shutdown uses the reverse dependency order:

```mermaid
sequenceDiagram
    participant Caller as Control client or terminal
    participant Main as Main process
    participant Handler as Request handler
    participant Runner as Model runner

    Caller->>Main: Shutdown RPC or Ctrl-C
    Main->>Handler: Shutdown RPC
    Handler-->>Main: Process exited and socket removed
    Main->>Runner: Shutdown command
    Runner-->>Main: Process exited and socket removed
    Main-->>Caller: Main process exits
```

If request-handler startup fails, the supervisor shuts down the already-running
model runner before returning the error. Process owners also forcibly stop
their child and remove its socket when graceful cleanup cannot complete.
