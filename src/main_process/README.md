# Main process architecture

The `main_process` module implements the supervisor that prepares model
artifacts, starts the serving processes, and coordinates their shutdown.

```mermaid
flowchart LR
    Cli["cli.rs<br/>Configuration parsing"] --> Supervisor["supervisor.rs<br/>Process orchestration"]
    Supervisor -->|"Prepares artifacts with"| Models["models<br/>Artifact download and validation"]
    Models -->|"Returns local paths"| Supervisor
    Supervisor -->|"Starts and owns"| Runner["Model runner process"]
    Supervisor -->|"Starts and owns"| Handler["Request handler process"]
    Supervisor -->|"Binds and owns"| Control["server.rs<br/>Shutdown service"]
    Control -->|"Signals shutdown"| Supervisor
```

## Component responsibilities

| Component | Responsibility |
| --- | --- |
| `cli.rs` | Parses model, device, KV-cache, scheduler, preprocessing-pool, and socket configuration, then normalizes values and supplies defaults. |
| `supervisor.rs` | Prepares artifacts, starts child processes in dependency order, waits for Ctrl-C or a control RPC, and coordinates shutdown. |
| `server.rs` | Exposes the main-process `Shutdown` RPC and owns the control socket. |
| [`models`](../models/README.md) | Validates model configuration and prepares local GGUF and tokenizer paths. |

Architecture selection and device-specific model loading happen later inside
each model-runner backend. The control server uses
`/tmp/mini-vllm-main-process.sock` by default and removes the socket when it is
dropped.

Startup follows the serving processes' dependency order:

```mermaid
sequenceDiagram
    participant Main as Main process
    participant Models
    participant Runner as Model runner
    participant Handler as Request handler
    participant Control as Control server

    Main->>Models: Download target and optional draft artifacts
    Models-->>Main: GGUF and tokenizer paths
    Main->>Runner: Start with model paths, device, and cache configuration
    Runner-->>Main: All configured backends loaded, socket is ready
    Main->>Handler: Start with tokenizer and model-runner socket paths
    Handler->>Runner: Connect and validate model metadata
    Handler-->>Main: Request-handler socket is ready
    Main->>Control: Bind the main-process control socket
```

The runner starts first because the handler connects to it during
initialization. A child's socket signals readiness; in mixed mode, the runner
binds its socket only after both CPU and GPU backends are loaded.

Shutdown uses the reverse dependency order:

```mermaid
sequenceDiagram
    participant Caller as Control client or terminal
    participant Main as Main process
    participant Handler as Request handler
    participant Runner as Model runner

    alt Shutdown RPC
        Caller->>Main: Shutdown RPC
        Main-->>Caller: CommandResult
    else Terminal signal
        Caller->>Main: Ctrl-C
    end
    Main->>Handler: Shutdown RPC
    Handler-->>Main: Process exited and socket removed
    Main->>Runner: Shutdown command
    Runner-->>Main: Process exited and socket removed
    Note over Main: Main process exits
```

The control RPC acknowledges the shutdown signal before child cleanup finishes.
Ctrl-C initiates the same cleanup directly.

If request-handler startup fails, the supervisor shuts down the already-running
model runner before returning the error. Process owners also forcibly stop
their child and remove its socket when graceful cleanup cannot complete.
