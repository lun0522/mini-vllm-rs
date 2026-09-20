# Models

The `models` module prepares GGUF artifacts and exposes a common inference
interface for the locally adapted quantized Llama and Qwen2 implementations.

## Artifact preparation and model loading

`ModelDownloader` validates each target or draft model configuration, downloads
the GGUF file and `tokenizer.json`, and returns their local Hugging Face cache
paths as `ModelArtifacts`. Existing cached files are reused.

Inside each inference backend, `LoadedModel` opens the GGUF file, reads its
metadata and vocabulary dimensions, and selects the Llama or Qwen2
implementation from `general.architecture`. It owns the initialized model and
its Candle device and reports metadata to the request handler.

Tokenizer loading, vocabulary-coverage checks, and target/draft tokenizer
compatibility checks belong to the
[request handler](../request_handler/README.md), not the models module.

## Shared model interface

| Type | Responsibility |
| --- | --- |
| `CausalLanguageModel` | Runs regular generation and speculative verification through a common backend interface. |
| `ModelInfo` | Reports layer count, KV-head geometry, head dimension, and activation dtype for cache allocation. |
| `ForwardInput` and `ForwardOutput` | Associate packed model inputs with requests and separate final-position generation logits from per-position verification logits. |
| `KvCache` | Gives model layers request-aware access to contiguous cache tensors or paged cache layouts during a forward pass. |

The `KvCache` interface intentionally contains only operations needed during
model execution. Request lifecycle operations such as cache restoration,
truncation, and cleanup remain internal to the model runner.

## KV cache and attention

Model layers do not own request-specific KV tensors. Each target or draft
`ModelInstance` has an independent cache manager, which passes the selected
cache backend into every model forward call. External cache ownership keeps
reusable model weights separate from request state and enables continuous
batching, prefix caching, and speculative cache rollback.

Contiguous and paged caches preallocate separate key and value storage from a
configured byte budget. Paged storage uses fixed-token-count pages and
per-layer block tables; its physical storage, request mappings, and reusable
prefix index are described in the
[KV-cache architecture](../model_runner/server/kv_cache/README.md).

GPU execution reconstructs contiguous tensors from paged storage before
attention. CPU can optionally calculate attention directly over the pages.
Reusable causal attention masks remain inside the model because they depend on
tensor shapes, not on the lifecycle of a particular request.

## Local Candle adaptations

The quantized Llama and Qwen2 implementations under `backends/` originated in
Candle 0.11.0. The local versions:

- Support GGUF only, with legacy GGML loading code and upstream utility tests
  removed.
- Use Candle's public crate APIs.
- Implement the shared batching, external KV-cache, and speculative-verification
  interfaces described above.

See the backend directory's [provenance notice](backends/README.md) for the
copied Candle source and its license. Downloaded model weights are covered
separately in [Model licenses](../../MODEL_LICENSES.md).
