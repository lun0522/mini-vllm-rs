# Model loaders

The model-loading code prepares GGUF artifacts, selects an implementation from
the GGUF architecture metadata, validates tokenizer vocabulary coverage, and
returns a `LoadedModel` containing reusable model weights and its device.

## Local Candle model adaptations

The quantized Llama and Qwen2 implementations under `models/` were copied from
Candle 0.11.0 and adapted locally:

- The existing mini-vLLM backend adapters were merged into the corresponding
  model files and implement the shared `CausalLanguageModel` interface.
- Legacy GGML loading code and upstream utility tests were removed because this
  project loads only GGUF models and does not maintain Candle's test suite.
- Candle-internal import paths were changed to use the public `candle-core`,
  `candle-nn`, and `candle-transformers` APIs available to this crate.
- Request-specific KV tensors were removed from the model layers. `ModelRunner`
  owns the selected cache implementation for each loaded target or draft model
  and passes it through the `KvCache` trait into every model `forward` call.
- Each backend records reusable `ModelInfo`, including its layer count, KV-head
  geometry, and activation data type, so the model runner can allocate caches
  without depending on architecture-specific model internals.
- Contiguous and paged caches both preallocate separate key and value pools from
  a configured total byte budget. Paged caches use a configurable number of
  tokens per page and maintain a block table for each layer.
- Paged caches reconstruct contiguous tensors before calling Candle's existing
  attention operations.
- Model backends provide logits for either the final input position or every
  input position; speculative decoding uses the latter to verify a draft token
  batch with one target-model forward pass.
- Reusable causal attention masks remain in the model because they are derived
  from tensor shapes rather than belonging to a particular request.

External cache ownership separates reusable model state from request state. It
provides the foundation for cache allocation, paged attention, batching, and
independent target/draft caches for speculative decoding.

See the model directory's [provenance notice](models/README.md) for the exact
upstream commit and licensing information.
