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
  owns a preallocated `ContiguousKvCache` for each loaded target or draft model
  and passes it through the `KvCache` trait into every model `forward` call.
- Each empty cache slot becomes a contiguous key/value tensor pair during
  prompt prefill. Decode steps append the new token's tensors to that pair. This
  is not paged attention yet.
- Reusable causal attention masks remain in the model because they are derived
  from tensor shapes rather than belonging to a particular request.

External cache ownership separates reusable model state from request state. It
provides the foundation for cache allocation, paged attention, batching, and
independent target/draft caches for speculative decoding.

See the model directory's [provenance notice](models/README.md) for the exact
upstream commit and licensing information.
