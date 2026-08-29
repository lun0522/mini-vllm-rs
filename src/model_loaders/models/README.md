# Candle model source provenance

The model implementations in this directory were initially copied from
[`huggingface/candle`](https://github.com/huggingface/candle) at the `0.11.0`
release (commit `31f35b147389700ed2a178ee66a91c3cc25cc80d`):

- `quantized_llama.rs` from
  `candle-transformers/src/models/quantized_llama.rs`.
- `quantized_qwen2.rs` from
  `candle-transformers/src/models/quantized_qwen2.rs`.

The copied code is distributed under Candle's MIT license. The complete license
text is included in `CANDLE_LICENSE-MIT` in this directory.

They have since been integrated with mini-vllm-rs's model-loading interface and
may diverge from the upstream implementations.
