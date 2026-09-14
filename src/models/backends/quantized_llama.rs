//! Quantized llama model implementation.
//!
//! This provides a quantized implementation of the llama language model architecture.
//! The model implements parameter efficient quantization for reduced memory usage
//! while maintaining model quality.
//!
//! Key characteristics:
//! - Transformer decoder architecture
//! - Support for 2/3/4/8-bit quantization
//! - Optimized memory usage through quantization
//! - Configurable model sizes and parameter counts
//!
//! - 💻 [GH Link](https://github.com/facebookresearch/llama)
//! - 📝 [Paper](https://arxiv.org/abs/2302.13971)
//!
//! ![](https://raw.githubusercontent.com/huggingface/candle/main/candle-examples/examples/quantized/assets/aoc.gif)
//!

use super::common::precompute_rotary_embedding_frequencies;
use super::common::QMatMul;
use super::common::RotaryEmbeddingContext;
use super::common::RotaryEmbeddingType;
use super::common::SwiGluMlp;
use super::common::TransformerBlock;
use super::common::TransformerModelWeights;
use crate::models::BatchedForwardInput;
use crate::models::BatchedKvCache;
use crate::models::CausalLanguageModel;
use crate::models::ForwardContext;
use crate::models::KvCache;
use crate::models::ModelInfo;
use anyhow::Result as AnyhowResult;
use candle::quantized::gguf_file;
use candle::{Device, Result, Tensor};
use candle_core as candle;
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use std::fs::File;

pub(crate) struct LlamaBackend {
    model: TransformerModelWeights,
    model_info: ModelInfo,
}

impl LlamaBackend {
    pub(crate) fn new(
        content: gguf_file::Content,
        gguf_file: &mut File,
        device: &Device,
    ) -> AnyhowResult<Self> {
        let model = load_model_weights_from_gguf(content, gguf_file, device)?;
        let model_info = model.model_info();
        Ok(Self { model, model_info })
    }
}

impl CausalLanguageModel for LlamaBackend {
    fn info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn forward(
        &mut self,
        input: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        self.model.forward(input, context, kv_cache)
    }

    fn forward_batched(
        &mut self,
        inputs: &[BatchedForwardInput],
        kv_cache: &mut dyn BatchedKvCache,
    ) -> Result<Vec<Tensor>> {
        self.model.forward_batched(inputs, kv_cache)
    }

    fn forward_for_speculative_verification(
        &mut self,
        input: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        self.model
            .forward_for_speculative_verification(input, context, kv_cache)
    }
}

fn load_model_weights_from_gguf<R: std::io::Seek + std::io::Read>(
    ct: gguf_file::Content,
    reader: &mut R,
    device: &Device,
) -> Result<TransformerModelWeights> {
    let md_get = |s: &str| match ct.metadata.get(s) {
        None => candle::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };

    let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
    let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
    let block_count = md_get("llama.block_count")?.to_u32()? as usize;
    let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
    let context_length = md_get("llama.context_length")?.to_u32()? as usize;
    let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
    // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
    let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let rope_freq_base = md_get("llama.rope.freq_base")
        .and_then(|m| m.to_f32())
        .unwrap_or(10000f32);

    let (cos, sin) =
        precompute_rotary_embedding_frequencies(rope_dim, rope_freq_base, context_length, device)?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
    let quantized_token_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
    let token_embeddings = quantized_token_embeddings.dequantize(device)?;
    let output_norm = RmsNorm::from_qtensor(
        ct.tensor(reader, "output_norm.weight", device)?,
        rms_norm_eps,
    )?;
    let output = match ct.tensor(reader, "output.weight", device) {
        Ok(tensor) => tensor,
        Err(_) => quantized_token_embeddings,
    };

    let mut layers = Vec::with_capacity(block_count);
    for layer_idx in 0..block_count {
        let prefix = format!("blk.{layer_idx}");
        let attn_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
        let attn_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
        let attn_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
        let attn_wo = ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

        let mlp = {
            let ffn_gate = ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let ffn_down = ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
            let ffn_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            SwiGluMlp {
                gate_proj: QMatMul::from_qtensor(ffn_gate)?,
                down_proj: QMatMul::from_qtensor(ffn_down)?,
                up_proj: QMatMul::from_qtensor(ffn_up)?,
            }
        };

        let attn_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
        let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rope = tracing::span!(tracing::Level::TRACE, "attn-rope");
        let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

        layers.push(TransformerBlock {
            attn_wq: QMatMul::from_qtensor(attn_wq)?,
            attn_wk: QMatMul::from_qtensor(attn_wk)?,
            attn_wv: QMatMul::from_qtensor(attn_wv)?,
            attn_wo: QMatMul::from_qtensor(attn_wo)?,
            attn_bq: None,
            attn_bk: None,
            attn_bv: None,
            attn_norm: RmsNorm::from_qtensor(attn_norm, rms_norm_eps)?,
            mlp,
            mlp_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
            num_q_heads: head_count,
            num_kv_heads: head_count_kv,
            head_dim: embedding_length / head_count,
            rope_context: RotaryEmbeddingContext {
                rope_type: RotaryEmbeddingType::Interleaved,
                cos: cos.clone(),
                sin: sin.clone(),
                span_rope,
            },
            neg_inf: neg_inf.clone(),
            span_attn,
            span_mlp,
        })
    }

    Ok(TransformerModelWeights::new(
        Embedding::new(token_embeddings, embedding_length),
        layers,
        output_norm,
        QMatMul::from_qtensor(output)?,
    ))
}
