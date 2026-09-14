//! Qwen2 model implementation with quantization support.
//!
//! Qwen2 is a chat-optimized language model that supports 8-bit quantization
//! for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Group Query Attention (GQA)
//! - RMSNorm for layer normalization
//! - Rotary positional embeddings (RoPE)
//! - Support for 8-bit quantization
//!
//! References:
//! - [Model Card](https://huggingface.co/Qwen/Qwen2)
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
use anyhow::Result;
use candle::quantized::gguf_file;
use candle::Device;
use candle::Tensor;
use candle_core as candle;
use candle_nn::Embedding;
use candle_transformers::quantized_nn::RmsNorm;
use std::fs::File;

pub(crate) struct Qwen2Backend {
    model: TransformerModelWeights,
    model_info: ModelInfo,
}

impl Qwen2Backend {
    pub(crate) fn new(
        content: gguf_file::Content,
        gguf_file: &mut File,
        device: &Device,
    ) -> Result<Self> {
        let model = load_model_weights_from_gguf(content, gguf_file, device)?;
        let model_info = model.model_info();
        Ok(Self { model, model_info })
    }
}

impl CausalLanguageModel for Qwen2Backend {
    fn info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn forward(
        &mut self,
        input: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        Ok(self.model.forward(input, context, kv_cache)?)
    }

    fn forward_batched(
        &mut self,
        inputs: &[BatchedForwardInput],
        kv_cache: &mut dyn BatchedKvCache,
    ) -> Result<Vec<Tensor>> {
        Ok(self.model.forward_batched(inputs, kv_cache)?)
    }

    fn forward_for_speculative_verification(
        &mut self,
        input: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        Ok(self
            .model
            .forward_for_speculative_verification(input, context, kv_cache)?)
    }
}

fn load_model_weights_from_gguf<R: std::io::Seek + std::io::Read>(
    ct: gguf_file::Content,
    reader: &mut R,
    device: &Device,
) -> candle::Result<TransformerModelWeights> {
    let md_get = |s: &str| match ct.metadata.get(s) {
        None => candle::bail!("cannot find {s} in metadata"),
        Some(v) => Ok(v),
    };

    let head_count = md_get("qwen2.attention.head_count")?.to_u32()? as usize;
    let head_count_kv = md_get("qwen2.attention.head_count_kv")?.to_u32()? as usize;
    let embedding_length = md_get("qwen2.embedding_length")?.to_u32()? as usize;
    let context_length = md_get("qwen2.context_length")?.to_u32()? as usize;
    let block_count = md_get("qwen2.block_count")?.to_u32()? as usize;
    let rms_norm_eps = md_get("qwen2.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
    let rope_freq_base = md_get("qwen2.rope.freq_base")
        .and_then(|m| m.to_f32())
        .unwrap_or(10000f32);

    let head_dim = embedding_length / head_count;
    let (cos, sin) =
        precompute_rotary_embedding_frequencies(head_dim, rope_freq_base, context_length, device)?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
    let token_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
    let token_embeddings = token_embeddings.dequantize(device)?;
    let output_norm = RmsNorm::from_qtensor(
        ct.tensor(reader, "output_norm.weight", device)?,
        rms_norm_eps,
    )?;
    let output = match ct.tensor(reader, "output.weight", device) {
        Ok(v) => QMatMul::from_qtensor(v)?,
        _ => {
            // use tie_word_embeddings
            QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?
        }
    };

    let mut layers = Vec::with_capacity(block_count);
    for layer_idx in 0..block_count {
        let prefix = format!("blk.{layer_idx}");
        let attn_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
        let attn_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
        let attn_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;

        let attn_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
        let attn_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
        let attn_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;

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
            attn_bq: Some(attn_bq.dequantize(device)?),
            attn_bk: Some(attn_bk.dequantize(device)?),
            attn_bv: Some(attn_bv.dequantize(device)?),
            attn_wo: QMatMul::from_qtensor(attn_wo)?,
            attn_norm: RmsNorm::from_qtensor(attn_norm, rms_norm_eps)?,
            rope_context: RotaryEmbeddingContext {
                rope_type: RotaryEmbeddingType::Neox,
                cos: cos.clone(),
                sin: sin.clone(),
                span_rope,
            },
            mlp,
            mlp_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
            num_q_heads: head_count,
            num_kv_heads: head_count_kv,
            head_dim,
            neg_inf: neg_inf.clone(),
            span_attn,
            span_mlp,
        });
    }

    Ok(TransformerModelWeights::new(
        Embedding::new(token_embeddings, embedding_length),
        layers,
        output_norm,
        output,
    ))
}
