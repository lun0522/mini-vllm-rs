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

use super::common::dequantize_to_activation_dtype;
use super::common::precompute_rotary_embedding_frequencies;
use super::common::QMatMul;
use super::common::RotaryEmbeddingContext;
use super::common::RotaryEmbeddingType;
use super::common::SwiGluMlp;
use super::common::TransformerBlock;
use super::common::TransformerModelWeights;
use crate::model_runner::ActivationDType;
use crate::models::CausalLanguageModel;
use crate::models::ForwardInput;
use crate::models::ForwardOutput;
use crate::models::KvCache;
use crate::models::ModelInfo;
use anyhow::Result;
use candle::quantized::gguf_file;
use candle::Device;
use candle::Tensor;
use candle_core as candle;
use candle_nn::Embedding;
use candle_nn::RmsNorm;
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
        activation_dtype: ActivationDType,
    ) -> Result<Self> {
        let model = load_model_weights_from_gguf(content, gguf_file, device, activation_dtype)?;
        let model_info = model.model_info();
        Ok(Self { model, model_info })
    }
}

impl CausalLanguageModel for Qwen2Backend {
    fn info(&self) -> &ModelInfo {
        &self.model_info
    }

    fn forward_with_speculative_verification(
        &mut self,
        generation_inputs: &[ForwardInput],
        verification_inputs: &[ForwardInput],
        kv_cache: &mut dyn KvCache,
    ) -> Result<ForwardOutput> {
        Ok(self
            .model
            .forward(generation_inputs, verification_inputs, kv_cache)?)
    }
}

fn load_model_weights_from_gguf<R: std::io::Seek + std::io::Read>(
    ct: gguf_file::Content,
    reader: &mut R,
    device: &Device,
    activation_dtype: ActivationDType,
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
    // These tensors are model parameters or constants rather than activations, but Candle's
    // elementwise and normalization operations require them to match the activation dtype.
    let (cos, sin) = precompute_rotary_embedding_frequencies(
        head_dim,
        rope_freq_base,
        context_length,
        device,
        activation_dtype,
    )?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?.to_dtype(activation_dtype.into())?;
    let quantized_token_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
    let token_embeddings =
        dequantize_to_activation_dtype(&quantized_token_embeddings, device, activation_dtype)?;
    let output_norm_weight = ct.tensor(reader, "output_norm.weight", device)?;
    let output_norm = RmsNorm::new(
        dequantize_to_activation_dtype(&output_norm_weight, device, activation_dtype)?,
        rms_norm_eps,
    );
    let output = match ct.tensor(reader, "output.weight", device) {
        Ok(tensor) => tensor,
        Err(_) => quantized_token_embeddings,
    };

    let mut layers = Vec::with_capacity(block_count);
    for layer_index in 0..block_count {
        let prefix = format!("blk.{layer_index}");
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
                gate_proj: QMatMul::from_qtensor(ffn_gate, "mlp-gate")?,
                down_proj: QMatMul::from_qtensor(ffn_down, "mlp-down")?,
                up_proj: QMatMul::from_qtensor(ffn_up, "mlp-up")?,
            }
        };

        let attn_norm = ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
        let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

        layers.push(TransformerBlock {
            attn_wq: QMatMul::from_qtensor(attn_wq, "attention-query")?,
            attn_wk: QMatMul::from_qtensor(attn_wk, "attention-key")?,
            attn_wv: QMatMul::from_qtensor(attn_wv, "attention-value")?,
            attn_bq: Some(dequantize_to_activation_dtype(
                &attn_bq,
                device,
                activation_dtype,
            )?),
            attn_bk: Some(dequantize_to_activation_dtype(
                &attn_bk,
                device,
                activation_dtype,
            )?),
            attn_bv: Some(dequantize_to_activation_dtype(
                &attn_bv,
                device,
                activation_dtype,
            )?),
            attn_wo: QMatMul::from_qtensor(attn_wo, "attention-output")?,
            attn_norm: RmsNorm::new(
                dequantize_to_activation_dtype(&attn_norm, device, activation_dtype)?,
                rms_norm_eps,
            ),
            rope_context: RotaryEmbeddingContext {
                rope_type: RotaryEmbeddingType::Neox,
                cos: cos.clone(),
                sin: sin.clone(),
            },
            mlp,
            mlp_norm: RmsNorm::new(
                dequantize_to_activation_dtype(&ffn_norm, device, activation_dtype)?,
                rms_norm_eps,
            ),
            num_q_heads: head_count,
            num_kv_heads: head_count_kv,
            head_dim,
            neg_inf: neg_inf.clone(),
        });
    }

    Ok(TransformerModelWeights::new(
        Embedding::new(token_embeddings, embedding_length),
        layers,
        output_norm,
        QMatMul::from_qtensor(output, "output")?,
    ))
}
