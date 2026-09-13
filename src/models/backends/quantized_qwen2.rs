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

use super::common::precomput_freqs_cis;
use super::common::QMatMul;
use super::common::RotaryEmbeddingType;
use super::common::SwiGluMlp;
use super::common::TransformerBlock;
use crate::models::CausalLanguageModel;
use crate::models::ForwardContext;
use crate::models::KvCache;
use crate::models::ModelInfo;
use anyhow::Result as AnyhowResult;
use candle::quantized::gguf_file;
use candle::{Device, IndexOp, Result, Tensor};
use candle_core as candle;
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use std::collections::HashMap;
use std::fs::File;

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    output: QMatMul,
    masks: HashMap<(usize, usize), Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
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
        let (cos, sin) = precomput_freqs_cis(head_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
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
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;

            let attention_bq = ct.tensor(reader, &format!("{prefix}.attn_q.bias"), device)?;
            let attention_bk = ct.tensor(reader, &format!("{prefix}.attn_k.bias"), device)?;
            let attention_bv = ct.tensor(reader, &format!("{prefix}.attn_v.bias"), device)?;

            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp = {
                let feed_forward_w1 =
                    ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let feed_forward_w2 =
                    ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let feed_forward_w3 =
                    ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                SwiGluMlp {
                    feed_forward_w1: QMatMul::from_qtensor(feed_forward_w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(feed_forward_w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(feed_forward_w3)?,
                }
            };

            let attention_norm =
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(TransformerBlock {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_bq: Some(attention_bq.dequantize(device)?),
                attention_bk: Some(attention_bk.dequantize(device)?),
                attention_bv: Some(attention_bv.dequantize(device)?),
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                cos: cos.clone(),
                sin: sin.clone(),
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                query_head_count: head_count,
                key_value_head_count: head_count_kv,
                head_dim,
                neg_inf: neg_inf.clone(),
                rotary_embedding_type: RotaryEmbeddingType::Neox,
                attention_span: span_attn,
                rotary_span: span_rot,
                mlp_span: span_mlp,
            });
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, seq_len: usize, index_pos: usize, device: &Device) -> Result<Tensor> {
        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(mask.clone())
        } else {
            let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let x = self.forward_hidden(x, context, kv_cache)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }

    pub fn forward_for_speculative_verification(
        &mut self,
        x: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let x = self.forward_hidden(x, context, kv_cache)?;
        let _enter = self.span_output.enter();
        self.output.forward(&x)
    }

    fn forward_hidden(
        &mut self,
        x: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let index_pos = context.start_position;
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len, index_pos, x.device())?)
        };
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let x = layer_in;
            let residual = &x;
            let x = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&x, mask.as_ref(), context, layer_index, kv_cache)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.mlp_span.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = (x + residual)?;
            layer_in = x
        }
        self.norm.forward(&layer_in)
    }
}

pub(crate) struct Qwen2Backend {
    model: ModelWeights,
    model_info: ModelInfo,
}

impl Qwen2Backend {
    pub(crate) fn new(
        content: gguf_file::Content,
        gguf_file: &mut File,
        device: &Device,
    ) -> AnyhowResult<Self> {
        let model = ModelWeights::from_gguf(content, gguf_file, device)?;
        let attention = &model.layers[0];
        let model_info = ModelInfo {
            layer_count: model.layers.len(),
            key_value_head_count: attention.key_value_head_count,
            head_dim: attention.head_dim,
            activation_dtype: model.tok_embeddings.embeddings().dtype(),
        };
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
        self.model.forward(input, context, kv_cache)?.unsqueeze(1)
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
