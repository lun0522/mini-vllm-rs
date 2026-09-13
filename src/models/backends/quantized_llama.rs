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

use std::collections::HashMap;

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
use std::fs::File;

#[derive(Debug, Clone)]
pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    output: QMatMul,
    /// Mask cache keyed by (seq_len, kv_len).
    /// kv_len = index_pos + seq_len, so the mask is rectangular when prefix
    /// KV cache entries exist (index_pos > 0).
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

        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base, context_length, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;
        let tok_embeddings_q = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings_q.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => tok_embeddings_q,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");
            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
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
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_bq: None,
                attention_bk: None,
                attention_bv: None,
                attention_norm: RmsNorm::from_qtensor(attention_norm, rms_norm_eps)?,
                mlp,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, rms_norm_eps)?,
                query_head_count: head_count,
                key_value_head_count: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                rotary_embedding_type: RotaryEmbeddingType::Interleaved,
                attention_span: span_attn,
                rotary_span: span_rot,
                mlp_span: span_mlp,
            })
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    /// Build a causal attention mask of shape `(seq_len, kv_len)` where
    /// `kv_len = index_pos + seq_len`.
    ///
    /// When `index_pos == 0` the mask is square `(seq_len, seq_len)` — the
    /// classic case with an empty KV cache.
    ///
    /// When `index_pos > 0` the KV cache already holds `index_pos` entries from
    /// a previously fed prefix.  The mask becomes rectangular: the first
    /// `index_pos` columns are all 0 (every query attends to every prefix key)
    /// and the remaining `seq_len` columns form the standard causal triangle
    /// (query at global position `index_pos + i` cannot attend to keys at global
    /// positions `> index_pos + i`).
    ///
    /// # Shape example  (index_pos=65, seq_len=4)
    /// ```text
    ///              kv 0..64 (prefix)   kv 65  kv 66  kv 67  kv 68
    /// query 65:       0  0 … 0           0      1      1      1
    /// query 66:       0  0 … 0           0      0      1      1
    /// query 67:       0  0 … 0           0      0      0      1
    /// query 68:       0  0 … 0           0      0      0      0
    /// ```
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

pub(crate) struct LlamaBackend {
    model: ModelWeights,
    model_info: ModelInfo,
}

impl LlamaBackend {
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
