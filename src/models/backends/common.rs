use crate::models::CachedKeyValue;
use crate::models::ForwardContext;
use crate::models::KvCache;
use crate::models::ModelInfo;
use candle_core::quantized::QTensor;
use candle_core::DType;
use candle_core::Device;
use candle_core::IndexOp;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::Embedding;
use candle_nn::Module;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use std::collections::HashMap;

/// Caches causal attention masks by `(seq_len, kv_len)`, where
/// `kv_len = index_pos + seq_len`.
///
/// When `index_pos == 0`, the mask is square. With a cached prefix, the mask is
/// rectangular: the first `index_pos` columns allow every query to attend to
/// every prefix key, and the remaining `seq_len` columns form the causal
/// triangle for the current forward call.
///
/// For example, with `index_pos = 65` and `seq_len = 4`:
///
/// ```text
///              kv 0..64 (prefix)   kv 65  kv 66  kv 67  kv 68
/// query 65:       0  0 … 0           0      1      1      1
/// query 66:       0  0 … 0           0      0      1      1
/// query 67:       0  0 … 0           0      0      0      1
/// query 68:       0  0 … 0           0      0      0      0
/// ```
#[derive(Debug, Clone, Default)]
pub(super) struct CausalMaskCache {
    masks: HashMap<(usize, usize), Tensor>,
}

impl CausalMaskCache {
    /// Returns no mask for single-token decode; otherwise, returns a cached or
    /// newly created causal mask.
    pub(super) fn get_or_create(
        &mut self,
        seq_len: usize,
        index_pos: usize,
        device: &Device,
    ) -> Result<Option<Tensor>> {
        if seq_len == 1 {
            return Ok(None);
        }

        let kv_len = index_pos + seq_len;
        if let Some(mask) = self.masks.get(&(seq_len, kv_len)) {
            Ok(Some(mask.clone()))
        } else {
            let mask = candle_transformers::utils::build_causal_mask(seq_len, index_pos, device)?;
            self.masks.insert((seq_len, kv_len), mask.clone());
            Ok(Some(mask))
        }
    }
}

// QMatMul wrapper adding tracing.
#[derive(Debug, Clone)]
pub(super) struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    pub(super) fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(input)
    }
}

#[derive(Debug, Clone)]
pub(super) struct SwiGluMlp {
    pub(super) gate_proj: QMatMul,
    pub(super) down_proj: QMatMul,
    pub(super) up_proj: QMatMul,
}

impl Module for SwiGluMlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(input)?;
        let up = self.up_proj.forward(input)?;
        self.down_proj
            .forward(&(candle_nn::ops::silu(&gate)? * up)?)
    }
}

#[derive(Debug, Clone)]
pub(super) struct TransformerBlock {
    pub(super) attn_wq: QMatMul,
    pub(super) attn_wk: QMatMul,
    pub(super) attn_wv: QMatMul,
    pub(super) attn_wo: QMatMul,
    pub(super) attn_bq: Option<Tensor>,
    pub(super) attn_bk: Option<Tensor>,
    pub(super) attn_bv: Option<Tensor>,
    pub(super) attn_norm: RmsNorm,
    pub(super) mlp: SwiGluMlp,
    pub(super) mlp_norm: RmsNorm,
    pub(super) num_q_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
    pub(super) neg_inf: Tensor,
    pub(super) rope_type: RotaryEmbeddingType,
    pub(super) span_attn: tracing::Span,
    pub(super) span_rope: tracing::Span,
    pub(super) span_mlp: tracing::Span,
}

impl TransformerBlock {
    pub(super) fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        context: &ForwardContext,
        layer_index: usize,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let x = self.forward_attention(&x, mask, context, layer_index, kv_cache)?;
        let x = (x + residual)?;
        self.forward_mlp(&x)
    }

    fn forward_attention(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        context: &ForwardContext,
        layer_index: usize,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();

        let index_pos = context.start_position;
        let (batch_size, query_len, embedding_len) = x.dims3()?;

        // query_len is the number of tokens in this forward call.
        // kv_len also includes tokens already stored in the KV cache.
        // x, q: [batch_size, query_len, embedding_len]
        // k, v: [batch_size, query_len, num_kv_heads * head_dim]
        // num_kv_heads * head_dim can be less than embedding_len for GQA or MQA.
        // Optional biases are broadcast across batch_size and query_len without
        // changing the dimensions of q, k, or v.
        let q = apply_projection(x, &self.attn_wq, self.attn_bq.as_ref())?;
        let k = apply_projection(x, &self.attn_wk, self.attn_bk.as_ref())?;
        let v = apply_projection(x, &self.attn_wv, self.attn_bv.as_ref())?;

        // After transpose:
        // q: [batch_size, num_q_heads, query_len, head_dim]
        // k, v: [batch_size, num_kv_heads, query_len, head_dim]
        let q = q
            .reshape((batch_size, query_len, self.num_q_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch_size, query_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, query_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Rotary embeddings preserve the dimensions of q and k.
        let q = apply_rotary_embedding(
            self.rope_type,
            &q,
            index_pos,
            &self.cos,
            &self.sin,
            &self.span_rope,
        )?;
        let k = apply_rotary_embedding(
            self.rope_type,
            &k,
            index_pos,
            &self.cos,
            &self.sin,
            &self.span_rope,
        )?;

        // Cached k and v:
        // [batch_size, num_kv_heads, kv_len, head_dim].
        let CachedKeyValue { key: k, value: v } = kv_cache
            .append(context, layer_index, &k, &v)
            .map_err(candle_core::Error::wrap)?;

        // y: [batch_size, num_q_heads, query_len, head_dim].
        let y = if q.device().is_metal() && query_len == 1 {
            // Metal SDPA handles GQA or MQA without explicitly repeating k and v.
            candle_nn::ops::sdpa(
                &q,
                &k,
                &v,
                None,
                false,
                1. / (self.head_dim as f32).sqrt(),
                1.,
            )?
        } else {
            // Expand grouped-query or multi-query heads to match q:
            // [batch_size, num_q_heads, kv_len, head_dim].
            let repetition_count = self.num_q_heads / self.num_kv_heads;
            let k = repeat_kv(k, repetition_count)?;
            let v = repeat_kv(v, repetition_count)?;

            // Attention scores:
            // [batch_size, num_q_heads, query_len, kv_len].
            let attn_scores = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let attn_scores = match mask {
                None => attn_scores,
                Some(mask) => {
                    let mask = mask.broadcast_as(attn_scores.shape())?;
                    masked_fill(&attn_scores, &mask, &self.neg_inf)?
                }
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)?;

            // Convert to contiguous as matmul doesn't support arbitrary strided v tensors.
            attn_weights.matmul(&v.contiguous()?)?
        };

        // After transpose and reshape:
        // [batch_size, query_len, embedding_len].
        let y = y
            .transpose(1, 2)?
            .reshape(&[batch_size, query_len, embedding_len])?;

        // The output projection preserves
        // [batch_size, query_len, embedding_len].
        self.attn_wo.forward(&y)
    }

    fn forward_mlp(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span_mlp.enter();
        let residual = x;
        let x = self.mlp_norm.forward(x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

pub(super) struct TransformerModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    output: QMatMul,
    mask_cache: CausalMaskCache,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl TransformerModelWeights {
    pub(super) fn new(
        tok_embeddings: Embedding,
        layers: Vec<TransformerBlock>,
        norm: RmsNorm,
        output: QMatMul,
    ) -> Self {
        Self {
            tok_embeddings,
            layers,
            norm,
            output,
            mask_cache: CausalMaskCache::default(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        }
    }

    pub(super) fn model_info(&self) -> ModelInfo {
        let attention = &self.layers[0];
        ModelInfo {
            layer_count: self.layers.len(),
            num_kv_heads: attention.num_kv_heads,
            head_dim: attention.head_dim,
            activation_dtype: self.tok_embeddings.embeddings().dtype(),
        }
    }

    pub(super) fn forward(
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

    pub(super) fn forward_for_speculative_verification(
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
        let (_, seq_len) = x.dims2()?;
        let mask = self
            .mask_cache
            .get_or_create(seq_len, index_pos, x.device())?;
        let _enter = self.span.enter();
        let mut layer_in = self.tok_embeddings.forward(x)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            layer_in = layer.forward(&layer_in, mask.as_ref(), context, layer_index, kv_cache)?;
        }
        self.norm.forward(&layer_in)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum RotaryEmbeddingType {
    Neox,
    Interleaved,
}

pub(super) fn precompute_rotary_embedding_frequencies(
    head_dim: usize,
    freq_base: f32,
    context_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_len, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

fn apply_projection(input: &Tensor, proj: &QMatMul, bias: Option<&Tensor>) -> Result<Tensor> {
    let projected = proj.forward(input)?;
    match bias {
        Some(bias) => projected.broadcast_add(bias),
        None => Ok(projected),
    }
}

fn apply_rotary_embedding(
    rope_type: RotaryEmbeddingType,
    x: &Tensor,
    index_pos: usize,
    cos: &Tensor,
    sin: &Tensor,
    span: &tracing::Span,
) -> Result<Tensor> {
    let _enter = span.enter();
    let (_, _, query_len, _) = x.dims4()?;
    let cos = cos.narrow(0, index_pos, query_len)?;
    let sin = sin.narrow(0, index_pos, query_len)?;
    match rope_type {
        RotaryEmbeddingType::Neox => candle_nn::rotary_emb::rope(x, &cos, &sin),
        RotaryEmbeddingType::Interleaved => candle_nn::rotary_emb::rope_i(x, &cos, &sin),
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}
