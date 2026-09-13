use crate::models::CachedKeyValue;
use crate::models::ForwardContext;
use crate::models::KvCache;
use candle_core::quantized::QTensor;
use candle_core::DType;
use candle_core::Device;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::Module;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

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
    pub(super) feed_forward_w1: QMatMul,
    pub(super) feed_forward_w2: QMatMul,
    pub(super) feed_forward_w3: QMatMul,
}

impl Module for SwiGluMlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(input)?;
        let w3 = self.feed_forward_w3.forward(input)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
pub(super) struct TransformerBlock {
    pub(super) attention_wq: QMatMul,
    pub(super) attention_wk: QMatMul,
    pub(super) attention_wv: QMatMul,
    pub(super) attention_wo: QMatMul,
    pub(super) attention_bq: Option<Tensor>,
    pub(super) attention_bk: Option<Tensor>,
    pub(super) attention_bv: Option<Tensor>,
    pub(super) attention_norm: RmsNorm,
    pub(super) mlp: SwiGluMlp,
    pub(super) ffn_norm: RmsNorm,
    pub(super) query_head_count: usize,
    pub(super) key_value_head_count: usize,
    pub(super) head_dim: usize,
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
    pub(super) neg_inf: Tensor,
    pub(super) rotary_embedding_type: RotaryEmbeddingType,
    pub(super) attention_span: tracing::Span,
    pub(super) rotary_span: tracing::Span,
    pub(super) mlp_span: tracing::Span,
}

impl TransformerBlock {
    pub(super) fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        context: &ForwardContext,
        layer_index: usize,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let _enter = self.attention_span.enter();

        let index_position = context.start_position;
        let (batch_size, query_length, embedding_length) = x.dims3()?;

        // query_length is the number of tokens in this forward call.
        // key_value_length also includes tokens already stored in the KV cache.
        // x, q: [batch_size, query_length, embedding_length]
        // k, v: [batch_size, query_length, key_value_head_count * head_dim]
        // key_value_head_count * head_dim can be less than embedding_length for GQA or MQA.
        // Optional biases are broadcast across batch_size and query_length without
        // changing the dimensions of q, k, or v.
        let q = apply_projection(x, &self.attention_wq, self.attention_bq.as_ref())?;
        let k = apply_projection(x, &self.attention_wk, self.attention_bk.as_ref())?;
        let v = apply_projection(x, &self.attention_wv, self.attention_bv.as_ref())?;

        // After transpose:
        // q: [batch_size, query_head_count, query_length, head_dim]
        // k, v: [batch_size, key_value_head_count, query_length, head_dim]
        let q = q
            .reshape((
                batch_size,
                query_length,
                self.query_head_count,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((
                batch_size,
                query_length,
                self.key_value_head_count,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((
                batch_size,
                query_length,
                self.key_value_head_count,
                self.head_dim,
            ))?
            .transpose(1, 2)?
            .contiguous()?;

        // Rotary embeddings preserve the dimensions of q and k.
        let q = apply_rotary_embedding(
            self.rotary_embedding_type,
            &q,
            index_position,
            &self.cos,
            &self.sin,
            &self.rotary_span,
        )?;
        let k = apply_rotary_embedding(
            self.rotary_embedding_type,
            &k,
            index_position,
            &self.cos,
            &self.sin,
            &self.rotary_span,
        )?;

        // Cached k and v:
        // [batch_size, key_value_head_count, key_value_length, head_dim].
        let CachedKeyValue { key: k, value: v } = kv_cache
            .append(context, layer_index, &k, &v)
            .map_err(candle_core::Error::wrap)?;

        // y: [batch_size, query_head_count, query_length, head_dim].
        let y = if q.device().is_metal() && query_length == 1 {
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
            // [batch_size, query_head_count, key_value_length, head_dim].
            let repetition_count = self.query_head_count / self.key_value_head_count;
            let k = repeat_kv(k, repetition_count)?;
            let v = repeat_kv(v, repetition_count)?;

            // Attention scores:
            // [batch_size, query_head_count, query_length, key_value_length].
            let attention = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let attention = match mask {
                None => attention,
                Some(mask) => {
                    let mask = mask.broadcast_as(attention.shape())?;
                    masked_fill(&attention, &mask, &self.neg_inf)?
                }
            };
            let attention = candle_nn::ops::softmax_last_dim(&attention)?;

            // Convert to contiguous as matmul doesn't support arbitrary strided v tensors.
            attention.matmul(&v.contiguous()?)?
        };

        // After transpose and reshape:
        // [batch_size, query_length, embedding_length].
        let y = y
            .transpose(1, 2)?
            .reshape(&[batch_size, query_length, embedding_length])?;

        // The output projection preserves
        // [batch_size, query_length, embedding_length].
        self.attention_wo.forward(&y)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum RotaryEmbeddingType {
    Neox,
    Interleaved,
}

pub(super) fn precomput_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    context_length: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, context_length as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((context_length, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

fn apply_projection(
    input: &Tensor,
    projection: &dyn Module,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let projected = projection.forward(input)?;
    match bias {
        Some(bias) => projected.broadcast_add(bias),
        None => Ok(projected),
    }
}

fn apply_rotary_embedding(
    rotary_embedding_type: RotaryEmbeddingType,
    x: &Tensor,
    index_position: usize,
    cos: &Tensor,
    sin: &Tensor,
    span: &tracing::Span,
) -> Result<Tensor> {
    let _enter = span.enter();
    let (_batch_size, _head_count, query_length, _head_dim) = x.dims4()?;
    let cos = cos.narrow(0, index_position, query_length)?;
    let sin = sin.narrow(0, index_position, query_length)?;
    match rotary_embedding_type {
        RotaryEmbeddingType::Neox => candle_nn::rotary_emb::rope(x, &cos, &sin),
        RotaryEmbeddingType::Interleaved => candle_nn::rotary_emb::rope_i(x, &cos, &sin),
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}
