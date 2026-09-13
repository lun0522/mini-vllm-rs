use crate::models::CachedKeyValue;
use crate::models::ForwardContext;
use crate::models::KvCache;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::Module;
use candle_transformers::utils::repeat_kv;

#[derive(Clone, Copy)]
pub(super) enum RotaryEmbeddingType {
    Neox,
    Interleaved,
}

pub(super) struct AttentionParameters<'a> {
    pub(super) query_proj: &'a dyn Module,
    pub(super) key_proj: &'a dyn Module,
    pub(super) value_proj: &'a dyn Module,
    pub(super) output_proj: &'a dyn Module,
    pub(super) query_bias: Option<&'a Tensor>,
    pub(super) key_bias: Option<&'a Tensor>,
    pub(super) value_bias: Option<&'a Tensor>,
    pub(super) query_head_count: usize,
    pub(super) key_value_head_count: usize,
    pub(super) head_dim: usize,
    pub(super) cos: &'a Tensor,
    pub(super) sin: &'a Tensor,
    pub(super) neg_inf: &'a Tensor,
    pub(super) rotary_embedding_type: RotaryEmbeddingType,
    pub(super) attention_span: &'a tracing::Span,
    pub(super) rotary_span: &'a tracing::Span,
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

pub(super) fn forward_attention(
    params: AttentionParameters<'_>,
    x: &Tensor,
    mask: Option<&Tensor>,
    context: &ForwardContext,
    layer_index: usize,
    kv_cache: &mut dyn KvCache,
) -> Result<Tensor> {
    let _enter = params.attention_span.enter();

    let index_position = context.start_position;
    let (batch_size, query_length, embedding_length) = x.dims3()?;

    // query_length is the number of tokens in this forward call.
    // key_value_length also includes tokens already stored in the KV cache.
    // x, q: [batch_size, query_length, embedding_length]
    // k, v: [batch_size, query_length, key_value_head_count * head_dim]
    // key_value_head_count * head_dim can be less than embedding_length for GQA or MQA.
    // Optional biases are broadcast across batch_size and query_length without
    // changing the dimensions of q, k, or v.
    let q = apply_projection(x, params.query_proj, params.query_bias)?;
    let k = apply_projection(x, params.key_proj, params.key_bias)?;
    let v = apply_projection(x, params.value_proj, params.value_bias)?;

    // After transpose:
    // q: [batch_size, query_head_count, query_length, head_dim]
    // k, v: [batch_size, key_value_head_count, query_length, head_dim]
    let q = q
        .reshape((
            batch_size,
            query_length,
            params.query_head_count,
            params.head_dim,
        ))?
        .transpose(1, 2)?
        .contiguous()?;
    let k = k
        .reshape((
            batch_size,
            query_length,
            params.key_value_head_count,
            params.head_dim,
        ))?
        .transpose(1, 2)?
        .contiguous()?;
    let v = v
        .reshape((
            batch_size,
            query_length,
            params.key_value_head_count,
            params.head_dim,
        ))?
        .transpose(1, 2)?
        .contiguous()?;

    // Rotary embeddings preserve the dimensions of q and k.
    let q = apply_rotary_embedding(
        params.rotary_embedding_type,
        &q,
        index_position,
        params.cos,
        params.sin,
        params.rotary_span,
    )?;
    let k = apply_rotary_embedding(
        params.rotary_embedding_type,
        &k,
        index_position,
        params.cos,
        params.sin,
        params.rotary_span,
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
            1. / (params.head_dim as f32).sqrt(),
            1.,
        )?
    } else {
        // Expand grouped-query or multi-query heads to match q:
        // [batch_size, query_head_count, key_value_length, head_dim].
        let repetition_count = params.query_head_count / params.key_value_head_count;
        let k = repeat_kv(k, repetition_count)?;
        let v = repeat_kv(v, repetition_count)?;

        // Attention scores:
        // [batch_size, query_head_count, query_length, key_value_length].
        let attention = (q.matmul(&k.t()?)? / (params.head_dim as f64).sqrt())?;
        let attention = match mask {
            None => attention,
            Some(mask) => {
                let mask = mask.broadcast_as(attention.shape())?;
                masked_fill(&attention, &mask, params.neg_inf)?
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
    params.output_proj.forward(&y)
}
