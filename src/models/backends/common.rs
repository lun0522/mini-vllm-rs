use crate::models::BatchedForwardInput;
use crate::models::BatchedForwardOutput;
use crate::models::BatchedKvCache;
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

pub(super) struct BatchedForwardContext {
    pub(super) request_id: u64,
    pub(super) start_pos: usize,
    pub(super) cached_kv_len: usize,
    pub(super) q_start_index: usize,
    pub(super) q_len: usize,
}

#[derive(Debug, Clone)]
pub(super) struct RotaryEmbeddingContext {
    pub(super) rope_type: RotaryEmbeddingType,
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
    pub(super) span_rope: tracing::Span,
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
    pub(super) rope_context: RotaryEmbeddingContext,
    pub(super) neg_inf: Tensor,
    pub(super) span_attn: tracing::Span,
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

    /// Runs one complete transformer block over packed requests.
    pub(super) fn forward_batched(
        &self,
        x: &Tensor,
        contexts: &[BatchedForwardContext],
        layer_index: usize,
        cache: &mut dyn BatchedKvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let x = self.forward_batched_attention(&x, contexts, layer_index, cache)?;
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
        let q = self.apply_rotary_embedding(&q, index_pos)?;
        let k = self.apply_rotary_embedding(&k, index_pos)?;

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

    /// Projects requests packed along the query-token dimension, then computes
    /// attention separately for each request to avoid materializing cross-request
    /// attention scores. The per-request outputs are packed again before the
    /// output projection.
    ///
    /// `packed_` values contain data from every request, while `request_` values
    /// contain one request's slice. `full` K/V values include both the cached
    /// prefix and the current query tokens.
    pub(super) fn forward_batched_attention(
        &self,
        packed_x: &Tensor,
        contexts: &[BatchedForwardContext],
        layer_index: usize,
        cache: &mut dyn BatchedKvCache,
    ) -> Result<Tensor> {
        // packed_x: [1, packed_q_len, embedding_len].
        // packed_q: [1, num_q_heads, packed_q_len, head_dim].
        // packed_k, packed_v: [1, num_kv_heads, packed_q_len, head_dim].
        let (_, packed_q_len, embedding_len) = packed_x.dims3()?;
        let packed_q = apply_projection(packed_x, &self.attn_wq, self.attn_bq.as_ref())?
            .reshape((1, packed_q_len, self.num_q_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let packed_k = apply_projection(packed_x, &self.attn_wk, self.attn_bk.as_ref())?
            .reshape((1, packed_q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let packed_v = apply_projection(packed_x, &self.attn_wv, self.attn_bv.as_ref())?
            .reshape((1, packed_q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let mut request_attention_outputs = Vec::with_capacity(contexts.len());
        for context in contexts {
            // Narrowing packed Q/K preserves packed strides, so materialize each
            // request slice before passing it to Candle's contiguous-only RoPE kernels.
            // request_q: [1, num_q_heads, q_len, head_dim].
            // request_k, request_v: [1, num_kv_heads, q_len, head_dim].
            let request_q = packed_q
                .narrow(/* dim */ 2, context.q_start_index, context.q_len)?
                .contiguous()?;
            let request_k = packed_k
                .narrow(/* dim */ 2, context.q_start_index, context.q_len)?
                .contiguous()?;
            let request_v =
                packed_v.narrow(/* dim */ 2, context.q_start_index, context.q_len)?;
            let request_q = self.apply_rotary_embedding(&request_q, context.start_pos)?;
            let request_k = self.apply_rotary_embedding(&request_k, context.start_pos)?;

            let CachedKeyValue {
                key: request_full_cached_k,
                value: request_full_cached_v,
            } = cache
                .append(
                    context.request_id,
                    layer_index,
                    context.cached_kv_len,
                    &request_k,
                    &request_v,
                )
                .map_err(candle_core::Error::wrap)?;
            let (_, _, request_full_cached_kv_len, _) = request_full_cached_k.dims4()?;
            let expected_full_cached_kv_len = context.cached_kv_len + context.q_len;
            if request_full_cached_kv_len != expected_full_cached_kv_len {
                candle_core::bail!("KV cache returned an inconsistent sequence length")
            }

            // request_y: [1, num_q_heads, q_len, head_dim].
            let request_y = if request_q.device().is_metal() && context.q_len == 1 {
                // Metal SDPA handles GQA or MQA without explicitly repeating K and V.
                candle_nn::ops::sdpa(
                    &request_q,
                    &request_full_cached_k,
                    &request_full_cached_v,
                    None,
                    false,
                    1. / (self.head_dim as f32).sqrt(),
                    1.,
                )?
            } else {
                // Expand grouped-query or multi-query heads to match request_q:
                // request_full_cached_k, request_full_cached_v:
                // [1, num_q_heads, request_full_cached_kv_len, head_dim].
                let repetition_count = self.num_q_heads / self.num_kv_heads;
                let request_full_cached_k = repeat_kv(request_full_cached_k, repetition_count)?;
                let request_full_cached_v = repeat_kv(request_full_cached_v, repetition_count)?;

                // request_attn_scores:
                // [1, num_q_heads, q_len, request_full_cached_kv_len].
                let request_attn_scores = (request_q.matmul(&request_full_cached_k.t()?)?
                    / (self.head_dim as f64).sqrt())?;
                // request_mask: [q_len, request_full_cached_kv_len].
                let request_mask = candle_transformers::utils::build_causal_mask(
                    context.q_len,
                    context.cached_kv_len,
                    packed_x.device(),
                )?
                .broadcast_as(request_attn_scores.shape())?;
                let request_attn_scores =
                    masked_fill(&request_attn_scores, &request_mask, &self.neg_inf)?;
                let request_attn_weights = candle_nn::ops::softmax_last_dim(&request_attn_scores)?;
                request_attn_weights.matmul(&request_full_cached_v.contiguous()?)?
            };
            request_attention_outputs.push(request_y);
        }

        // packed_y: [1, num_q_heads, packed_q_len, head_dim].
        let packed_y = Tensor::cat(&request_attention_outputs, /* dim */ 2)?;
        // After transpose and reshape:
        // packed_y: [1, packed_q_len, embedding_len].
        let packed_y = packed_y
            .transpose(1, 2)?
            .reshape((1, packed_q_len, embedding_len))?;
        self.attn_wo.forward(&packed_y)
    }

    fn forward_mlp(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span_mlp.enter();
        let residual = x;
        let x = self.mlp_norm.forward(x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }

    fn apply_rotary_embedding(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let context = &self.rope_context;
        let _enter = context.span_rope.enter();
        let (_, _, query_len, _) = x.dims4()?;
        let cos = context.cos.narrow(0, index_pos, query_len)?;
        let sin = context.sin.narrow(0, index_pos, query_len)?;
        match context.rope_type {
            RotaryEmbeddingType::Neox => candle_nn::rotary_emb::rope(x, &cos, &sin),
            RotaryEmbeddingType::Interleaved => candle_nn::rotary_emb::rope_i(x, &cos, &sin),
        }
    }
}

pub(super) struct TransformerModelWeights {
    token_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    output_norm: RmsNorm,
    output_proj: QMatMul,
    mask_cache: CausalMaskCache,
    span_model: tracing::Span,
    span_output: tracing::Span,
}

impl TransformerModelWeights {
    pub(super) fn new(
        token_embeddings: Embedding,
        layers: Vec<TransformerBlock>,
        output_norm: RmsNorm,
        output_proj: QMatMul,
    ) -> Self {
        Self {
            token_embeddings,
            layers,
            output_norm,
            output_proj,
            mask_cache: CausalMaskCache::default(),
            span_model: tracing::span!(tracing::Level::TRACE, "model"),
            span_output: tracing::span!(tracing::Level::TRACE, "output"),
        }
    }

    pub(super) fn model_info(&self) -> ModelInfo {
        let attention = &self.layers[0];
        ModelInfo {
            layer_count: self.layers.len(),
            num_kv_heads: attention.num_kv_heads,
            head_dim: attention.head_dim,
            activation_dtype: self.token_embeddings.embeddings().dtype(),
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
        self.output_proj.forward(&x)?.squeeze(0)
    }

    pub(super) fn forward_batched_with_speculative_verification(
        &mut self,
        generation_inputs: &[BatchedForwardInput],
        verification_inputs: &[BatchedForwardInput],
        kv_cache: &mut dyn BatchedKvCache,
    ) -> Result<BatchedForwardOutput> {
        let combined_inputs = generation_inputs
            .iter()
            .chain(verification_inputs)
            .map(|input| BatchedForwardInput {
                request_id: input.request_id,
                input: input.input.clone(),
                start_position: input.start_position,
            })
            .collect::<Vec<_>>();
        let (packed_hidden_states, contexts) =
            self.forward_batched_hidden(&combined_inputs, kv_cache)?;

        let (generation_contexts, verification_contexts) =
            contexts.split_at(generation_inputs.len());
        let packed_output_hidden_states = pack_output_hidden_states(
            &packed_hidden_states,
            generation_contexts,
            verification_contexts,
        )?;

        let _enter = self.span_output.enter();
        // packed_logits:
        // [generation_request_count + verification_token_count, vocabulary_size].
        let packed_logits = self.output_proj.forward(&packed_output_hidden_states)?;
        create_batched_forward_output(
            &packed_logits,
            generation_inputs.len(),
            verification_contexts,
        )
    }

    pub(super) fn forward_for_speculative_verification(
        &mut self,
        x: &Tensor,
        context: &ForwardContext,
        kv_cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let x = self.forward_hidden(x, context, kv_cache)?;
        let _enter = self.span_output.enter();
        self.output_proj.forward(&x)
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
        let _enter = self.span_model.enter();
        let mut layer_in = self.token_embeddings.forward(x)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            layer_in = layer.forward(&layer_in, mask.as_ref(), context, layer_index, kv_cache)?;
        }
        self.output_norm.forward(&layer_in)
    }

    fn forward_batched_hidden(
        &mut self,
        inputs: &[BatchedForwardInput],
        kv_cache: &mut dyn BatchedKvCache,
    ) -> Result<(Tensor, Vec<BatchedForwardContext>)> {
        let _enter = self.span_model.enter();
        // packed_input: [1, packed_q_len].
        let (packed_input, contexts) = prepare_batched_forward(inputs)?;
        // packed_hidden_states: [1, packed_q_len, embedding_len].
        let mut packed_hidden_states = self.token_embeddings.forward(&packed_input)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            packed_hidden_states =
                layer.forward_batched(&packed_hidden_states, &contexts, layer_index, kv_cache)?;
        }
        let packed_hidden_states = self.output_norm.forward(&packed_hidden_states)?;
        Ok((packed_hidden_states, contexts))
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

fn validate_batched_forward_inputs(inputs: &[BatchedForwardInput]) -> Result<()> {
    if inputs.is_empty() {
        candle_core::bail!("batched forward requires at least one request")
    }
    for input in inputs {
        let (batch_size, q_len) = input.input.dims2()?;
        if batch_size != 1 {
            candle_core::bail!("each batched forward input must have batch size 1")
        }
        if q_len == 0 {
            candle_core::bail!("each batched forward input must contain at least one token")
        }
    }
    Ok(())
}

fn prepare_batched_forward(
    inputs: &[BatchedForwardInput],
) -> Result<(Tensor, Vec<BatchedForwardContext>)> {
    validate_batched_forward_inputs(inputs)?;

    // Each input: [1, q_len].
    let mut input_tensors = Vec::with_capacity(inputs.len());
    let mut contexts = Vec::with_capacity(inputs.len());
    let mut q_start_index = 0;
    for input in inputs {
        let (_, q_len) = input.input.dims2()?;
        input_tensors.push(&input.input);
        contexts.push(BatchedForwardContext {
            request_id: input.request_id,
            start_pos: input.start_position,
            cached_kv_len: input.start_position,
            q_start_index,
            q_len,
        });
        q_start_index += q_len;
    }

    // packed_input: [1, packed_q_len].
    let packed_input = Tensor::cat(&input_tensors, /* dim */ 1)?;
    Ok((packed_input, contexts))
}

fn pack_output_hidden_states(
    packed_hidden_states: &Tensor,
    generation_contexts: &[BatchedForwardContext],
    verification_contexts: &[BatchedForwardContext],
) -> Result<Tensor> {
    let mut output_hidden_states = if generation_contexts.is_empty() {
        Vec::with_capacity(verification_contexts.len())
    } else {
        let mut states = Vec::with_capacity(1 + verification_contexts.len());
        // generation_hidden_states: [generation_request_count, embedding_len].
        states.push(pack_generation_output_hidden_states(
            packed_hidden_states,
            generation_contexts,
        )?);
        states
    };
    for context in verification_contexts {
        // verification_hidden_state: [q_len, embedding_len].
        output_hidden_states.push(
            packed_hidden_states
                .narrow(/* dim */ 1, context.q_start_index, context.q_len)?
                .squeeze(0)?,
        );
    }
    // packed_output_hidden_states:
    // [generation_request_count + total_verification_token_count, embedding_len]
    Tensor::cat(&output_hidden_states, /* dim */ 0)
}

fn pack_generation_output_hidden_states(
    packed_hidden_states: &Tensor,
    contexts: &[BatchedForwardContext],
) -> Result<Tensor> {
    let mut request_last_hidden_states = Vec::with_capacity(contexts.len());
    for context in contexts {
        // request_last_hidden_state: [1, 1, embedding_len].
        let request_last_hidden_state = packed_hidden_states.narrow(
            /* dim */ 1,
            context.q_start_index + context.q_len - 1,
            /* len */ 1,
        )?;
        request_last_hidden_states.push(request_last_hidden_state);
    }
    // packed_last_hidden_states: [request_count, embedding_len].
    Tensor::cat(&request_last_hidden_states, /* dim */ 1)?.squeeze(0)
}

fn create_batched_forward_output(
    packed_logits: &Tensor,
    generation_request_count: usize,
    verification_contexts: &[BatchedForwardContext],
) -> Result<BatchedForwardOutput> {
    let generation_logits = (0..generation_request_count)
        .map(|request_index| {
            packed_logits
                .narrow(/* dim */ 0, request_index, /* len */ 1)?
                .squeeze(0)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut verification_start_index = generation_request_count;
    let verification_logits = verification_contexts
        .iter()
        .map(|context| {
            let logits =
                packed_logits.narrow(/* dim */ 0, verification_start_index, context.q_len)?;
            verification_start_index += context.q_len;
            Ok(logits)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BatchedForwardOutput {
        generation_logits,
        verification_logits,
    })
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}
