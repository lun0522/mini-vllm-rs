use crate::models::CachedKeyValue;
use crate::models::ForwardInput;
use crate::models::ForwardOutput;
use crate::models::KvCache;
use crate::models::ModelInfo;
use candle_core::quantized::QTensor;
use candle_core::DType;
use candle_core::Device;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::Embedding;
use candle_nn::Module;
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

const DEFAULT_METAL_GEMV_MAX_ROWS: usize = 4;
const METAL_GEMV_MAX_ROWS: usize = match option_env!("MINI_VLLM_METAL_GEMV_MAX_ROWS") {
    Some(value) => const_str::parse!(value, u32) as usize,
    None => DEFAULT_METAL_GEMV_MAX_ROWS,
};

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

    /// Uses separate GEMV operations for small Metal inputs because Candle's quantized GEMM
    /// kernel performs poorly at very small row counts. Returns `None` when the compile-time
    /// threshold is disabled, the input is not on Metal, or the regular GEMM path is preferable.
    fn forward_metal_rows_with_gemv(&self, input: &Tensor) -> Result<Option<Tensor>> {
        if !input.device().is_metal() {
            return Ok(None);
        }

        // Candle selects its Metal GEMV kernel when the penultimate dimension is one.
        // Split only the two input layouts used by the model backends, preserving rank so the
        // concatenated result has exactly the same shape as a single quantized matmul.
        let row_dim = match input.rank() {
            2 => 0,
            3 if input.dim(0)? == 1 => 1,
            _ => return Ok(None),
        };
        let row_count = input.dim(row_dim)?;
        if row_count <= 1 || row_count > METAL_GEMV_MAX_ROWS {
            return Ok(None);
        }

        let mut outputs = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            let row = input.narrow(row_dim, row_index, /* len */ 1)?;
            outputs.push(self.inner.forward(&row)?);
        }
        let output_refs = outputs.iter().collect::<Vec<_>>();
        Ok(Some(Tensor::cat(&output_refs, row_dim)?))
    }
}

impl Module for QMatMul {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        if let Some(output) = self.forward_metal_rows_with_gemv(input)? {
            return Ok(output);
        }
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

pub(super) struct ForwardContext {
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
    /// Runs one complete transformer block over packed requests.
    pub(super) fn forward(
        &self,
        x: &Tensor,
        contexts: &[ForwardContext],
        layer_index: usize,
        cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.attn_norm.forward(x)?;
        let x = self.forward_attention(&x, contexts, layer_index, cache)?;
        let x = (x + residual)?;
        self.forward_mlp(&x)
    }

    /// Projects requests packed along the query-token dimension, then computes
    /// attention separately for each request to avoid materializing cross-request
    /// attention scores. The per-request outputs are packed again before the
    /// output projection.
    ///
    /// `packed_` values contain data from every request, while `request_` values
    /// contain one request's slice. `full` K/V values include both the cached
    /// prefix and the current query tokens.
    pub(super) fn forward_attention(
        &self,
        packed_x: &Tensor,
        contexts: &[ForwardContext],
        layer_index: usize,
        cache: &mut dyn KvCache,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();

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
        generation_inputs: &[ForwardInput],
        verification_inputs: &[ForwardInput],
        kv_cache: &mut dyn KvCache,
    ) -> Result<ForwardOutput> {
        let _enter_model = self.span_model.enter();

        let combined_inputs = generation_inputs
            .iter()
            .chain(verification_inputs)
            .map(|input| ForwardInput {
                request_id: input.request_id,
                input: input.input.clone(),
                start_position: input.start_position,
            })
            .collect::<Vec<_>>();

        // packed_input: [1, packed_q_len].
        let (packed_input, contexts) = prepare_forward(&combined_inputs)?;
        // packed_hidden_states: [1, packed_q_len, embedding_len].
        let mut packed_hidden_states = self.token_embeddings.forward(&packed_input)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            packed_hidden_states =
                layer.forward(&packed_hidden_states, &contexts, layer_index, kv_cache)?;
        }
        let packed_hidden_states = self.output_norm.forward(&packed_hidden_states)?;

        let (generation_contexts, verification_contexts) =
            contexts.split_at(generation_inputs.len());
        let packed_output_hidden_states = pack_output_hidden_states(
            &packed_hidden_states,
            generation_contexts,
            verification_contexts,
        )?;

        let _enter_output = self.span_output.enter();
        // packed_logits:
        // [generation_request_count + verification_token_count, vocabulary_size].
        let packed_logits = self.output_proj.forward(&packed_output_hidden_states)?;
        create_forward_output(
            &packed_logits,
            generation_inputs.len(),
            verification_contexts,
        )
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

fn validate_forward_inputs(inputs: &[ForwardInput]) -> Result<()> {
    if inputs.is_empty() {
        candle_core::bail!("forward requires at least one request")
    }
    for input in inputs {
        let (batch_size, q_len) = input.input.dims2()?;
        if batch_size != 1 {
            candle_core::bail!("each forward input must have batch size 1")
        }
        if q_len == 0 {
            candle_core::bail!("each forward input must contain at least one token")
        }
    }
    Ok(())
}

fn prepare_forward(inputs: &[ForwardInput]) -> Result<(Tensor, Vec<ForwardContext>)> {
    validate_forward_inputs(inputs)?;

    // Each input: [1, q_len].
    let mut input_tensors = Vec::with_capacity(inputs.len());
    let mut contexts = Vec::with_capacity(inputs.len());
    let mut q_start_index = 0;
    for input in inputs {
        let (_, q_len) = input.input.dims2()?;
        input_tensors.push(&input.input);
        contexts.push(ForwardContext {
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
    generation_contexts: &[ForwardContext],
    verification_contexts: &[ForwardContext],
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
    contexts: &[ForwardContext],
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

fn create_forward_output(
    packed_logits: &Tensor,
    generation_request_count: usize,
    verification_contexts: &[ForwardContext],
) -> Result<ForwardOutput> {
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
    Ok(ForwardOutput {
        generation_logits,
        verification_logits,
    })
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)
}
