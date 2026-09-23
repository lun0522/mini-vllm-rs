use super::trace;
use crate::model_runner::ActivationDType;
use crate::models::ContiguousCacheTensors;
use crate::models::ForwardInput;
use crate::models::ForwardOutput;
use crate::models::KvCache;
use crate::models::ModelInfo;
use crate::models::PagedCacheLayout;
use candle_core::quantized::QTensor;
use candle_core::DType;
use candle_core::Device;
use candle_core::Result;
use candle_core::Tensor;
use candle_nn::Embedding;
use candle_nn::Module;
use candle_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;

const DEFAULT_ENABLE_CPU_GROUPED_QUERY_MATMUL: bool = true;
const DEFAULT_ENABLE_CPU_PAGED_ATTENTION: bool = false;
const DEFAULT_ENABLE_CPU_PAGEWISE_VALUE_MATMUL: bool = false;
const DEFAULT_METAL_GEMV_MAX_ROWS: usize = 4;

const ENABLE_CPU_GROUPED_QUERY_MATMUL: bool =
    match option_env!("MINI_VLLM_ENABLE_CPU_GROUPED_QUERY_MATMUL") {
        Some(value) => const_str::parse!(value, bool),
        None => DEFAULT_ENABLE_CPU_GROUPED_QUERY_MATMUL,
    };

const ENABLE_CPU_PAGED_ATTENTION: bool = match option_env!("MINI_VLLM_ENABLE_CPU_PAGED_ATTENTION") {
    Some(value) => const_str::parse!(value, bool),
    None => DEFAULT_ENABLE_CPU_PAGED_ATTENTION,
};

const ENABLE_CPU_PAGEWISE_VALUE_MATMUL: bool =
    match option_env!("MINI_VLLM_ENABLE_CPU_PAGEWISE_VALUE_MATMUL") {
        Some(value) => const_str::parse!(value, bool),
        None => DEFAULT_ENABLE_CPU_PAGEWISE_VALUE_MATMUL,
    };

const METAL_GEMV_MAX_ROWS: usize = match option_env!("MINI_VLLM_METAL_GEMV_MAX_ROWS") {
    Some(value) => const_str::parse!(value, u32) as usize,
    None => DEFAULT_METAL_GEMV_MAX_ROWS,
};

// QMatMul wrapper adding tracing.
#[derive(Debug, Clone)]
pub(super) struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    operation: &'static str,
}

impl QMatMul {
    pub(super) fn from_qtensor(qtensor: QTensor, operation: &'static str) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner, operation })
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

        let _enter = trace::qmatmul(self.operation, "metal-rowwise-gemv");
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
        // TODO: Revisit CPU F16 projection performance. Casting inputs to F32 around the
        // optimized quantized matmul was close to F32 throughput in benchmarks, and the final
        // output projection can retain F32 logits to avoid an F16-to-F32 sampling round trip.
        if let Some(output) = self.forward_metal_rows_with_gemv(input)? {
            return Ok(output);
        }
        let _enter = trace::qmatmul(self.operation, "quantized-matmul");
        self.inner.forward(input)
    }
}

pub(super) fn dequantize_to_activation_dtype(
    tensor: &QTensor,
    device: &Device,
    activation_dtype: ActivationDType,
) -> Result<Tensor> {
    match activation_dtype {
        ActivationDType::F32 => tensor.dequantize(device),
        ActivationDType::F16 => tensor.dequantize_f16(device),
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

#[derive(Clone, Copy)]
pub(super) struct ForwardContext {
    pub(super) request_id: u64,
    pub(super) start_pos: usize,
    pub(super) cached_kv_len: usize,
    pub(super) q_start_index: usize,
    pub(super) q_len: usize,
}

#[derive(Clone, Copy)]
struct AttentionDimensions {
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    cached_kv_len: usize,
    q_len: usize,
}

impl AttentionDimensions {
    fn full_kv_len(self) -> usize {
        self.cached_kv_len + self.q_len
    }

    fn repetition_count(self) -> usize {
        self.num_q_heads / self.num_kv_heads
    }
}

#[derive(Debug, Clone)]
pub(super) struct RotaryEmbeddingContext {
    pub(super) rope_type: RotaryEmbeddingType,
    pub(super) cos: Tensor,
    pub(super) sin: Tensor,
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
        let _enter = trace::layer(layer_index);
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
        let _enter = trace::attention();

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
        for &context in contexts {
            let ForwardContext {
                request_id,
                start_pos,
                cached_kv_len,
                q_start_index,
                q_len,
            } = context;

            // Narrowing packed Q/K preserves packed strides, so materialize each
            // request slice before passing it to Candle's contiguous-only RoPE kernels.
            // request_q: [1, num_q_heads, q_len, head_dim].
            // request_k, request_v: [1, num_kv_heads, q_len, head_dim].
            let request_q = packed_q
                .narrow(/* dim */ 2, q_start_index, q_len)?
                .contiguous()?;
            let request_k = packed_k
                .narrow(/* dim */ 2, q_start_index, q_len)?
                .contiguous()?;
            let request_v = packed_v.narrow(/* dim */ 2, q_start_index, q_len)?;
            let request_q = self.forward_rope(&request_q, start_pos, "query")?;
            let request_k = self.forward_rope(&request_k, start_pos, "key")?;

            {
                let _enter = trace::attention_cache_append();
                cache
                    .append_new_key_value(
                        request_id,
                        layer_index,
                        cached_kv_len,
                        &request_k,
                        &request_v,
                    )
                    .map_err(candle_core::Error::wrap)?;
            }
            // request_y: [1, num_q_heads, q_len, head_dim].
            let request_y = self.forward_self_attention(context, layer_index, cache, &request_q)?;
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
        let _enter = trace::mlp();
        let residual = x;
        let x = self.mlp_norm.forward(x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }

    fn forward_rope(
        &self,
        x: &Tensor,
        index_pos: usize,
        tensor_name: &'static str,
    ) -> Result<Tensor> {
        let context = &self.rope_context;
        let _enter = trace::rope(tensor_name);
        let (_, _, query_len, _) = x.dims4()?;
        let cos = context.cos.narrow(/* dim */ 0, index_pos, query_len)?;
        let sin = context.sin.narrow(/* dim */ 0, index_pos, query_len)?;
        match context.rope_type {
            RotaryEmbeddingType::Neox => candle_nn::rotary_emb::rope(x, &cos, &sin),
            RotaryEmbeddingType::Interleaved => candle_nn::rotary_emb::rope_i(x, &cos, &sin),
        }
    }

    fn forward_self_attention(
        &self,
        context: ForwardContext,
        layer_index: usize,
        cache: &mut dyn KvCache,
        request_q: &Tensor,
    ) -> Result<Tensor> {
        let dimensions = AttentionDimensions {
            num_q_heads: self.num_q_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            cached_kv_len: context.cached_kv_len,
            q_len: context.q_len,
        };
        if let Some(request_y) =
            self.forward_cpu_paged_attention(context, dimensions, layer_index, cache, request_q)?
        {
            return Ok(request_y);
        }

        // request_full_k, request_full_v: [1, num_kv_heads, request_kv_len, head_dim].
        let ContiguousCacheTensors {
            key: request_full_k,
            value: request_full_v,
        } = {
            let _enter = trace::attention_cache_access("contiguous");
            cache
                .get_contiguous_cache_tensors(context.request_id, layer_index)
                .map_err(candle_core::Error::wrap)?
        };
        let (_, _, request_kv_len, _) = request_full_k.dims4()?;
        let expected_request_kv_len = dimensions.full_kv_len();
        if request_kv_len != expected_request_kv_len {
            candle_core::bail!(
                "KV cache returned sequence length {request_kv_len}, expected \
                 {expected_request_kv_len}"
            )
        }

        // Metal SDPA handles GQA or MQA without explicitly repeating K and V.
        if request_q.device().is_metal() && dimensions.q_len == 1 {
            let _enter = trace::attention_sdpa("metal");
            return candle_nn::ops::sdpa(
                request_q,
                &request_full_k,
                &request_full_v,
                None,
                false,
                1. / (self.head_dim as f32).sqrt(),
                1.,
            );
        }

        forward_contiguous_attention(
            request_q,
            request_full_k,
            request_full_v,
            dimensions,
            ENABLE_CPU_GROUPED_QUERY_MATMUL,
            &self.neg_inf,
        )
    }

    fn forward_cpu_paged_attention(
        &self,
        context: ForwardContext,
        dimensions: AttentionDimensions,
        layer_index: usize,
        cache: &mut dyn KvCache,
        request_q: &Tensor,
    ) -> Result<Option<Tensor>> {
        if !request_q.device().is_cpu() || !ENABLE_CPU_PAGED_ATTENTION {
            return Ok(None);
        }

        let Some(paged_cache_layout) = ({
            let _enter = trace::attention_cache_access("paged");
            cache
                .get_paged_cache_layout(context.request_id, layer_index)
                .map_err(candle_core::Error::wrap)?
        }) else {
            return Ok(None);
        };

        let expected_cached_token_count = dimensions.full_kv_len();
        if paged_cache_layout.cached_token_count != expected_cached_token_count {
            let cached_token_count = paged_cache_layout.cached_token_count;
            candle_core::bail!(
                "paged KV cache reports sequence length {cached_token_count}, expected \
                 {expected_cached_token_count}"
            )
        }

        Ok(Some(forward_paged_attention(
            request_q,
            paged_cache_layout,
            dimensions,
            ENABLE_CPU_GROUPED_QUERY_MATMUL,
            ENABLE_CPU_PAGEWISE_VALUE_MATMUL,
            &self.neg_inf,
        )?))
    }
}

pub(super) struct TransformerModelWeights {
    token_embeddings: Embedding,
    layers: Vec<TransformerBlock>,
    output_norm: RmsNorm,
    output_proj: QMatMul,
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
        let _enter_model = trace::model();

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

        let _enter_output = trace::output();
        let packed_hidden_states = self.output_norm.forward(&packed_hidden_states)?;

        let (generation_contexts, verification_contexts) =
            contexts.split_at(generation_inputs.len());
        let packed_output_hidden_states = pack_output_hidden_states(
            &packed_hidden_states,
            generation_contexts,
            verification_contexts,
        )?;

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
    activation_dtype: ActivationDType,
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
    let cos = idx_theta.cos()?.to_dtype(activation_dtype.into())?;
    let sin = idx_theta.sin()?.to_dtype(activation_dtype.into())?;
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

fn forward_contiguous_attention(
    request_q: &Tensor,
    request_full_k: Tensor,
    request_full_v: Tensor,
    dimensions: AttentionDimensions,
    enable_group_query_heads: bool,
    neg_inf: &Tensor,
) -> Result<Tensor> {
    let request_kv_len = request_full_k.dim(2)?;
    // request_attn_scores: [1, num_q_heads, q_len, kv_len].
    let request_attn_scores = (matmul_query_key(
        request_q,
        request_full_k,
        dimensions,
        request_kv_len,
        enable_group_query_heads,
    )? / (dimensions.head_dim as f64).sqrt())?;
    let request_attn_weights = {
        let _enter = trace::attention_mask_softmax();
        let request_attn_scores = apply_causal_attention_mask(
            request_attn_scores,
            dimensions.q_len,
            dimensions.cached_kv_len,
            neg_inf,
        )?;
        candle_nn::ops::softmax_last_dim(&request_attn_scores)?
    };
    matmul_attention_value(
        request_attn_weights,
        request_full_v,
        dimensions,
        request_kv_len,
        enable_group_query_heads,
    )
}

fn forward_paged_attention(
    request_q: &Tensor,
    paged_cache_layout: PagedCacheLayout,
    dimensions: AttentionDimensions,
    enable_group_query_heads: bool,
    enable_pagewise_value_matmul: bool,
    neg_inf: &Tensor,
) -> Result<Tensor> {
    let PagedCacheLayout {
        key_pool,
        value_pool,
        page_ids,
        per_page_token_count,
        cached_token_count,
    } = paged_cache_layout;

    let mut remaining_token_count = cached_token_count;
    let mut request_attn_score_slices = Vec::new();
    let mut request_v_slices = Vec::new();
    for page_id in page_ids {
        let slice_token_count = per_page_token_count.min(remaining_token_count);
        // request_k_slice, request_v_slice: [1, num_kv_heads, kv_slice_len, head_dim].
        let request_k_slice = get_page_slice_from_pool(&key_pool, page_id, slice_token_count)?;
        let request_v_slice = get_page_slice_from_pool(&value_pool, page_id, slice_token_count)?;
        remaining_token_count -= slice_token_count;

        // request_attn_score_slice: [1, num_q_heads, q_len, kv_slice_len].
        let request_attn_score_slice = matmul_query_key(
            request_q,
            request_k_slice,
            dimensions,
            slice_token_count,
            enable_group_query_heads,
        )?;
        request_attn_score_slices.push(request_attn_score_slice);
        request_v_slices.push(request_v_slice);
    }
    if remaining_token_count != 0 {
        candle_core::bail!(
            "paged KV cache contains fewer pages than its sequence length, leaving \
             {remaining_token_count} tokens uncovered"
        )
    }

    // request_attn_scores: [1, num_q_heads, q_len, kv_len].
    let request_attn_scores = (Tensor::cat(&request_attn_score_slices, /* dim */ 3)?
        / (dimensions.head_dim as f64).sqrt())?;
    let request_attn_weights = {
        let _enter = trace::attention_mask_softmax();
        let request_attn_scores = apply_causal_attention_mask(
            request_attn_scores,
            dimensions.q_len,
            dimensions.cached_kv_len,
            neg_inf,
        )?;
        candle_nn::ops::softmax_last_dim(&request_attn_scores)?
    };

    if enable_pagewise_value_matmul {
        let _enter = trace::attention_paged_value("pagewise");
        matmul_paged_attention_value(
            request_attn_weights,
            request_v_slices,
            dimensions,
            enable_group_query_heads,
        )
    } else {
        let _enter = trace::attention_paged_value("concatenated");
        // request_full_v: [1, num_kv_heads, kv_len, head_dim].
        let request_full_v = Tensor::cat(&request_v_slices, /* dim */ 2)?.contiguous()?;
        matmul_attention_value(
            request_attn_weights,
            request_full_v,
            dimensions,
            cached_token_count,
            enable_group_query_heads,
        )
    }
}

/// Multiplies Q of shape [1, q_heads, q_len, head_dim] by transposed K of shape
/// [1, kv_heads, head_dim, kv_len], returning [1, q_heads, q_len, kv_len] shaped scores.
///
/// - The reference path repeats each K head to match `q_heads`.
/// - The grouped path reshapes Q to [kv_heads, repetitions * q_len, head_dim], treating query
///   heads that share a KV head as additional matrix rows. One batched matmul can then use each K
///   head without copying it before reshaping the result back to the query-head layout. This relies
///   on each KV head's query heads being consecutive in contiguous storage, which the projection
///   and attention tensor layouts in this project guarantee.
fn matmul_query_key(
    request_q: &Tensor,
    request_k: Tensor,
    dimensions: AttentionDimensions,
    kv_len: usize,
    enable_group_query_heads: bool,
) -> Result<Tensor> {
    let repetition_count = dimensions.repetition_count();
    if !enable_group_query_heads || repetition_count == 1 {
        // request_repeated_k: [1, num_q_heads, kv_len, head_dim].
        let request_repeated_k = {
            let _enter = trace::attention_repeat_kv("key");
            repeat_kv(request_k, repetition_count)?
        };
        let _enter = trace::attention_query_key("repeated-kv");
        return request_q.matmul(&request_repeated_k.t()?);
    }

    let _enter = trace::attention_query_key("grouped");
    // grouped_q: [num_kv_heads, repetition_count * q_len, head_dim].
    let grouped_q = request_q.squeeze(0)?.reshape((
        dimensions.num_kv_heads,
        repetition_count * dimensions.q_len,
        dimensions.head_dim,
    ))?;
    // grouped_k: [num_kv_heads, head_dim, kv_len].
    let grouped_k = request_k.squeeze(0)?.transpose(1, 2)?;
    // grouped_attn_scores: [num_kv_heads, repetition_count * q_len, kv_len].
    let grouped_attn_scores = grouped_q.matmul(&grouped_k)?;
    // request_attn_scores: [1, num_q_heads, q_len, kv_len].
    grouped_attn_scores.reshape((1, dimensions.num_q_heads, dimensions.q_len, kv_len))
}

/// Multiplies attention weights of shape [1, q_heads, q_len, kv_len] by V of shape
/// [1, kv_heads, kv_len, head_dim], returning [1, q_heads, q_len, head_dim].
///
/// - The reference path repeats each V head to match `q_heads`.
/// - The grouped path reshapes the weights to [kv_heads, repetitions * q_len, kv_len], so all
///   query heads sharing a KV head consume that V head as separate matrix rows before the output is
///   reshaped back to the query-head layout. This relies on each KV head's query heads being
///   consecutive in contiguous storage, which the attention tensor layout in this project
///   guarantees.
fn matmul_attention_value(
    request_attn_weights: Tensor,
    request_v: Tensor,
    dimensions: AttentionDimensions,
    kv_len: usize,
    enable_group_query_heads: bool,
) -> Result<Tensor> {
    let repetition_count = dimensions.repetition_count();
    if !enable_group_query_heads || repetition_count == 1 {
        // request_repeated_v: [1, num_q_heads, kv_len, head_dim].
        let request_repeated_v = {
            let _enter = trace::attention_repeat_kv("value");
            repeat_kv(request_v, repetition_count)?
        };
        let _enter = trace::attention_value("repeated-kv");
        return request_attn_weights.matmul(&request_repeated_v.contiguous()?);
    }

    let _enter = trace::attention_value("grouped");
    // grouped_weights: [num_kv_heads, repetition_count * q_len, kv_len].
    let grouped_weights = request_attn_weights.reshape((
        dimensions.num_kv_heads,
        repetition_count * dimensions.q_len,
        kv_len,
    ))?;
    // grouped_v: [num_kv_heads, kv_len, head_dim].
    let grouped_v = request_v.squeeze(0)?;
    // grouped_y: [num_kv_heads, repetition_count * q_len, head_dim].
    let grouped_y = grouped_weights.matmul(&grouped_v)?;
    // request_y: [1, num_q_heads, q_len, head_dim].
    grouped_y.reshape((
        1,
        dimensions.num_q_heads,
        dimensions.q_len,
        dimensions.head_dim,
    ))
}

/// Multiplies normalized attention weights of shape [1, q_heads, q_len, kv_len] by V pages of shape
/// [1, kv_heads, page_token_count, head_dim], returning [1, q_heads, q_len, head_dim].
///
/// Each page consumes the matching slice of the full attention weights. The page outputs all have
/// the final attention-output shape, so summing them is equivalent to multiplying by concatenated
/// V while avoiding materializing that full tensor.
fn matmul_paged_attention_value(
    request_attn_weights: Tensor,
    request_v_slices: Vec<Tensor>,
    dimensions: AttentionDimensions,
    enable_group_query_heads: bool,
) -> Result<Tensor> {
    // request_y: [1, num_q_heads, q_len, head_dim].
    let mut request_y: Option<Tensor> = None;
    let mut weight_start = 0;
    for request_v_slice in request_v_slices {
        // request_v_slice: [1, num_kv_heads, slice_token_count, head_dim].
        let slice_token_count = request_v_slice.dim(2)?;
        // request_attn_weight_slice: [1, num_q_heads, q_len, slice_token_count].
        let request_attn_weight_slice =
            request_attn_weights.narrow(/* dim */ 3, weight_start, slice_token_count)?;
        // request_page_y: [1, num_q_heads, q_len, head_dim].
        let request_page_y = matmul_attention_value(
            request_attn_weight_slice,
            request_v_slice.contiguous()?,
            dimensions,
            slice_token_count,
            enable_group_query_heads,
        )?;
        request_y = Some(match request_y {
            Some(request_y) => (request_y + request_page_y)?,
            None => request_page_y,
        });
        weight_start += slice_token_count;
    }
    request_y.ok_or_else(|| {
        candle_core::Error::Msg("paged KV cache contains no value pages".to_string())
    })
}

fn apply_causal_attention_mask(
    request_attn_scores: Tensor,
    q_len: usize,
    cached_kv_len: usize,
    neg_inf: &Tensor,
) -> Result<Tensor> {
    // request_mask: [q_len, cached_kv_len + q_len].
    let request_mask = candle_transformers::utils::build_causal_mask(
        q_len,
        cached_kv_len,
        request_attn_scores.device(),
    )?
    .broadcast_as(request_attn_scores.shape())?;
    request_mask.where_cond(
        &neg_inf.broadcast_as(request_mask.shape().dims())?,
        &request_attn_scores,
    )
}

/// returns a tensor of shape [/* batch */ 1, num_kv_heads, slice_token_count, head_dim].
fn get_page_slice_from_pool(
    pool: &Tensor,
    page_id: usize,
    slice_token_count: usize,
) -> Result<Tensor> {
    // According to the dimension contract of the KvCache trait, a cache pool is a tensor of shape:
    // [page_count, /* batch */ 1, num_kv_heads, per_page_token_count, head_dim]
    pool.narrow(/* dim */ 0, /* start */ page_id, /* len */ 1)?
        .narrow(/* dim */ 3, /* start */ 0, slice_token_count)?
        .squeeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f32 = 1e-5;

    fn tensor4(
        values: &[f32],
        dim_0: usize,
        dim_1: usize,
        dim_2: usize,
        dim_3: usize,
    ) -> Result<Tensor> {
        Tensor::from_vec(values.to_vec(), (dim_0, dim_1, dim_2, dim_3), &Device::Cpu)
    }

    fn neg_inf() -> Result<Tensor> {
        Tensor::new(f32::NEG_INFINITY, &Device::Cpu)
    }

    fn attention_dimensions(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        cached_kv_len: usize,
        q_len: usize,
    ) -> AttentionDimensions {
        AttentionDimensions {
            num_q_heads,
            num_kv_heads,
            head_dim,
            cached_kv_len,
            q_len,
        }
    }

    fn assert_close(actual: &Tensor, expected: &[f32]) -> Result<()> {
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= TOLERANCE,
                "value {index} differs: expected {expected}, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn contiguous_attention_computes_multi_head_attention() -> Result<()> {
        let q = tensor4(&[0., 0.], 1, 2, 1, 1)?;
        let k = tensor4(&[1., 2., 3., 4.], 1, 2, 2, 1)?;
        let v = tensor4(&[2., 4., 10., 14.], 1, 2, 2, 1)?;

        let output = forward_contiguous_attention(
            &q,
            k,
            v,
            attention_dimensions(2, 2, 1, 1, 1),
            false,
            &neg_inf()?,
        )?;

        assert_close(&output, &[3., 12.])
    }

    #[test]
    fn grouped_query_matmul_matches_repeated_kv_attention() -> Result<()> {
        let q = tensor4(&[0., 0., 0., 0.], 1, 4, 1, 1)?;
        let k = tensor4(&[1., 2.], 1, 1, 2, 1)?;
        let v = tensor4(&[2., 6.], 1, 1, 2, 1)?;
        let neg_inf = neg_inf()?;

        let repeated = forward_contiguous_attention(
            &q,
            k.clone(),
            v.clone(),
            attention_dimensions(4, 1, 1, 1, 1),
            false,
            &neg_inf,
        )?;
        let grouped = forward_contiguous_attention(
            &q,
            k,
            v,
            attention_dimensions(4, 1, 1, 1, 1),
            true,
            &neg_inf,
        )?;

        assert_close(&repeated, &[4., 4., 4., 4.])?;
        assert_close(&grouped, &repeated.flatten_all()?.to_vec1::<f32>()?)
    }

    #[test]
    fn contiguous_attention_applies_a_causal_prefill_mask() -> Result<()> {
        let q = tensor4(&[0., 0., 0.], 1, 1, 3, 1)?;
        let k = tensor4(&[1., 2., 3.], 1, 1, 3, 1)?;
        let v = tensor4(&[1., 3., 5.], 1, 1, 3, 1)?;

        let output = forward_contiguous_attention(
            &q,
            k,
            v,
            attention_dimensions(1, 1, 1, 0, 3),
            false,
            &neg_inf()?,
        )?;

        assert_close(&output, &[1., 2., 3.])
    }

    #[test]
    fn f16_rotary_frequencies_and_attention_preserve_dtype() -> Result<()> {
        let (cos, sin) = precompute_rotary_embedding_frequencies(
            2,
            10_000.,
            3,
            &Device::Cpu,
            ActivationDType::F16,
        )?;
        assert_eq!(cos.dtype(), DType::F16);
        assert_eq!(sin.dtype(), DType::F16);

        let q = tensor4(&[0., 0., 0.], 1, 1, 3, 1)?.to_dtype(DType::F16)?;
        let k = tensor4(&[1., 2., 3.], 1, 1, 3, 1)?.to_dtype(DType::F16)?;
        let v = tensor4(&[1., 3., 5.], 1, 1, 3, 1)?.to_dtype(DType::F16)?;
        let neg_inf = neg_inf()?.to_dtype(DType::F16)?;

        let output = forward_contiguous_attention(
            &q,
            k,
            v,
            attention_dimensions(1, 1, 1, 0, 3),
            false,
            &neg_inf,
        )?;

        assert_eq!(output.dtype(), DType::F16);
        assert_close(&output, &[1., 2., 3.])
    }

    #[test]
    fn contiguous_attention_decode_attends_to_the_cached_prefix() -> Result<()> {
        let q = tensor4(&[0.], 1, 1, 1, 1)?;
        let k = tensor4(&[1., 2., 3., 4.], 1, 1, 4, 1)?;
        let v = tensor4(&[1., 3., 5., 7.], 1, 1, 4, 1)?;

        let output = forward_contiguous_attention(
            &q,
            k,
            v,
            attention_dimensions(1, 1, 1, 3, 1),
            false,
            &neg_inf()?,
        )?;

        assert_close(&output, &[4.])
    }

    #[test]
    fn paged_attention_matches_contiguous_attention_across_partial_pages() -> Result<()> {
        let q = tensor4(&[0.2, -0.1, 0.4, 0.3, -0.2, 0.5, 0.1, -0.4], 1, 2, 2, 2)?;
        let k_values = [0.1, 0.2, 0.3, -0.1, -0.2, 0.4, 0.5, 0.6, -0.3, 0.2];
        let v_values = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let contiguous_k = tensor4(&k_values, 1, 1, 5, 2)?;
        let contiguous_v = tensor4(&v_values, 1, 1, 5, 2)?;

        // Logical page order is physical page 2 followed by physical page 0. The final page has
        // two valid tokens; its third slot and all of physical page 1 must not affect attention.
        let key_pool = Tensor::from_vec(
            vec![
                0.5_f32, 0.6, -0.3, 0.2, 99., 99., // physical page 0
                88., 88., 88., 88., 88., 88., // physical page 1
                0.1, 0.2, 0.3, -0.1, -0.2, 0.4, // physical page 2
            ],
            (3, 1, 1, 3, 2),
            &Device::Cpu,
        )?;
        let value_pool = Tensor::from_vec(
            vec![
                7_f32, 8., 9., 10., 99., 99., // physical page 0
                88., 88., 88., 88., 88., 88., // physical page 1
                1., 2., 3., 4., 5., 6., // physical page 2
            ],
            (3, 1, 1, 3, 2),
            &Device::Cpu,
        )?;
        let neg_inf = neg_inf()?;
        let paged_cache_layout = || PagedCacheLayout {
            key_pool: key_pool.clone(),
            value_pool: value_pool.clone(),
            page_ids: vec![2, 0],
            per_page_token_count: 3,
            cached_token_count: 5,
        };

        let contiguous = forward_contiguous_attention(
            &q,
            contiguous_k,
            contiguous_v,
            attention_dimensions(2, 1, 2, 3, 2),
            false,
            &neg_inf,
        )?;
        let contiguous = contiguous.flatten_all()?.to_vec1::<f32>()?;
        for enable_group_query_heads in [false, true] {
            for enable_pagewise_value_matmul in [false, true] {
                let paged = forward_paged_attention(
                    &q,
                    paged_cache_layout(),
                    attention_dimensions(2, 1, 2, 3, 2),
                    enable_group_query_heads,
                    enable_pagewise_value_matmul,
                    &neg_inf,
                )?;
                assert_close(&paged, &contiguous)?;
            }
        }
        Ok(())
    }
}
