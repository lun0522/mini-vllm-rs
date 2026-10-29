use tracing::span::EnteredSpan;

// Colors identify trace entry points; per-range fields only affect the label.
const MODEL_COLOR: u32 = 0xFF0072B2;
const LAYER_COLOR: u32 = 0xFFD55E00;
const ATTENTION_COLOR: u32 = 0xFF009E73;
const QMATMUL_COLOR: u32 = 0xFFCC79A7;
const ROPE_COLOR: u32 = 0xFFE69F00;
const ATTENTION_CACHE_APPEND_COLOR: u32 = 0xFF56B4E9;
const ATTENTION_CACHE_ACCESS_COLOR: u32 = 0xFFF0E442;
const ATTENTION_SDPA_COLOR: u32 = 0xFF882255;
const ATTENTION_REPEAT_KV_COLOR: u32 = 0xFF44AA99;
const ATTENTION_QUERY_KEY_COLOR: u32 = 0xFF332288;
const ATTENTION_MASK_SOFTMAX_COLOR: u32 = 0xFFAA4499;
const ATTENTION_PAGED_VALUE_COLOR: u32 = 0xFF999933;
const ATTENTION_VALUE_COLOR: u32 = 0xFFDDCC77;
const MLP_COLOR: u32 = 0xFF117733;
const OUTPUT_COLOR: u32 = 0xFFEE7733;

pub(super) struct TraceGuard {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    _nvtx: Option<super::nvtx_trace::NvtxRangeGuard>,
    _span: EnteredSpan,
}

fn enter(span: tracing::Span, nvtx_name: impl FnOnce() -> String, color: u32) -> TraceGuard {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let nvtx = super::nvtx_trace::enter(nvtx_name, color);

    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    let _ = (nvtx_name, color);

    TraceGuard {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        _nvtx: nvtx,
        _span: span.entered(),
    }
}

macro_rules! trace_span {
    ($name:literal, $color:expr $(,)?) => {
        enter(tracing::trace_span!($name), || $name.to_owned(), $color)
    };
    ($name:literal, $color:expr; $($field:ident = $value:expr),+ $(,)?) => {{
        $(let $field = $value;)+
        let span = tracing::trace_span!($name, $($field = $field),+);
        enter(span, || {
            format!(
                concat!($name $(, " ", stringify!($field), "={}")+),
                $($field),+
            )
        }, $color)
    }};
}

pub(super) fn model() -> TraceGuard {
    trace_span!("model", MODEL_COLOR)
}

pub(super) fn layer(layer_index: usize) -> TraceGuard {
    trace_span!("layer", LAYER_COLOR; layer_index = layer_index)
}

pub(super) fn attention() -> TraceGuard {
    trace_span!("attn", ATTENTION_COLOR)
}

pub(super) fn qmatmul(operation: &'static str, implementation: &'static str) -> TraceGuard {
    trace_span!(
        "qmatmul",
        QMATMUL_COLOR;
        operation = operation,
        implementation = implementation,
    )
}

pub(super) fn rope(tensor: &'static str) -> TraceGuard {
    trace_span!("attn-rope", ROPE_COLOR; tensor = tensor)
}

pub(super) fn attention_cache_append() -> TraceGuard {
    trace_span!("attn-cache", ATTENTION_CACHE_APPEND_COLOR; operation = "append")
}

pub(super) fn attention_cache_access(result_layout: &'static str) -> TraceGuard {
    trace_span!(
        "attn-cache",
        ATTENTION_CACHE_ACCESS_COLOR;
        operation = "access",
        result_layout = result_layout,
    )
}

pub(super) fn attention_sdpa(implementation: &'static str) -> TraceGuard {
    trace_span!("attn-sdpa", ATTENTION_SDPA_COLOR; implementation = implementation)
}

pub(super) fn attention_repeat_kv(tensor: &'static str) -> TraceGuard {
    trace_span!("attn-repeat-kv", ATTENTION_REPEAT_KV_COLOR; tensor = tensor)
}

pub(super) fn attention_query_key(implementation: &'static str) -> TraceGuard {
    trace_span!("attn-qk", ATTENTION_QUERY_KEY_COLOR; implementation = implementation)
}

pub(super) fn attention_mask_softmax() -> TraceGuard {
    trace_span!("attn-mask-softmax", ATTENTION_MASK_SOFTMAX_COLOR)
}

pub(super) fn attention_paged_value(value_layout: &'static str) -> TraceGuard {
    trace_span!("attn-paged-v", ATTENTION_PAGED_VALUE_COLOR; value_layout = value_layout)
}

pub(super) fn attention_value(implementation: &'static str) -> TraceGuard {
    trace_span!("attn-v", ATTENTION_VALUE_COLOR; implementation = implementation)
}

pub(super) fn mlp() -> TraceGuard {
    trace_span!("mlp", MLP_COLOR)
}

pub(super) fn output() -> TraceGuard {
    trace_span!("output", OUTPUT_COLOR)
}
