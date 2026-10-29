use tracing::span::EnteredSpan;

pub(super) struct TraceGuard {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    _nvtx: Option<super::nvtx_trace::NvtxRangeGuard>,
    _span: EnteredSpan,
}

fn enter(span: tracing::Span, nvtx_name: impl FnOnce() -> String) -> TraceGuard {
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    let nvtx = super::nvtx_trace::enter(nvtx_name);

    #[cfg(not(all(target_os = "linux", feature = "cuda")))]
    let _ = nvtx_name;

    TraceGuard {
        #[cfg(all(target_os = "linux", feature = "cuda"))]
        _nvtx: nvtx,
        _span: span.entered(),
    }
}

macro_rules! trace_span {
    ($name:literal $(,)?) => {
        enter(tracing::trace_span!($name), || $name.to_owned())
    };
    ($name:literal, $($field:ident = $value:expr),+ $(,)?) => {{
        $(let $field = $value;)+
        let span = tracing::trace_span!($name, $($field = $field),+);
        enter(span, || {
            format!(
                concat!($name $(, " ", stringify!($field), "={}")+),
                $($field),+
            )
        })
    }};
}

pub(super) fn model() -> TraceGuard {
    trace_span!("model")
}

pub(super) fn layer(layer_index: usize) -> TraceGuard {
    trace_span!("layer", layer_index = layer_index)
}

pub(super) fn attention() -> TraceGuard {
    trace_span!("attn")
}

pub(super) fn qmatmul(operation: &'static str, implementation: &'static str) -> TraceGuard {
    trace_span!(
        "qmatmul",
        operation = operation,
        implementation = implementation,
    )
}

pub(super) fn rope(tensor: &'static str) -> TraceGuard {
    trace_span!("attn-rope", tensor = tensor)
}

pub(super) fn attention_cache_append() -> TraceGuard {
    trace_span!("attn-cache", operation = "append")
}

pub(super) fn attention_cache_access(result_layout: &'static str) -> TraceGuard {
    trace_span!(
        "attn-cache",
        operation = "access",
        result_layout = result_layout,
    )
}

pub(super) fn attention_sdpa(implementation: &'static str) -> TraceGuard {
    trace_span!("attn-sdpa", implementation = implementation)
}

pub(super) fn attention_repeat_kv(tensor: &'static str) -> TraceGuard {
    trace_span!("attn-repeat-kv", tensor = tensor)
}

pub(super) fn attention_query_key(implementation: &'static str) -> TraceGuard {
    trace_span!("attn-qk", implementation = implementation)
}

pub(super) fn attention_mask_softmax() -> TraceGuard {
    trace_span!("attn-mask-softmax")
}

pub(super) fn attention_paged_value(value_layout: &'static str) -> TraceGuard {
    trace_span!("attn-paged-v", value_layout = value_layout)
}

pub(super) fn attention_value(implementation: &'static str) -> TraceGuard {
    trace_span!("attn-v", implementation = implementation)
}

pub(super) fn mlp() -> TraceGuard {
    trace_span!("mlp")
}

pub(super) fn output() -> TraceGuard {
    trace_span!("output")
}
