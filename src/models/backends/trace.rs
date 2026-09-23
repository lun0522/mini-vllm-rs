use tracing::span::EnteredSpan;

pub(super) fn model() -> EnteredSpan {
    tracing::trace_span!("model").entered()
}

pub(super) fn layer(layer_index: usize) -> EnteredSpan {
    tracing::trace_span!("layer", layer_index = layer_index).entered()
}

pub(super) fn attention() -> EnteredSpan {
    tracing::trace_span!("attn").entered()
}

pub(super) fn qmatmul(operation: &'static str, implementation: &'static str) -> EnteredSpan {
    tracing::trace_span!(
        "qmatmul",
        operation = operation,
        implementation = implementation,
    )
    .entered()
}

pub(super) fn rope(tensor: &'static str) -> EnteredSpan {
    tracing::trace_span!("attn-rope", tensor = tensor).entered()
}

pub(super) fn attention_cache_append() -> EnteredSpan {
    tracing::trace_span!("attn-cache", operation = "append").entered()
}

pub(super) fn attention_cache_access(result_layout: &'static str) -> EnteredSpan {
    tracing::trace_span!(
        "attn-cache",
        operation = "access",
        result_layout = result_layout,
    )
    .entered()
}

pub(super) fn attention_sdpa(implementation: &'static str) -> EnteredSpan {
    tracing::trace_span!("attn-sdpa", implementation = implementation).entered()
}

pub(super) fn attention_repeat_kv(tensor: &'static str) -> EnteredSpan {
    tracing::trace_span!("attn-repeat-kv", tensor = tensor).entered()
}

pub(super) fn attention_query_key(implementation: &'static str) -> EnteredSpan {
    tracing::trace_span!("attn-qk", implementation = implementation).entered()
}

pub(super) fn attention_mask_softmax() -> EnteredSpan {
    tracing::trace_span!("attn-mask-softmax").entered()
}

pub(super) fn attention_paged_value(value_layout: &'static str) -> EnteredSpan {
    tracing::trace_span!("attn-paged-v", value_layout = value_layout).entered()
}

pub(super) fn attention_value(implementation: &'static str) -> EnteredSpan {
    tracing::trace_span!("attn-v", implementation = implementation).entered()
}

pub(super) fn mlp() -> EnteredSpan {
    tracing::trace_span!("mlp").entered()
}

pub(super) fn output() -> EnteredSpan {
    tracing::trace_span!("output").entered()
}
