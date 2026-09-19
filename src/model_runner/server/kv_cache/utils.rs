use super::LayerCache;
use crate::models::ModelInfo;
use anyhow::ensure;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;

/// Token axis in one request's rank-4 [/* batch */ 1, num_kv_heads, token_count, head_dim] cache
/// tensor, commonly used on the tensor returned from `get_page_from_pool`.
pub(super) const TOKEN_DIMENSION: usize = 2;

/// Validates an append and returns the number of newly supplied tokens on the sequence axis.
pub(super) fn validate_cache_append(
    token_count: usize,
    layer_index: usize,
    start_position: usize,
    key: &Tensor,
    value: &Tensor,
) -> Result<usize> {
    ensure!(
        start_position == token_count,
        "KV-cache layer {layer_index} contains {token_count} tokens, but append starts at \
         position {start_position}"
    );
    let key_dimensions = key.dims4()?;
    let value_dimensions = value.dims4()?;
    ensure!(
        key_dimensions == value_dimensions,
        "key and value cache tensors have different dimensions: {key_dimensions:?} and \
         {value_dimensions:?}"
    );
    let appending_token_count = key_dimensions.2;
    ensure!(
        appending_token_count > 0,
        "cannot append an empty KV-cache tensor"
    );
    Ok(appending_token_count)
}

pub(super) fn validate_truncation<T: LayerCache>(
    layer_caches: &[T],
    target_token_count: usize,
) -> Result<()> {
    for (layer_index, layer_cache) in layer_caches.iter().enumerate() {
        let current_token_count = layer_cache.cached_token_count();
        ensure!(
            target_token_count <= current_token_count,
            "cannot truncate KV-cache layer {layer_index} from {current_token_count} to \
             {target_token_count} tokens"
        );
    }
    Ok(())
}

/// Allocates a cache pool of shape:
/// [per_pool_page_count, /* batch */ 1, num_kv_heads, per_page_token_count, head_dim].
pub(super) fn allocate_pool(
    model_info: &ModelInfo,
    device: &Device,
    per_pool_page_count: usize,
    per_page_token_count: usize,
) -> Result<Tensor> {
    Ok(Tensor::zeros(
        (
            per_pool_page_count,
            1,
            model_info.num_kv_heads,
            per_page_token_count,
            model_info.head_dim,
        ),
        model_info.activation_dtype,
        device,
    )?)
}

/// Strips out the page dimension and returns a tensor of shape:
/// [/* batch */ 1, num_kv_heads, per_page_token_count, head_dim].
pub(super) fn get_page_from_pool(pool: &Tensor, page_id: usize) -> Result<Tensor> {
    Ok(pool
        .narrow(/* dim */ 0, /* start */ page_id, /* len */ 1)?
        .squeeze(0)?)
}
