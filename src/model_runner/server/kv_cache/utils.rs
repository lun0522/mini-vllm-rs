use super::LayerCache;
use crate::model_loaders::ModelInfo;
use anyhow::bail;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;

pub(super) const TOKEN_DIMENSION: usize = 2;

/// Validates an append and returns the number of newly supplied tokens on the sequence axis.
pub(super) fn validate_cache_append(
    token_count: usize,
    layer_index: usize,
    start_position: usize,
    key: &Tensor,
    value: &Tensor,
) -> Result<usize> {
    if start_position != token_count {
        bail!(
            "KV-cache layer {layer_index} contains {token_count} tokens, but append starts at \
             position {start_position}"
        );
    }
    let key_dimensions = key.dims4()?;
    let value_dimensions = value.dims4()?;
    if key_dimensions != value_dimensions {
        bail!(
            "key and value cache tensors have different dimensions: {key_dimensions:?} and \
             {value_dimensions:?}"
        );
    }
    let appending_token_count = key_dimensions.2;
    if appending_token_count == 0 {
        bail!("cannot append an empty KV-cache tensor");
    }
    Ok(appending_token_count)
}

pub(super) fn validate_truncation<T: LayerCache>(
    layer_caches: &[T],
    target_token_count: usize,
) -> Result<()> {
    for (layer_index, layer_cache) in layer_caches.iter().enumerate() {
        let current_token_count = layer_cache.cached_token_count();
        if target_token_count > current_token_count {
            bail!(
                "cannot truncate KV-cache layer {layer_index} from {current_token_count} to \
                 {target_token_count} tokens"
            );
        }
    }
    Ok(())
}

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
            model_info.kv_head_count,
            per_page_token_count,
            model_info.head_dimension,
        ),
        model_info.activation_dtype,
        device,
    )?)
}

pub(super) fn pool_page(pool: &Tensor, page_id: usize) -> Result<Tensor> {
    Ok(pool.narrow(0, page_id, 1)?.squeeze(0)?)
}
