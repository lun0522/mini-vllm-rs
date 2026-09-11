use super::utils::allocate_pool;
use super::utils::pool_page;
use super::utils::validate_cache_append;
use super::utils::validate_truncation;
use super::utils::TOKEN_DIMENSION;
use super::LayerCache;
use crate::models::CachedKeyValue;
use crate::models::ModelInfo;
use crate::models::ModelRole;
use anyhow::bail;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;
use thousands::Separable;

#[derive(Default)]
struct ContiguousLayerCache {
    token_count: usize,
}

impl LayerCache for ContiguousLayerCache {
    fn cached_token_count(&self) -> usize {
        self.token_count
    }
}

/// Stores each layer's key and value tensors in fixed-size, preallocated pools.
pub(in crate::model_runner::server) struct ContiguousKvCache {
    per_layer_token_capacity: usize,
    key_pool: Tensor,
    value_pool: Tensor,
}

/// Tracks how many tokens one request has written to each contiguous layer cache.
///
/// The backend has only one physical storage region, so only one of these states may be active at
/// a time. Keeping progress here still separates request lifetime from shared tensor ownership.
pub(super) struct RequestContiguousCacheState {
    layer_caches: Vec<ContiguousLayerCache>,
}

impl RequestContiguousCacheState {
    pub(super) fn new(layer_count: usize) -> Self {
        Self {
            layer_caches: (0..layer_count)
                .map(|_| ContiguousLayerCache::default())
                .collect(),
        }
    }
}

impl ContiguousKvCache {
    pub(super) fn new(
        model_info: &ModelInfo,
        model_role: ModelRole,
        device: &Device,
        total_size_bytes: usize,
    ) -> Result<Self> {
        let per_pool_size_bytes = total_size_bytes / 2;
        let per_layer_token_capacity =
            per_pool_size_bytes / model_info.layer_count / model_info.kv_cache_bytes_per_token();
        if per_layer_token_capacity == 0 {
            bail!(
                "contiguous KV cache size {} bytes cannot hold one token for each of {} layers",
                total_size_bytes.separate_with_commas(),
                model_info.layer_count
            );
        }
        let cache = Self {
            per_layer_token_capacity,
            key_pool: allocate_pool(
                model_info,
                device,
                model_info.layer_count,
                per_layer_token_capacity,
            )?,
            value_pool: allocate_pool(
                model_info,
                device,
                model_info.layer_count,
                per_layer_token_capacity,
            )?,
        };
        let allocated_size_bytes = 2
            * model_info.layer_count
            * per_layer_token_capacity
            * model_info.kv_cache_bytes_per_token();
        log::info!(
            "Created {model_role} model contiguous KV cache with capacity for \
             {per_layer_token_capacity} cached tokens using {} bytes",
            allocated_size_bytes.separate_with_commas()
        );
        Ok(cache)
    }

    pub(super) fn truncate(
        &mut self,
        request_state: &mut RequestContiguousCacheState,
        target_token_count: usize,
    ) -> Result<()> {
        validate_truncation(&request_state.layer_caches, target_token_count)?;
        for layer_cache in &mut request_state.layer_caches {
            layer_cache.token_count = target_token_count;
        }
        Ok(())
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.per_layer_token_capacity
    }

    pub(super) fn layer_count(&self) -> usize {
        self.key_pool.dims()[0]
    }

    pub(super) fn append(
        &mut self,
        request_state: &mut RequestContiguousCacheState,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        let Some(layer_cache) = request_state.layer_caches.get(layer_index) else {
            bail!("invalid KV-cache layer {layer_index}");
        };
        let current_token_count = layer_cache.token_count;
        let appending_token_count =
            validate_cache_append(current_token_count, layer_index, start_position, key, value)?;
        let available_token_count = self.per_layer_token_capacity - current_token_count;
        if appending_token_count > available_token_count {
            bail!(
                "contiguous KV cache requires {appending_token_count} additional tokens for layer \
                 {layer_index} but only {available_token_count} of {} are available",
                self.per_layer_token_capacity
            );
        }

        let key_layer = pool_page(&self.key_pool, layer_index)?;
        let value_layer = pool_page(&self.value_pool, layer_index)?;
        key_layer.slice_set(&key.contiguous()?, TOKEN_DIMENSION, current_token_count)?;
        value_layer.slice_set(&value.contiguous()?, TOKEN_DIMENSION, current_token_count)?;
        let cached_token_count = current_token_count + appending_token_count;
        request_state.layer_caches[layer_index].token_count = cached_token_count;
        Ok(CachedKeyValue {
            key: key_layer.narrow(TOKEN_DIMENSION, 0, cached_token_count)?,
            value: value_layer.narrow(TOKEN_DIMENSION, 0, cached_token_count)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use candle_core::DType;

    const TOKEN_CAPACITY: usize = 8;

    fn cache_tensor(start: u32, token_count: usize) -> Result<Tensor> {
        Ok(
            Tensor::arange(start, start + token_count as u32, &Device::Cpu)?.reshape((
                1,
                1,
                token_count,
                1,
            ))?,
        )
    }

    fn tensor_values(tensor: &Tensor) -> Result<Vec<u32>> {
        Ok(tensor.flatten_all()?.to_vec1()?)
    }

    fn test_model_info(layer_count: usize) -> ModelInfo {
        ModelInfo {
            layer_count,
            kv_head_count: 1,
            head_dimension: 1,
            activation_dtype: DType::U32,
        }
    }

    struct TestContiguousKvCache {
        cache: ContiguousKvCache,
        request_state: RequestContiguousCacheState,
    }

    impl TestContiguousKvCache {
        fn append(
            &mut self,
            layer_index: usize,
            start_position: usize,
            key: &Tensor,
            value: &Tensor,
        ) -> Result<CachedKeyValue> {
            self.cache.append(
                &mut self.request_state,
                layer_index,
                start_position,
                key,
                value,
            )
        }

        fn truncate(&mut self, target_token_count: usize) -> Result<()> {
            self.cache
                .truncate(&mut self.request_state, target_token_count)
        }
    }

    fn contiguous_cache(
        layer_count: usize,
        token_capacity: usize,
    ) -> Result<TestContiguousKvCache> {
        let model_info = test_model_info(layer_count);
        let total_size_bytes =
            2 * layer_count * token_capacity * model_info.kv_cache_bytes_per_token();
        Ok(TestContiguousKvCache {
            cache: ContiguousKvCache::new(
                &model_info,
                ModelRole::Target,
                &Device::Cpu,
                total_size_bytes,
            )?,
            request_state: RequestContiguousCacheState::new(layer_count),
        })
    }

    #[test]
    fn preallocates_pools_and_rejects_exceeding_capacity() -> Result<()> {
        let mut cache = contiguous_cache(/* layer_count */ 1, /* token_capacity */ 2)?;
        assert_eq!(cache.cache.key_pool.dim(0)?, 1);
        assert_eq!(cache.cache.value_pool.dim(0)?, 1);
        assert_eq!(cache.cache.key_pool.dim(TOKEN_DIMENSION + 1)?, 2);
        assert_eq!(cache.cache.value_pool.dim(TOKEN_DIMENSION + 1)?, 2);

        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let error = cache
            .append(0, 2, &cache_tensor(2, 1)?, &cache_tensor(102, 1)?)
            .err()
            .context("append should exceed contiguous cache capacity")?
            .to_string();
        assert!(
            error
                .contains("requires 1 additional tokens for layer 0 but only 0 of 2 are available"),
            "{error}"
        );
        assert_eq!(cache.request_state.layer_caches[0].token_count, 2);

        let model_info = test_model_info(/* layer_count */ 1);
        assert!(ContiguousKvCache::new(&model_info, ModelRole::Target, &Device::Cpu, 0).is_err());
        Ok(())
    }

    #[test]
    fn preserves_values_across_appends() -> Result<()> {
        let mut cache = contiguous_cache(/* layer_count */ 1, TOKEN_CAPACITY)?;

        let mut cached = None;
        for (start, token_count) in [(0, 1), (1, 3), (4, 2)] {
            cached = Some(cache.append(
                0,
                start,
                &cache_tensor(start as u32, token_count)?,
                &cache_tensor(100 + start as u32, token_count)?,
            )?);
        }
        let cached = cached.context("test should append cache values")?;
        assert_eq!(tensor_values(&cached.key)?, (0..6).collect::<Vec<_>>());
        assert_eq!(
            tensor_values(&cached.value)?,
            (100..106).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn truncates_before_reuse() -> Result<()> {
        let mut cache = contiguous_cache(/* layer_count */ 1, TOKEN_CAPACITY)?;
        cache.append(0, 0, &cache_tensor(0, 6)?, &cache_tensor(100, 6)?)?;

        cache.truncate(3)?;
        let cached = cache.append(0, 3, &cache_tensor(3, 2)?, &cache_tensor(103, 2)?)?;

        assert_eq!(tensor_values(&cached.key)?, (0..5).collect::<Vec<_>>());
        assert_eq!(
            tensor_values(&cached.value)?,
            (100..105).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn rejects_truncation_beyond_cached_token_count() -> Result<()> {
        let mut cache = contiguous_cache(/* layer_count */ 1, TOKEN_CAPACITY)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;

        assert!(cache.truncate(3).is_err());
        Ok(())
    }
}
