use crate::model_loaders::loaded_model::LoadedModel;
use crate::model_loaders::CachedKeyValue;
use crate::model_loaders::KvCache;
use crate::model_loaders::ModelInfo;
use crate::model_loaders::ModelRole;
use crate::model_runner::KvCacheType;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;

const TOKEN_DIMENSION: usize = 2;

#[derive(Default)]
struct ContiguousLayerCache {
    token_count: usize,
}

trait LayerCache {
    fn cached_token_count(&self) -> usize;
}

impl LayerCache for ContiguousLayerCache {
    fn cached_token_count(&self) -> usize {
        self.token_count
    }
}

/// Stores each layer's key and value tensors in fixed-size, preallocated pools.
pub(super) struct ContiguousKvCache {
    per_layer_token_count: usize,
    key_pool: Tensor,
    value_pool: Tensor,
    layer_caches: Vec<ContiguousLayerCache>,
}

impl ContiguousKvCache {
    fn new(
        model_info: &ModelInfo,
        model_role: ModelRole,
        device: &Device,
        total_size_bytes: usize,
    ) -> Result<Self> {
        let per_pool_size_bytes = total_size_bytes / 2;
        let per_layer_token_count =
            per_pool_size_bytes / model_info.layer_count / model_info.kv_cache_bytes_per_token();
        if per_layer_token_count == 0 {
            bail!(
                "contiguous KV cache size {total_size_bytes} bytes cannot hold one token for each \
                 of {} layers",
                model_info.layer_count
            );
        }
        let cache = Self {
            per_layer_token_count,
            key_pool: allocate_pool(
                model_info,
                device,
                model_info.layer_count,
                per_layer_token_count,
            )?,
            value_pool: allocate_pool(
                model_info,
                device,
                model_info.layer_count,
                per_layer_token_count,
            )?,
            layer_caches: (0..model_info.layer_count)
                .map(|_| ContiguousLayerCache::default())
                .collect(),
        };
        log::info!(
            "Created {model_role} model contiguous KV cache with capacity for \
             {per_layer_token_count} cached tokens"
        );
        Ok(cache)
    }
}

impl KvCache for ContiguousKvCache {
    fn append(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        let Some(layer_cache) = self.layer_caches.get(layer_index) else {
            bail!("invalid KV-cache layer {layer_index}");
        };
        let current_token_count = layer_cache.token_count;
        let appending_token_count =
            validate_cache_append(current_token_count, layer_index, start_position, key, value)?;
        let available_token_count = self.per_layer_token_count - current_token_count;
        if appending_token_count > available_token_count {
            bail!(
                "contiguous KV cache requires {appending_token_count} additional tokens for layer \
                 {layer_index} but only {available_token_count} of {} are available",
                self.per_layer_token_count
            );
        }

        let key_layer = pool_page(&self.key_pool, layer_index)?;
        let value_layer = pool_page(&self.value_pool, layer_index)?;
        key_layer.slice_set(&key.contiguous()?, TOKEN_DIMENSION, current_token_count)?;
        value_layer.slice_set(&value.contiguous()?, TOKEN_DIMENSION, current_token_count)?;
        let cached_token_count = current_token_count + appending_token_count;
        self.layer_caches[layer_index].token_count = cached_token_count;
        Ok(CachedKeyValue {
            key: key_layer.narrow(TOKEN_DIMENSION, 0, cached_token_count)?,
            value: value_layer.narrow(TOKEN_DIMENSION, 0, cached_token_count)?,
        })
    }

    fn truncate(&mut self, target_token_count: usize) -> Result<()> {
        validate_truncation(&self.layer_caches, target_token_count)?;
        for layer_cache in &mut self.layer_caches {
            layer_cache.token_count = target_token_count;
        }
        Ok(())
    }

    fn clear(&mut self) {
        for layer_cache in &mut self.layer_caches {
            *layer_cache = ContiguousLayerCache::default();
        }
    }
}

#[derive(Default)]
struct PagedLayerCache {
    page_ids: Vec<usize>,
    token_count: usize,
}

impl LayerCache for PagedLayerCache {
    fn cached_token_count(&self) -> usize {
        self.token_count
    }
}

/// Stores KV caches in reusable physical pages addressed through per-layer block tables.
///
/// The key and value pools are monolithic tensors whose leading dimension addresses physical
/// pages. Each layer records the page IDs assigned to it, while the cache separately tracks page
/// IDs that were released and can be reassigned. The pools have a fixed physical-page capacity.
pub(super) struct PagedKvCache {
    per_page_token_count: usize,
    per_pool_page_count: usize,
    key_pool: Tensor,
    value_pool: Tensor,
    free_page_ids: Vec<usize>,
    layer_caches: Vec<PagedLayerCache>,
}

impl PagedKvCache {
    fn new(
        model_info: &ModelInfo,
        model_role: ModelRole,
        device: &Device,
        per_page_token_count: usize,
        total_size_bytes: usize,
    ) -> Result<Self> {
        let page_size_bytes = model_info.kv_cache_bytes_per_token() * per_page_token_count;
        let per_pool_size_bytes = total_size_bytes / 2;
        let per_pool_page_count = per_pool_size_bytes / page_size_bytes;
        if per_pool_page_count == 0 {
            bail!(
                "paged KV cache size {total_size_bytes} bytes cannot hold one \
                 {per_page_token_count}-token page per pool"
            );
        }
        let cache = Self {
            per_page_token_count,
            per_pool_page_count,
            key_pool: allocate_pool(
                model_info,
                device,
                per_pool_page_count,
                per_page_token_count,
            )?,
            value_pool: allocate_pool(
                model_info,
                device,
                per_pool_page_count,
                per_page_token_count,
            )?,
            free_page_ids: (0..per_pool_page_count).rev().collect(),
            layer_caches: (0..model_info.layer_count)
                .map(|_| PagedLayerCache::default())
                .collect(),
        };
        let total_cached_token_count =
            per_pool_page_count / model_info.layer_count * per_page_token_count;
        log::info!(
            "Created {model_role} model paged KV cache with {per_pool_page_count} pages per pool \
             and capacity for {total_cached_token_count} cached tokens"
        );
        Ok(cache)
    }

    /// Allocates or reuses a key/value page pair and returns their shared page ID.
    fn allocate_page(&mut self) -> Result<usize> {
        self.free_page_ids
            .pop()
            .context("paged KV cache has no free physical pages")
    }

    /// Returns physical key/value page pairs to the free pool.
    fn release_pages(&mut self, page_ids: impl IntoIterator<Item = usize>) {
        self.free_page_ids.extend(page_ids);
    }

    fn write_page(
        &mut self,
        page_id: usize,
        page_offset: usize,
        key: &Tensor,
        value: &Tensor,
        input_offset: usize,
        token_count: usize,
    ) -> Result<()> {
        let key_slice = key.narrow(TOKEN_DIMENSION, input_offset, token_count)?;
        let value_slice = value.narrow(TOKEN_DIMENSION, input_offset, token_count)?;
        let key_page = pool_page(&self.key_pool, page_id)?;
        let value_page = pool_page(&self.value_pool, page_id)?;
        key_page.slice_set(&key_slice.contiguous()?, TOKEN_DIMENSION, page_offset)?;
        value_page.slice_set(&value_slice.contiguous()?, TOKEN_DIMENSION, page_offset)?;
        Ok(())
    }

    fn validate_append_capacity(
        &self,
        current_token_count: usize,
        appending_token_count: usize,
    ) -> Result<()> {
        let remaining_tokens_in_last_page = match current_token_count % self.per_page_token_count {
            0 => 0,
            page_offset => self.per_page_token_count - page_offset,
        };
        let required_page_count = appending_token_count
            .saturating_sub(remaining_tokens_in_last_page)
            .div_ceil(self.per_page_token_count);
        let available_page_count = self.free_page_ids.len();
        if required_page_count > available_page_count {
            bail!(
                "paged KV cache requires {required_page_count} additional physical pages but only \
                 {available_page_count} of {} are available",
                self.per_pool_page_count
            );
        }
        Ok(())
    }

    fn append_to_pages(
        &mut self,
        layer_index: usize,
        key: &Tensor,
        value: &Tensor,
        appending_token_count: usize,
    ) -> Result<()> {
        let mut input_offset = 0;
        while input_offset < appending_token_count {
            let page_offset =
                self.layer_caches[layer_index].token_count % self.per_page_token_count;
            if page_offset == 0 {
                let page_id = self.allocate_page()?;
                self.layer_caches[layer_index].page_ids.push(page_id);
            }
            let page_id = self.layer_caches[layer_index]
                .page_ids
                .last()
                .copied()
                .context("partial KV-cache page is missing from the block table")?;
            let written_token_count =
                (self.per_page_token_count - page_offset).min(appending_token_count - input_offset);
            self.write_page(
                page_id,
                page_offset,
                key,
                value,
                input_offset,
                written_token_count,
            )?;
            self.layer_caches[layer_index].token_count += written_token_count;
            input_offset += written_token_count;
        }
        Ok(())
    }

    fn reconstruct_full_cache(&self, layer_index: usize) -> Result<CachedKeyValue> {
        let layer_cache = &self.layer_caches[layer_index];
        Ok(CachedKeyValue {
            key: reconstruct_contiguous_tensor(
                &self.key_pool,
                layer_cache,
                self.per_page_token_count,
            )?,
            value: reconstruct_contiguous_tensor(
                &self.value_pool,
                layer_cache,
                self.per_page_token_count,
            )?,
        })
    }
}

impl KvCache for PagedKvCache {
    fn append(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        let Some(layer_cache) = self.layer_caches.get(layer_index) else {
            bail!("invalid KV-cache layer {layer_index}");
        };
        let current_token_count = layer_cache.token_count;
        let appending_token_count =
            validate_cache_append(current_token_count, layer_index, start_position, key, value)?;
        self.validate_append_capacity(current_token_count, appending_token_count)?;
        self.append_to_pages(layer_index, key, value, appending_token_count)?;
        self.reconstruct_full_cache(layer_index)
    }

    fn truncate(&mut self, target_token_count: usize) -> Result<()> {
        validate_truncation(&self.layer_caches, target_token_count)?;
        let retained_page_count = target_token_count.div_ceil(self.per_page_token_count);
        for layer_index in 0..self.layer_caches.len() {
            let released_page_ids = self.layer_caches[layer_index]
                .page_ids
                .split_off(retained_page_count);
            self.layer_caches[layer_index].token_count = target_token_count;
            self.release_pages(released_page_ids);
        }
        Ok(())
    }

    fn clear(&mut self) {
        for layer_index in 0..self.layer_caches.len() {
            let page_ids = std::mem::take(&mut self.layer_caches[layer_index].page_ids);
            self.layer_caches[layer_index].token_count = 0;
            self.release_pages(page_ids);
        }
    }
}

/// Validates an append and returns the number of newly supplied tokens on the sequence axis.
fn validate_cache_append(
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

fn validate_truncation<T: LayerCache>(layer_caches: &[T], target_token_count: usize) -> Result<()> {
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

fn allocate_pool(
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

fn pool_page(pool: &Tensor, page_id: usize) -> Result<Tensor> {
    Ok(pool.narrow(0, page_id, 1)?.squeeze(0)?)
}

/// Reads pages in logical block-table order and returns one contiguous attention tensor.
///
/// Every page except the last is full. The last page is narrowed to its valid token count before
/// concatenation so unwritten page capacity never reaches the attention calculation.
fn reconstruct_contiguous_tensor(
    pool: &Tensor,
    layer_cache: &PagedLayerCache,
    per_page_token_count: usize,
) -> Result<Tensor> {
    let mut remaining_token_count = layer_cache.token_count;
    let mut page_slices = Vec::with_capacity(layer_cache.page_ids.len());
    for &page_id in &layer_cache.page_ids {
        let slice_token_count = per_page_token_count.min(remaining_token_count);
        page_slices.push(pool_page(pool, page_id)?.narrow(
            TOKEN_DIMENSION,
            0,
            slice_token_count,
        )?);
        remaining_token_count -= slice_token_count;
    }
    let materialized = match page_slices.as_slice() {
        [page] => page.clone(),
        [] => bail!("cannot reconstruct an empty KV cache"),
        pages => {
            let page_references: Vec<_> = pages.iter().collect();
            Tensor::cat(&page_references, TOKEN_DIMENSION)?
        }
    };
    Ok(materialized.contiguous()?)
}

pub(super) fn create_kv_cache(
    kv_cache_type: KvCacheType,
    model: &LoadedModel,
    model_role: ModelRole,
    total_size_bytes: usize,
) -> Result<Box<dyn KvCache>> {
    match kv_cache_type {
        KvCacheType::Contiguous => Ok(Box::new(
            ContiguousKvCache::new(model.info(), model_role, model.device(), total_size_bytes)
                .with_context(|| {
                    format!("failed to allocate {model_role} model contiguous KV-cache pools")
                })?,
        )),
        KvCacheType::Paged {
            per_page_token_count,
        } => Ok(Box::new(
            PagedKvCache::new(
                model.info(),
                model_role,
                model.device(),
                per_page_token_count,
                total_size_bytes,
            )
            .with_context(|| {
                format!("failed to allocate {model_role} model paged KV-cache pools")
            })?,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    const PAGE_TOKEN_COUNT: usize = 16;
    const TEST_PAGE_CAPACITY: usize = 4;
    const TEST_CONTIGUOUS_TOKEN_CAPACITY: usize = 8;

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

    fn contiguous_cache(
        layer_count: usize,
        per_layer_token_count: usize,
    ) -> Result<ContiguousKvCache> {
        let model_info = test_model_info(layer_count);
        let total_size_bytes =
            2 * layer_count * per_layer_token_count * model_info.kv_cache_bytes_per_token();
        ContiguousKvCache::new(
            &model_info,
            ModelRole::Target,
            &Device::Cpu,
            total_size_bytes,
        )
    }

    fn paged_cache(
        layer_count: usize,
        per_page_token_count: usize,
        per_pool_page_count: usize,
    ) -> Result<PagedKvCache> {
        let model_info = test_model_info(layer_count);
        let total_size_bytes =
            2 * per_pool_page_count * per_page_token_count * model_info.kv_cache_bytes_per_token();
        PagedKvCache::new(
            &model_info,
            ModelRole::Target,
            &Device::Cpu,
            per_page_token_count,
            total_size_bytes,
        )
    }

    #[test]
    fn preallocates_contiguous_pools_and_rejects_exceeding_their_capacity() -> Result<()> {
        let mut cache =
            contiguous_cache(/* layer_count */ 1, /* per_layer_token_count */ 2)?;
        assert_eq!(cache.key_pool.dim(0)?, 1);
        assert_eq!(cache.value_pool.dim(0)?, 1);
        assert_eq!(cache.key_pool.dim(TOKEN_DIMENSION + 1)?, 2);
        assert_eq!(cache.value_pool.dim(TOKEN_DIMENSION + 1)?, 2);

        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let error = cache
            .append(0, 2, &cache_tensor(2, 1)?, &cache_tensor(102, 1)?)
            .err()
            .expect("append should exceed the contiguous cache capacity")
            .to_string();
        assert!(
            error
                .contains("requires 1 additional tokens for layer 0 but only 0 of 2 are available"),
            "{error}"
        );
        assert_eq!(cache.layer_caches[0].token_count, 2);

        let model_info = test_model_info(/* layer_count */ 1);
        assert!(ContiguousKvCache::new(&model_info, ModelRole::Target, &Device::Cpu, 0).is_err());
        Ok(())
    }

    #[test]
    fn allocates_pages_for_short_full_and_long_prompts() -> Result<()> {
        for (per_page_token_count, token_count, expected_page_count) in [
            (PAGE_TOKEN_COUNT, 15, 1),
            (PAGE_TOKEN_COUNT, 16, 1),
            (PAGE_TOKEN_COUNT, 17, 2),
            (2, 3, 2),
        ] {
            let mut cache = paged_cache(
                /* layer_count */ 1,
                per_page_token_count,
                TEST_PAGE_CAPACITY,
            )?;
            let key = cache_tensor(0, token_count)?;
            let value = cache_tensor(100, token_count)?;
            let cached = cache.append(0, 0, &key, &value)?;
            assert_eq!(cache.layer_caches[0].page_ids.len(), expected_page_count);
            assert_eq!(
                cache.per_pool_page_count - cache.free_page_ids.len(),
                expected_page_count
            );
            assert_eq!(cache.key_pool.dim(0)?, TEST_PAGE_CAPACITY);
            assert_eq!(cache.value_pool.dim(0)?, TEST_PAGE_CAPACITY);
            assert_eq!(
                cache.key_pool.dim(TOKEN_DIMENSION + 1)?,
                per_page_token_count
            );
            assert_eq!(
                cache.value_pool.dim(TOKEN_DIMENSION + 1)?,
                per_page_token_count
            );
            assert_eq!(tensor_values(&cached.key)?, tensor_values(&key)?);
            assert_eq!(tensor_values(&cached.value)?, tensor_values(&value)?);
        }
        Ok(())
    }

    #[test]
    fn contiguous_and_paged_caches_match_across_appends() -> Result<()> {
        let mut contiguous =
            contiguous_cache(/* layer_count */ 1, TEST_CONTIGUOUS_TOKEN_CAPACITY)?;
        let mut paged = paged_cache(
            /* layer_count */ 1,
            /* per_page_token_count */ 2,
            TEST_PAGE_CAPACITY,
        )?;

        for (start, token_count) in [(0, 1), (1, 3), (4, 2)] {
            let key = cache_tensor(start as u32, token_count)?;
            let value = cache_tensor(100 + start as u32, token_count)?;
            let contiguous_cache = contiguous.append(0, start, &key, &value)?;
            let paged_cache = paged.append(0, start, &key, &value)?;
            assert_eq!(
                tensor_values(&contiguous_cache.key)?,
                tensor_values(&paged_cache.key)?
            );
            assert_eq!(
                tensor_values(&contiguous_cache.value)?,
                tensor_values(&paged_cache.value)?
            );
        }
        Ok(())
    }

    #[test]
    fn truncates_contiguous_and_paged_caches_before_reuse() -> Result<()> {
        let mut contiguous =
            contiguous_cache(/* layer_count */ 1, TEST_CONTIGUOUS_TOKEN_CAPACITY)?;
        let mut paged = paged_cache(
            /* layer_count */ 1,
            /* per_page_token_count */ 2,
            TEST_PAGE_CAPACITY,
        )?;
        let key = cache_tensor(0, 6)?;
        let value = cache_tensor(100, 6)?;
        contiguous.append(0, 0, &key, &value)?;
        paged.append(0, 0, &key, &value)?;

        contiguous.truncate(3)?;
        paged.truncate(3)?;
        assert_eq!(paged.free_page_ids.len(), TEST_PAGE_CAPACITY - 2);

        let key = cache_tensor(3, 2)?;
        let value = cache_tensor(103, 2)?;
        let contiguous_cache = contiguous.append(0, 3, &key, &value)?;
        let paged_cache = paged.append(0, 3, &key, &value)?;
        assert_eq!(
            tensor_values(&contiguous_cache.key)?,
            (0..5).collect::<Vec<_>>()
        );
        assert_eq!(
            tensor_values(&contiguous_cache.key)?,
            tensor_values(&paged_cache.key)?
        );
        assert_eq!(
            tensor_values(&contiguous_cache.value)?,
            tensor_values(&paged_cache.value)?
        );
        Ok(())
    }

    #[test]
    fn rejects_truncation_beyond_the_cached_token_count() -> Result<()> {
        let mut contiguous =
            contiguous_cache(/* layer_count */ 1, TEST_CONTIGUOUS_TOKEN_CAPACITY)?;
        let mut paged = paged_cache(
            /* layer_count */ 1,
            /* per_page_token_count */ 2,
            TEST_PAGE_CAPACITY,
        )?;
        let key = cache_tensor(0, 2)?;
        let value = cache_tensor(100, 2)?;
        contiguous.append(0, 0, &key, &value)?;
        paged.append(0, 0, &key, &value)?;

        assert!(contiguous.truncate(3).is_err());
        assert!(paged.truncate(3).is_err());
        Ok(())
    }

    #[test]
    fn appends_across_a_page_boundary() -> Result<()> {
        let mut cache = paged_cache(
            /* layer_count */ 1,
            PAGE_TOKEN_COUNT,
            TEST_PAGE_CAPACITY,
        )?;
        cache.append(0, 0, &cache_tensor(0, 15)?, &cache_tensor(100, 15)?)?;
        let cached = cache.append(0, 15, &cache_tensor(15, 3)?, &cache_tensor(115, 3)?)?;
        assert_eq!(cache.layer_caches[0].page_ids.len(), 2);
        assert_eq!(tensor_values(&cached.key)?, (0..18).collect::<Vec<_>>());
        assert_eq!(
            tensor_values(&cached.value)?,
            (100..118).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn assigns_different_pages_to_different_layers() -> Result<()> {
        let mut cache = paged_cache(
            /* layer_count */ 2,
            PAGE_TOKEN_COUNT,
            TEST_PAGE_CAPACITY,
        )?;
        let key = cache_tensor(0, 1)?;
        let value = cache_tensor(100, 1)?;
        cache.append(0, 0, &key, &value)?;
        cache.append(1, 0, &key, &value)?;
        assert_ne!(
            cache.layer_caches[0].page_ids,
            cache.layer_caches[1].page_ids
        );
        Ok(())
    }

    #[test]
    fn reuses_pages_after_clear() -> Result<()> {
        let mut cache = paged_cache(
            /* layer_count */ 1,
            PAGE_TOKEN_COUNT,
            TEST_PAGE_CAPACITY,
        )?;
        let key = cache_tensor(0, 17)?;
        let value = cache_tensor(100, 17)?;
        cache.append(0, 0, &key, &value)?;
        let free_page_count = cache.free_page_ids.len();
        let mut original_page_ids = cache.layer_caches[0].page_ids.clone();
        cache.clear();
        cache.append(0, 0, &key, &value)?;
        let mut reused_page_ids = cache.layer_caches[0].page_ids.clone();
        original_page_ids.sort_unstable();
        reused_page_ids.sort_unstable();
        assert_eq!(cache.free_page_ids.len(), free_page_count);
        assert_eq!(reused_page_ids, original_page_ids);
        Ok(())
    }

    #[test]
    fn rejects_an_inconsistent_start_position() -> Result<()> {
        let mut cache = paged_cache(
            /* layer_count */ 1,
            PAGE_TOKEN_COUNT,
            TEST_PAGE_CAPACITY,
        )?;
        let key = cache_tensor(0, 1)?;
        let value = cache_tensor(100, 1)?;
        cache.append(0, 0, &key, &value)?;
        assert!(cache.append(0, 0, &key, &value).is_err());
        Ok(())
    }

    #[test]
    fn rejects_appends_that_exceed_the_per_pool_page_count() -> Result<()> {
        let mut cache = paged_cache(
            /* layer_count */ 1, /* per_page_token_count */ 2,
            /* per_pool_page_count */ 1,
        )?;
        let error = cache
            .append(0, 0, &cache_tensor(0, 3)?, &cache_tensor(100, 3)?)
            .err()
            .expect("append should exceed the physical page capacity")
            .to_string();
        assert!(
            error.contains("requires 2 additional physical pages but only 1 of 1 are available"),
            "{error}"
        );
        assert_eq!(cache.free_page_ids.len(), cache.per_pool_page_count);
        assert_eq!(cache.layer_caches[0].token_count, 0);
        Ok(())
    }
}
