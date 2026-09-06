use super::prefix_index::PrefixBlockIndex;
use super::utils::allocate_pool;
use super::utils::pool_page;
use super::utils::validate_cache_append;
use super::utils::validate_truncation;
use super::utils::TOKEN_DIMENSION;
use super::LayerCache;
use crate::model_loaders::CachedKeyValue;
use crate::model_loaders::ModelInfo;
use crate::model_loaders::ModelRole;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;
use thousands::Separable;

#[derive(Default)]
struct PagedLayerCache {
    page_ids: Vec<PageId>,
    token_count: usize,
}

impl LayerCache for PagedLayerCache {
    fn cached_token_count(&self) -> usize {
        self.token_count
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PageId(pub(super) usize);

/// Holds the block tables and token counts used by the sequence being processed.
///
/// "Active" means these tables hold references to physical pages attached to the current
/// sequence; they do not own the underlying key/value tensor storage. Resetting the tables
/// releases only their references, so a future prefix-cache entry can keep shared pages resident.
struct ActiveBlockTables {
    layers: Vec<PagedLayerCache>,
}

impl ActiveBlockTables {
    fn new(layer_count: usize) -> Self {
        Self {
            layers: (0..layer_count)
                .map(|_| PagedLayerCache::default())
                .collect(),
        }
    }
}

/// Owns physical key/value tensors, page reference counts, and reusable page IDs.
///
/// "Physical" means these pages are actual storage locations in the preallocated tensor pools.
/// They exist independently of whichever active sequence or cached prefix refers to them. A page
/// becomes reusable only after all such owners release their references.
struct PhysicalPagePool {
    per_page_token_count: usize,
    page_count: usize,
    key_pool: Tensor,
    value_pool: Tensor,
    reference_counts: Vec<usize>,
    free_page_ids: Vec<PageId>,
}

impl PhysicalPagePool {
    fn new(
        model_info: &ModelInfo,
        device: &Device,
        per_page_token_count: usize,
        total_size_bytes: usize,
    ) -> Result<Self> {
        let page_size_bytes = model_info.kv_cache_bytes_per_token() * per_page_token_count;
        let per_pool_size_bytes = total_size_bytes / 2;
        let per_pool_page_count = per_pool_size_bytes / page_size_bytes;
        if per_pool_page_count == 0 {
            bail!(
                "paged KV cache size {} bytes cannot hold one \
                 {per_page_token_count}-token page per pool",
                total_size_bytes.separate_with_commas()
            );
        }
        Ok(Self {
            per_page_token_count,
            page_count: per_pool_page_count,
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
            reference_counts: vec![0; per_pool_page_count],
            free_page_ids: (0..per_pool_page_count).rev().map(PageId).collect(),
        })
    }

    /// Allocates a key/value page pair for the active block tables.
    ///
    /// The page's reference count is initialized to one for that active owner, so the caller must
    /// not immediately call `retain_allocated_page` for the same ownership.
    fn allocate_page(&mut self) -> Result<PageId> {
        let page_id = self
            .free_page_ids
            .pop()
            .context("paged KV cache has no free physical pages")?;
        self.reference_counts[page_id.0] = 1;
        Ok(page_id)
    }

    /// Adds another owner, such as a cached-prefix entry, to an allocated page.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "prefix-cache entries will retain page references in the next feature"
        )
    )]
    fn retain_allocated_page(&mut self, page_id: PageId) -> Result<()> {
        let reference_count = self
            .reference_counts
            .get_mut(page_id.0)
            .with_context(|| format!("invalid physical page ID {}", page_id.0))?;
        if *reference_count == 0 {
            bail!("cannot retain unallocated physical page {}", page_id.0);
        }
        *reference_count = reference_count
            .checked_add(1)
            .context("physical page reference count overflow")?;
        Ok(())
    }

    /// Releases one owner and makes the page reusable after its final reference is removed.
    fn release_allocated_page(&mut self, page_id: PageId) -> Result<()> {
        let reference_count = self
            .reference_counts
            .get_mut(page_id.0)
            .with_context(|| format!("invalid physical page ID {}", page_id.0))?;
        if *reference_count == 0 {
            bail!("physical page {} was released twice", page_id.0);
        }
        *reference_count -= 1;
        if *reference_count == 0 {
            self.free_page_ids.push(page_id);
        }
        Ok(())
    }

    fn release_allocated_pages(
        &mut self,
        page_ids: impl IntoIterator<Item = PageId>,
    ) -> Result<()> {
        for page_id in page_ids {
            self.release_allocated_page(page_id)?;
        }
        Ok(())
    }

    fn write_page(
        &mut self,
        page_id: PageId,
        page_offset: usize,
        key: &Tensor,
        value: &Tensor,
        input_offset: usize,
        token_count: usize,
    ) -> Result<()> {
        if self.reference_counts[page_id.0] > 1 {
            bail!("cannot mutate shared physical page {}", page_id.0);
        }
        let key_slice = key.narrow(TOKEN_DIMENSION, input_offset, token_count)?;
        let value_slice = value.narrow(TOKEN_DIMENSION, input_offset, token_count)?;
        let key_page = pool_page(&self.key_pool, page_id.0)?;
        let value_page = pool_page(&self.value_pool, page_id.0)?;
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
                self.page_count
            );
        }
        Ok(())
    }
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
        page_slices.push(pool_page(pool, page_id.0)?.narrow(
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

/// Stores KV caches in reusable physical pages addressed through active per-layer block tables.
pub(in crate::model_runner::server) struct PagedKvCache {
    physical_page_pool: PhysicalPagePool,
    active_block_tables: ActiveBlockTables,
    #[expect(
        dead_code,
        reason = "sequence indexing will be connected to request processing separately"
    )]
    prefix_block_index: Option<PrefixBlockIndex>,
}

impl PagedKvCache {
    pub(super) fn new(
        model_info: &ModelInfo,
        model_role: ModelRole,
        device: &Device,
        per_page_token_count: usize,
        enable_prefix_caching: bool,
        total_size_bytes: usize,
    ) -> Result<Self> {
        let physical_page_pool =
            PhysicalPagePool::new(model_info, device, per_page_token_count, total_size_bytes)?;
        let total_cached_token_count =
            physical_page_pool.page_count / model_info.layer_count * per_page_token_count;
        let allocated_size_bytes = 2
            * physical_page_pool.page_count
            * model_info.kv_cache_bytes_per_token()
            * per_page_token_count;
        let prefix_block_index = enable_prefix_caching
            .then(|| PrefixBlockIndex::new(per_page_token_count))
            .transpose()?;
        log::info!(
            "Created {model_role} model paged KV cache with {} pages per pool and capacity for \
             {total_cached_token_count} cached tokens using {} bytes",
            physical_page_pool.page_count,
            allocated_size_bytes.separate_with_commas()
        );
        Ok(Self {
            physical_page_pool,
            active_block_tables: ActiveBlockTables::new(model_info.layer_count),
            prefix_block_index,
        })
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
            let page_offset = self.active_block_tables.layers[layer_index].token_count
                % self.physical_page_pool.per_page_token_count;
            if page_offset == 0 {
                let page_id = self.physical_page_pool.allocate_page()?;
                self.active_block_tables.layers[layer_index]
                    .page_ids
                    .push(page_id);
            }
            let page_id = self.active_block_tables.layers[layer_index]
                .page_ids
                .last()
                .copied()
                .context("partial KV-cache page is missing from the block table")?;
            let written_token_count = (self.physical_page_pool.per_page_token_count - page_offset)
                .min(appending_token_count - input_offset);
            self.physical_page_pool.write_page(
                page_id,
                page_offset,
                key,
                value,
                input_offset,
                written_token_count,
            )?;
            self.active_block_tables.layers[layer_index].token_count += written_token_count;
            input_offset += written_token_count;
        }
        Ok(())
    }

    fn reconstruct_full_cache(&self, layer_index: usize) -> Result<CachedKeyValue> {
        let layer_cache = &self.active_block_tables.layers[layer_index];
        Ok(CachedKeyValue {
            key: reconstruct_contiguous_tensor(
                &self.physical_page_pool.key_pool,
                layer_cache,
                self.physical_page_pool.per_page_token_count,
            )?,
            value: reconstruct_contiguous_tensor(
                &self.physical_page_pool.value_pool,
                layer_cache,
                self.physical_page_pool.per_page_token_count,
            )?,
        })
    }

    pub(super) fn truncate(&mut self, target_token_count: usize) -> Result<()> {
        validate_truncation(&self.active_block_tables.layers, target_token_count)?;
        let retained_page_count =
            target_token_count.div_ceil(self.physical_page_pool.per_page_token_count);
        for layer_cache in &mut self.active_block_tables.layers {
            let released_page_ids = layer_cache.page_ids.split_off(retained_page_count);
            layer_cache.token_count = target_token_count;
            self.physical_page_pool
                .release_allocated_pages(released_page_ids)?;
        }
        Ok(())
    }

    pub(super) fn reset_active_block_tables(&mut self) -> Result<()> {
        for layer_cache in &mut self.active_block_tables.layers {
            let page_ids = std::mem::take(&mut layer_cache.page_ids);
            layer_cache.token_count = 0;
            self.physical_page_pool.release_allocated_pages(page_ids)?;
        }
        Ok(())
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.physical_page_pool.page_count / self.active_block_tables.layers.len()
            * self.physical_page_pool.per_page_token_count
    }

    pub(super) fn append(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        let Some(layer_cache) = self.active_block_tables.layers.get(layer_index) else {
            bail!("invalid KV-cache layer {layer_index}");
        };
        let current_token_count = layer_cache.token_count;
        let appending_token_count =
            validate_cache_append(current_token_count, layer_index, start_position, key, value)?;
        self.physical_page_pool
            .validate_append_capacity(current_token_count, appending_token_count)?;
        self.append_to_pages(layer_index, key, value, appending_token_count)?;
        self.reconstruct_full_cache(layer_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    const PAGE_TOKEN_COUNT: usize = 16;
    const PAGE_CAPACITY: usize = 4;

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
            /* enable_prefix_caching */ false,
            total_size_bytes,
        )
    }

    #[test]
    fn allocates_pages_for_short_full_and_long_prompts() -> Result<()> {
        for (per_page_token_count, token_count, expected_page_count) in [
            (PAGE_TOKEN_COUNT, 15, 1),
            (PAGE_TOKEN_COUNT, 16, 1),
            (PAGE_TOKEN_COUNT, 17, 2),
            (2, 3, 2),
        ] {
            let mut cache = paged_cache(1, per_page_token_count, PAGE_CAPACITY)?;
            let key = cache_tensor(0, token_count)?;
            let value = cache_tensor(100, token_count)?;
            let cached = cache.append(0, 0, &key, &value)?;
            assert_eq!(
                cache.active_block_tables.layers[0].page_ids.len(),
                expected_page_count
            );
            assert_eq!(
                cache.physical_page_pool.page_count - cache.physical_page_pool.free_page_ids.len(),
                expected_page_count
            );
            assert_eq!(cache.physical_page_pool.key_pool.dim(0)?, PAGE_CAPACITY);
            assert_eq!(cache.physical_page_pool.value_pool.dim(0)?, PAGE_CAPACITY);
            assert_eq!(
                cache.physical_page_pool.key_pool.dim(TOKEN_DIMENSION + 1)?,
                per_page_token_count
            );
            assert_eq!(tensor_values(&cached.key)?, tensor_values(&key)?);
            assert_eq!(tensor_values(&cached.value)?, tensor_values(&value)?);
        }
        Ok(())
    }

    #[test]
    fn preserves_values_across_appends() -> Result<()> {
        let mut cache = paged_cache(1, 2, PAGE_CAPACITY)?;

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
        let mut cache = paged_cache(1, 2, PAGE_CAPACITY)?;
        cache.append(0, 0, &cache_tensor(0, 6)?, &cache_tensor(100, 6)?)?;

        cache.truncate(3)?;
        assert_eq!(
            cache.physical_page_pool.free_page_ids.len(),
            PAGE_CAPACITY - 2
        );
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
        let mut cache = paged_cache(1, 2, PAGE_CAPACITY)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;

        assert!(cache.truncate(3).is_err());
        Ok(())
    }

    #[test]
    fn appends_across_a_page_boundary() -> Result<()> {
        let mut cache = paged_cache(1, PAGE_TOKEN_COUNT, PAGE_CAPACITY)?;
        cache.append(0, 0, &cache_tensor(0, 15)?, &cache_tensor(100, 15)?)?;
        let cached = cache.append(0, 15, &cache_tensor(15, 3)?, &cache_tensor(115, 3)?)?;

        assert_eq!(cache.active_block_tables.layers[0].page_ids.len(), 2);
        assert_eq!(tensor_values(&cached.key)?, (0..18).collect::<Vec<_>>());
        assert_eq!(
            tensor_values(&cached.value)?,
            (100..118).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn assigns_different_pages_to_different_layers() -> Result<()> {
        let mut cache = paged_cache(2, PAGE_TOKEN_COUNT, PAGE_CAPACITY)?;
        let key = cache_tensor(0, 1)?;
        let value = cache_tensor(100, 1)?;
        cache.append(0, 0, &key, &value)?;
        cache.append(1, 0, &key, &value)?;

        assert_ne!(
            cache.active_block_tables.layers[0].page_ids,
            cache.active_block_tables.layers[1].page_ids
        );
        Ok(())
    }

    #[test]
    fn reuses_pages_after_resetting_active_block_tables() -> Result<()> {
        let mut cache = paged_cache(1, PAGE_TOKEN_COUNT, PAGE_CAPACITY)?;
        let key = cache_tensor(0, 17)?;
        let value = cache_tensor(100, 17)?;
        cache.append(0, 0, &key, &value)?;
        let free_page_count = cache.physical_page_pool.free_page_ids.len();
        let mut original_page_ids = cache.active_block_tables.layers[0].page_ids.clone();

        cache.reset_active_block_tables()?;
        cache.append(0, 0, &key, &value)?;

        let mut reused_page_ids = cache.active_block_tables.layers[0].page_ids.clone();
        original_page_ids.sort_unstable();
        reused_page_ids.sort_unstable();
        assert_eq!(
            cache.physical_page_pool.free_page_ids.len(),
            free_page_count
        );
        assert_eq!(reused_page_ids, original_page_ids);
        Ok(())
    }

    #[test]
    fn retains_a_cached_page_until_its_prefix_entry_is_evicted() -> Result<()> {
        let mut cache = paged_cache(1, 2, 1)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let page_id = cache.active_block_tables.layers[0].page_ids[0];

        cache.physical_page_pool.retain_allocated_page(page_id)?;
        assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 2);
        cache.reset_active_block_tables()?;
        assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 1);
        assert!(cache.physical_page_pool.free_page_ids.is_empty());

        cache.physical_page_pool.release_allocated_page(page_id)?;
        assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 0);
        assert_eq!(cache.physical_page_pool.free_page_ids, vec![page_id]);
        assert_eq!(cache.physical_page_pool.allocate_page()?, page_id);
        Ok(())
    }

    #[test]
    fn rejects_releasing_a_physical_page_twice() -> Result<()> {
        let mut cache = paged_cache(1, 2, 1)?;
        let page_id = cache.physical_page_pool.allocate_page()?;
        cache.physical_page_pool.release_allocated_page(page_id)?;

        let error = cache
            .physical_page_pool
            .release_allocated_page(page_id)
            .unwrap_err()
            .to_string();
        assert!(error.contains("released twice"), "{error}");
        assert_eq!(cache.physical_page_pool.free_page_ids, vec![page_id]);
        Ok(())
    }

    #[test]
    fn rejects_mutating_a_shared_physical_page() -> Result<()> {
        let mut cache = paged_cache(1, 2, 1)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let page_id = cache.active_block_tables.layers[0].page_ids[0];
        cache.physical_page_pool.retain_allocated_page(page_id)?;
        cache.truncate(1)?;

        let error = cache
            .append(0, 1, &cache_tensor(1, 1)?, &cache_tensor(101, 1)?)
            .err()
            .context("append should not mutate a shared physical page")?
            .to_string();
        assert!(
            error.contains("cannot mutate shared physical page"),
            "{error}"
        );
        Ok(())
    }

    #[test]
    fn rejects_an_inconsistent_start_position() -> Result<()> {
        let mut cache = paged_cache(1, PAGE_TOKEN_COUNT, PAGE_CAPACITY)?;
        let key = cache_tensor(0, 1)?;
        let value = cache_tensor(100, 1)?;
        cache.append(0, 0, &key, &value)?;

        assert!(cache.append(0, 0, &key, &value).is_err());
        Ok(())
    }

    #[test]
    fn rejects_appends_that_exceed_page_capacity() -> Result<()> {
        let mut cache = paged_cache(1, 2, 1)?;
        let error = cache
            .append(0, 0, &cache_tensor(0, 3)?, &cache_tensor(100, 3)?)
            .err()
            .context("append should exceed physical page capacity")?
            .to_string();

        assert!(
            error.contains("requires 2 additional physical pages but only 1 of 1 are available"),
            "{error}"
        );
        assert_eq!(
            cache.physical_page_pool.free_page_ids.len(),
            cache.physical_page_pool.page_count
        );
        assert_eq!(cache.active_block_tables.layers[0].token_count, 0);
        Ok(())
    }
}
