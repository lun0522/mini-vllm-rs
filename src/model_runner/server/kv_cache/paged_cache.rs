use super::physical_page_pool::PhysicalPagePool;
use super::prefix_index::PrefixBlockIndex;
use super::prefix_index::PrefixCacheAffinitySnapshot;
use super::request_state::LayerCacheView;
use super::request_state::RequestPagedCacheState;
use super::utils::get_page_from_pool;
use super::utils::validate_cache_append;
use super::utils::TOKEN_DIMENSION;
use super::PrefixCacheTelemetry;
use crate::models::ContiguousCacheTensors;
use crate::models::ModelInfo;
use crate::models::ModelRole;
use crate::models::PagedCacheLayout;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;
use log::info;
use thousands::Separable;

use crate::model_runner::server::BackendId;

/// Coordinates shared physical pages and prefix metadata with request-specific virtual tables.
pub(in crate::model_runner::server) struct PagedKvCache {
    physical_page_pool: PhysicalPagePool,
    prefix_block_index: Option<PrefixBlockIndex>,
    token_capacity: usize,
    layer_count: usize,
    prefix_cache_counters: PrefixCacheCounters,
}

#[derive(Default)]
struct PrefixCacheCounters {
    cumulative_indexed_token_count: usize,
    cumulative_evicted_token_count: usize,
    cumulative_restored_token_count: usize,
    cumulative_lookup_count: usize,
    cumulative_hit_count: usize,
}

impl PrefixCacheCounters {
    fn add_indexed_token_count(&mut self, token_count: usize) {
        self.cumulative_indexed_token_count = self
            .cumulative_indexed_token_count
            .saturating_add(token_count);
    }

    fn add_evicted_token_count(&mut self, token_count: usize) {
        self.cumulative_evicted_token_count = self
            .cumulative_evicted_token_count
            .saturating_add(token_count);
    }

    fn record_hit(&mut self, restored_token_count: usize) {
        self.cumulative_restored_token_count = self
            .cumulative_restored_token_count
            .saturating_add(restored_token_count);
        self.cumulative_hit_count = self.cumulative_hit_count.saturating_add(1);
    }

    fn record_lookup(&mut self) {
        self.cumulative_lookup_count = self.cumulative_lookup_count.saturating_add(1);
    }
}

impl PagedKvCache {
    pub(super) fn new(
        backend_id: BackendId,
        model_info: &ModelInfo,
        model_role: ModelRole,
        device: &Device,
        per_page_token_count: usize,
        enable_prefix_caching: bool,
        total_size_bytes: usize,
    ) -> Result<Self> {
        let physical_page_pool =
            PhysicalPagePool::new(model_info, device, per_page_token_count, total_size_bytes)?;
        let token_capacity =
            physical_page_pool.page_count / model_info.layer_count * per_page_token_count;
        let allocated_size_bytes = 2
            * physical_page_pool.page_count
            * model_info.kv_cache_bytes_per_token()
            * per_page_token_count;
        let prefix_block_index = enable_prefix_caching
            .then(|| PrefixBlockIndex::new(per_page_token_count))
            .transpose()?;
        info!(
            "[Backend {backend_id}] Created {model_role} model paged KV cache with {} pages per \
             pool and capacity for {token_capacity} cached tokens using {} bytes",
            physical_page_pool.page_count,
            allocated_size_bytes.separate_with_commas()
        );
        Ok(Self {
            physical_page_pool,
            prefix_block_index,
            token_capacity,
            layer_count: model_info.layer_count,
            prefix_cache_counters: PrefixCacheCounters::default(),
        })
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.token_capacity
    }

    pub(super) fn per_page_token_count(&self) -> usize {
        self.physical_page_pool.per_page_token_count
    }

    pub(super) fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub(super) fn prefix_cache_telemetry(&self) -> Option<PrefixCacheTelemetry> {
        let prefix_block_index = self.prefix_block_index.as_ref()?;
        let current_indexed_token_count = prefix_block_index
            .block_count()
            .checked_mul(self.physical_page_pool.per_page_token_count)
            .expect("indexed prefix token count must fit within cache capacity");
        Some(PrefixCacheTelemetry {
            token_capacity: self.token_capacity,
            current_indexed_token_count,
            cumulative_indexed_token_count: self
                .prefix_cache_counters
                .cumulative_indexed_token_count,
            cumulative_evicted_token_count: self
                .prefix_cache_counters
                .cumulative_evicted_token_count,
            cumulative_restored_token_count: self
                .prefix_cache_counters
                .cumulative_restored_token_count,
            cumulative_lookup_count: self.prefix_cache_counters.cumulative_lookup_count,
            cumulative_hit_count: self.prefix_cache_counters.cumulative_hit_count,
        })
    }

    pub(super) fn create_cache_affinity_snapshot(&self) -> Option<PrefixCacheAffinitySnapshot> {
        self.prefix_block_index
            .as_ref()
            .map(PrefixBlockIndex::create_cache_affinity_snapshot)
    }

    /// Attaches the longest reusable prefix to the current request and returns its token count.
    pub(super) fn restore_cached_prefix(
        &mut self,
        request_state: &mut RequestPagedCacheState,
        input_token_ids: &[u32],
    ) -> Result<usize> {
        let Some(prefix_block_index) = self.prefix_block_index.as_ref() else {
            return Ok(0);
        };
        ensure!(
            !request_state.is_block_table_populated(),
            "cannot attach a cached prefix after the request block table has been populated"
        );
        self.prefix_cache_counters.record_lookup();
        let prefix_match = prefix_block_index.find_longest_cached_prefix(input_token_ids)?;
        self.physical_page_pool
            .retain_allocated_pages_or_rollback(prefix_match.page_ids_by_layer.iter().flatten())?;
        request_state.attach_cached_prefix(prefix_match);
        let cached_token_count = request_state.cached_token_count();
        if cached_token_count > 0 {
            self.prefix_cache_counters.record_hit(cached_token_count);
        }
        Ok(cached_token_count)
    }

    pub(super) fn append_new_key_value(
        &mut self,
        request_state: &mut RequestPagedCacheState,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<()> {
        let layer_cache = request_state
            .layer_cache(layer_index)
            .context(format!("invalid KV-cache layer {layer_index}"))?;
        let appending_token_count = validate_cache_append(
            layer_cache.cached_token_count,
            layer_index,
            start_position,
            key,
            value,
        )?;
        self.ensure_append_capacity_with_prefix_eviction(
            layer_cache.cached_token_count,
            appending_token_count,
        )?;
        self.append_to_pages(
            request_state,
            layer_index,
            key,
            value,
            appending_token_count,
        )
    }

    pub(super) fn get_contiguous_cache_tensors(
        &self,
        request_state: &RequestPagedCacheState,
        layer_index: usize,
    ) -> Result<ContiguousCacheTensors> {
        let layer_cache = request_state
            .layer_cache(layer_index)
            .context(format!("invalid KV-cache layer {layer_index}"))?;
        Ok(ContiguousCacheTensors {
            key: self
                .reconstruct_contiguous_tensor(&self.physical_page_pool.key_pool, layer_cache)?,
            value: self
                .reconstruct_contiguous_tensor(&self.physical_page_pool.value_pool, layer_cache)?,
        })
    }

    pub(super) fn get_paged_cache_layout(
        &self,
        request_state: &RequestPagedCacheState,
        layer_index: usize,
    ) -> Result<PagedCacheLayout> {
        let layer_cache = request_state
            .layer_cache(layer_index)
            .context(format!("invalid KV-cache layer {layer_index}"))?;
        Ok(PagedCacheLayout {
            key_pool: self.physical_page_pool.key_pool.clone(),
            value_pool: self.physical_page_pool.value_pool.clone(),
            page_ids: layer_cache.page_ids.iter().map(|&id| id.0).collect(),
            per_page_token_count: self.physical_page_pool.per_page_token_count,
            cached_token_count: layer_cache.cached_token_count,
        })
    }

    pub(super) fn truncate(
        &mut self,
        request_state: &mut RequestPagedCacheState,
        target_token_count: usize,
    ) -> Result<()> {
        let released_page_ids = request_state.truncate(target_token_count)?;
        self.physical_page_pool
            .release_allocated_pages(released_page_ids)
    }

    /// Publishes complete prefix blocks not previously published by this request.
    pub(super) fn publish_completed_prefix(
        &mut self,
        request_state: &mut RequestPagedCacheState,
        token_ids: &[u32],
    ) -> Result<()> {
        let Some(prefix_block_index) = self.prefix_block_index.as_mut() else {
            return Ok(());
        };
        let cached_token_count = request_state.cached_token_count();
        let cached_token_ids = token_ids
            .get(..cached_token_count)
            .context("active KV cache contains more tokens than the completed request")?;
        let indexing_result = prefix_block_index.index_cached_sequence(
            request_state.prefix_block_cursor(),
            cached_token_ids,
            request_state.page_ids_by_layer(),
        )?;
        self.physical_page_pool
            .retain_allocated_pages_or_rollback(indexing_result.page_ids.iter())?;
        let newly_indexed_token_count = request_state.record_prefix_publication(indexing_result)?;
        self.prefix_cache_counters
            .add_indexed_token_count(newly_indexed_token_count);
        Ok(())
    }

    pub(super) fn reset_request(
        &mut self,
        request_state: &mut RequestPagedCacheState,
    ) -> Result<()> {
        let released_page_ids = request_state.reset();
        self.physical_page_pool
            .release_allocated_pages(released_page_ids)
    }

    fn append_to_pages(
        &mut self,
        request_state: &mut RequestPagedCacheState,
        layer_index: usize,
        key: &Tensor,
        value: &Tensor,
        appending_token_count: usize,
    ) -> Result<()> {
        let mut input_offset = 0;
        while input_offset < appending_token_count {
            let page_offset = request_state
                .layer_cache(layer_index)
                .context("KV-cache layer is missing from the active block tables")?
                .cached_token_count
                % self.physical_page_pool.per_page_token_count;
            if page_offset == 0 {
                let page_id = self.physical_page_pool.allocate_page()?;
                request_state.append_page(layer_index, page_id);
            }
            let page_id = request_state
                .layer_cache(layer_index)
                .and_then(|layer_cache| layer_cache.page_ids.last().copied())
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
            request_state.record_appended_tokens(layer_index, written_token_count)?;
            input_offset += written_token_count;
        }
        Ok(())
    }

    /// Evicts inactive cached-prefix leaves until the requested append fits or none remain.
    fn ensure_append_capacity_with_prefix_eviction(
        &mut self,
        current_token_count: usize,
        appending_token_count: usize,
    ) -> Result<()> {
        let required_page_count = self
            .physical_page_pool
            .compute_required_page_count(current_token_count, appending_token_count);
        while self.physical_page_pool.free_page_count() < required_page_count {
            let physical_page_pool = &self.physical_page_pool;
            let Some(prefix_block_index) = self.prefix_block_index.as_mut() else {
                break;
            };
            let Some(evicted_block) =
                prefix_block_index.evict_least_recently_used_leaf(|page_ids| {
                    physical_page_pool.are_pages_inactive(page_ids)
                })?
            else {
                break;
            };
            self.physical_page_pool
                .release_allocated_pages(evicted_block.page_ids_by_layer.iter().copied())?;
            self.prefix_cache_counters
                .add_evicted_token_count(self.physical_page_pool.per_page_token_count);
        }
        self.physical_page_pool
            .validate_append_capacity(current_token_count, appending_token_count)
    }

    /// Reads pages in logical block-table order and returns one contiguous attention tensor.
    ///
    /// Every page except the last is full. The last page is narrowed to its valid token count before
    /// concatenation so unwritten page capacity never reaches the attention calculation.
    fn reconstruct_contiguous_tensor(
        &self,
        pool: &Tensor,
        layer_cache: LayerCacheView<'_>,
    ) -> Result<Tensor> {
        let mut remaining_token_count = layer_cache.cached_token_count;
        let mut page_slices = Vec::with_capacity(layer_cache.page_ids.len());
        for &page_id in layer_cache.page_ids {
            let slice_token_count = self
                .physical_page_pool
                .per_page_token_count
                .min(remaining_token_count);
            page_slices.push(get_page_from_pool(pool, page_id.0)?.narrow(
                TOKEN_DIMENSION,
                /* start */ 0,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_runner::server::kv_cache::physical_page_pool::PageId;
    use crate::model_runner::server::kv_cache::physical_page_pool::PageState;
    use crate::model_runner::server::kv_cache::prefix_index::PrefixBlockCursor;
    use candle_core::DType;

    const PAGE_TOKEN_COUNT: usize = 16;
    const PAGE_CAPACITY: usize = 4;

    fn page_id_slices(
        page_ids_by_layer: &[Vec<PageId>],
    ) -> impl Clone + Iterator<Item = &[PageId]> {
        page_ids_by_layer.iter().map(Vec::as_slice)
    }

    struct TestPagedKvCache {
        cache: PagedKvCache,
        request_state: RequestPagedCacheState,
    }

    impl TestPagedKvCache {
        fn page_ids(&self, layer_index: usize) -> &[PageId] {
            self.request_state
                .layer_cache(layer_index)
                .expect("test layer should exist")
                .page_ids
        }

        fn cached_token_count(&self, layer_index: usize) -> usize {
            self.request_state
                .layer_cache(layer_index)
                .expect("test layer should exist")
                .cached_token_count
        }

        fn append(
            &mut self,
            layer_index: usize,
            start_position: usize,
            key: &Tensor,
            value: &Tensor,
        ) -> Result<ContiguousCacheTensors> {
            self.cache.append_new_key_value(
                &mut self.request_state,
                layer_index,
                start_position,
                key,
                value,
            )?;
            self.cache
                .get_contiguous_cache_tensors(&self.request_state, layer_index)
        }

        fn restore_cached_prefix(&mut self, token_ids: &[u32]) -> Result<usize> {
            self.cache
                .restore_cached_prefix(&mut self.request_state, token_ids)
        }

        fn truncate(&mut self, target_token_count: usize) -> Result<()> {
            self.cache
                .truncate(&mut self.request_state, target_token_count)
        }

        fn reset_active_block_tables(&mut self) -> Result<()> {
            self.cache.reset_request(&mut self.request_state)
        }
    }

    impl std::ops::Deref for TestPagedKvCache {
        type Target = PagedKvCache;

        fn deref(&self) -> &Self::Target {
            &self.cache
        }
    }

    impl std::ops::DerefMut for TestPagedKvCache {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.cache
        }
    }

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
            num_kv_heads: 1,
            head_dim: 1,
            activation_dtype: DType::U32,
            context_length: 1,
        }
    }

    fn paged_cache(
        layer_count: usize,
        per_page_token_count: usize,
        per_pool_page_count: usize,
    ) -> Result<TestPagedKvCache> {
        paged_cache_with_prefix_caching(
            layer_count,
            per_page_token_count,
            per_pool_page_count,
            false,
        )
    }

    fn paged_cache_with_prefix_caching(
        layer_count: usize,
        per_page_token_count: usize,
        per_pool_page_count: usize,
        enable_prefix_caching: bool,
    ) -> Result<TestPagedKvCache> {
        let model_info = test_model_info(layer_count);
        let total_size_bytes =
            2 * per_pool_page_count * per_page_token_count * model_info.kv_cache_bytes_per_token();
        Ok(TestPagedKvCache {
            cache: PagedKvCache::new(
                BackendId::from_index(0),
                &model_info,
                ModelRole::Target,
                &Device::Cpu,
                per_page_token_count,
                enable_prefix_caching,
                total_size_bytes,
            )?,
            request_state: RequestPagedCacheState::new(layer_count, per_page_token_count),
        })
    }

    fn retain_active_pages_for_prefix(cache: &mut TestPagedKvCache) -> Result<Vec<Vec<PageId>>> {
        let cached_page_ids_by_layer: Vec<Vec<_>> = cache
            .request_state
            .page_ids_by_layer()
            .map(<[PageId]>::to_vec)
            .collect();
        for &page_id in cached_page_ids_by_layer.iter().flatten() {
            cache.physical_page_pool.retain_allocated_page(page_id)?;
        }
        Ok(cached_page_ids_by_layer)
    }

    fn finish_request(cache: &mut TestPagedKvCache, token_ids: &[u32]) -> Result<usize> {
        cache
            .cache
            .publish_completed_prefix(&mut cache.request_state, token_ids)?;
        let newly_indexed_token_count = cache.request_state.accumulated_indexed_token_count();
        cache.reset_active_block_tables()?;
        Ok(newly_indexed_token_count)
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
            assert_eq!(cache.page_ids(0).len(), expected_page_count);
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

        assert_eq!(cache.page_ids(0).len(), 2);
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

        assert_ne!(cache.page_ids(0), cache.page_ids(1));
        Ok(())
    }

    #[test]
    fn reuses_pages_after_resetting_active_block_tables() -> Result<()> {
        let mut cache = paged_cache(1, PAGE_TOKEN_COUNT, PAGE_CAPACITY)?;
        let key = cache_tensor(0, 17)?;
        let value = cache_tensor(100, 17)?;
        cache.append(0, 0, &key, &value)?;
        let free_page_count = cache.physical_page_pool.free_page_ids.len();
        let mut original_page_ids = cache.page_ids(0).to_vec();

        cache.reset_active_block_tables()?;
        cache.append(0, 0, &key, &value)?;

        let mut reused_page_ids = cache.page_ids(0).to_vec();
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
    fn attaches_cached_prefix_pages_to_empty_active_block_tables() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(2, 2, 6, true)?;
        for layer_index in 0..2 {
            cache.append(layer_index, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        }
        let cached_page_ids_by_layer = retain_active_pages_for_prefix(&mut cache)?;
        cache
            .prefix_block_index
            .as_mut()
            .context("prefix caching should be enabled")?
            .index_cached_sequence(
                &PrefixBlockCursor::at_root(),
                &[1, 2, 3, 4],
                page_id_slices(&cached_page_ids_by_layer),
            )?;
        cache.reset_active_block_tables()?;

        assert_eq!(cache.restore_cached_prefix(&[1, 2, 3, 4, 5])?, 4);
        for (layer_index, expected_page_ids) in cached_page_ids_by_layer.iter().enumerate() {
            assert_eq!(&cache.page_ids(layer_index), expected_page_ids);
            assert_eq!(cache.cached_token_count(layer_index), 4);
            for page_id in expected_page_ids {
                assert_eq!(
                    cache.physical_page_pool.page_states[page_id.0],
                    PageState::WrittenInUse { reader_count: 1 }
                );
            }
        }

        cache.reset_active_block_tables()?;
        for page_id in cached_page_ids_by_layer.iter().flatten() {
            assert_eq!(
                cache.physical_page_pool.page_states[page_id.0],
                PageState::WrittenNotInUse
            );
        }
        Ok(())
    }

    #[test]
    fn attaches_only_the_complete_matching_prefix_blocks() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 4, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let cached_page_ids_by_layer = retain_active_pages_for_prefix(&mut cache)?;
        cache
            .prefix_block_index
            .as_mut()
            .context("prefix caching should be enabled")?
            .index_cached_sequence(
                &PrefixBlockCursor::at_root(),
                &[1, 2, 3, 4],
                page_id_slices(&cached_page_ids_by_layer),
            )?;
        cache.reset_active_block_tables()?;

        assert_eq!(cache.restore_cached_prefix(&[1, 2, 8, 9])?, 2);
        assert_eq!(cache.page_ids(0), &cached_page_ids_by_layer[0][..1]);
        assert_eq!(cache.cached_token_count(0), 2);
        assert_eq!(
            cache.physical_page_pool.page_states[cached_page_ids_by_layer[0][0].0],
            PageState::WrittenInUse { reader_count: 1 }
        );
        assert_eq!(
            cache.physical_page_pool.page_states[cached_page_ids_by_layer[0][1].0],
            PageState::WrittenNotInUse
        );
        Ok(())
    }

    #[test]
    fn rejects_attaching_a_prefix_after_block_table_population() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 2, true)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let cached_page_ids_by_layer = retain_active_pages_for_prefix(&mut cache)?;
        cache
            .prefix_block_index
            .as_mut()
            .context("prefix caching should be enabled")?
            .index_cached_sequence(
                &PrefixBlockCursor::at_root(),
                &[1, 2],
                page_id_slices(&cached_page_ids_by_layer),
            )?;

        let error = cache
            .restore_cached_prefix(&[1, 2])
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "cannot attach a cached prefix after the request block table has been populated"
        );
        assert_eq!(
            cache.physical_page_pool.page_states[cached_page_ids_by_layer[0][0].0],
            PageState::WrittenInUse { reader_count: 1 }
        );
        Ok(())
    }

    #[test]
    fn retains_complete_blocks_when_a_request_finishes() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(2, 2, 8, true)?;
        for layer_index in 0..2 {
            cache.append(layer_index, 0, &cache_tensor(0, 5)?, &cache_tensor(100, 5)?)?;
        }
        let active_page_ids_by_layer: Vec<Vec<_>> = cache
            .request_state
            .page_ids_by_layer()
            .map(<[PageId]>::to_vec)
            .collect();

        // The final token ID has not been processed by the model and is not part of the active
        // KV cache. Request finalization indexes only the five tokens that have cached values.
        let newly_indexed_token_count = finish_request(&mut cache, &[1, 2, 3, 4, 5, 6])?;
        assert_eq!(newly_indexed_token_count, 4);

        assert!(!cache.request_state.is_block_table_populated());
        for layer_page_ids in &active_page_ids_by_layer {
            assert_eq!(
                cache.physical_page_pool.page_states[layer_page_ids[0].0],
                PageState::WrittenNotInUse
            );
            assert_eq!(
                cache.physical_page_pool.page_states[layer_page_ids[1].0],
                PageState::WrittenNotInUse
            );
            assert_eq!(
                cache.physical_page_pool.page_states[layer_page_ids[2].0],
                PageState::Unallocated
            );
        }
        let prefix_match = cache
            .prefix_block_index
            .as_ref()
            .context("prefix caching should be enabled")?
            .find_longest_cached_prefix(&[1, 2, 3, 4, 5, 6])?;
        assert_eq!(
            prefix_match.page_ids_by_layer,
            active_page_ids_by_layer
                .iter()
                .map(|page_ids| page_ids[..2].to_vec())
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn repeatedly_publishes_only_new_complete_prefix_blocks() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 4, true)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;

        cache
            .cache
            .publish_completed_prefix(&mut cache.request_state, &[1, 2])?;
        assert_eq!(cache.request_state.accumulated_indexed_token_count(), 2);
        cache
            .cache
            .publish_completed_prefix(&mut cache.request_state, &[1, 2])?;
        assert_eq!(cache.request_state.accumulated_indexed_token_count(), 2);
        assert_eq!(
            cache
                .request_state
                .prefix_block_cursor()
                .matched_token_count(),
            2
        );

        cache.append(0, 2, &cache_tensor(2, 2)?, &cache_tensor(102, 2)?)?;
        cache
            .cache
            .publish_completed_prefix(&mut cache.request_state, &[1, 2, 3, 4])?;
        assert_eq!(cache.request_state.accumulated_indexed_token_count(), 4);
        cache
            .cache
            .publish_completed_prefix(&mut cache.request_state, &[1, 2, 3, 4])?;
        assert_eq!(cache.request_state.accumulated_indexed_token_count(), 4);
        assert_eq!(
            cache
                .request_state
                .prefix_block_cursor()
                .matched_token_count(),
            4
        );
        assert_eq!(
            cache
                .prefix_block_index
                .as_ref()
                .context("prefix caching should be enabled")?
                .block_count(),
            2
        );
        Ok(())
    }

    #[test]
    fn does_not_retain_recomputed_pages_for_existing_prefix_blocks() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 4, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let indexed_page_ids = cache.page_ids(0).to_vec();
        let first_newly_indexed_token_count = finish_request(&mut cache, &[1, 2, 3, 4])?;
        assert_eq!(first_newly_indexed_token_count, 4);

        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let recomputed_page_ids = cache.page_ids(0).to_vec();
        let second_newly_indexed_token_count = finish_request(&mut cache, &[1, 2, 3, 4])?;
        assert_eq!(second_newly_indexed_token_count, 0);

        for page_id in &indexed_page_ids {
            assert_eq!(
                cache.physical_page_pool.page_states[page_id.0],
                PageState::WrittenNotInUse
            );
        }
        for page_id in &recomputed_page_ids {
            assert_eq!(
                cache.physical_page_pool.page_states[page_id.0],
                PageState::Unallocated
            );
        }
        assert_eq!(
            cache
                .prefix_block_index
                .as_ref()
                .context("prefix caching should be enabled")?
                .find_longest_cached_prefix(&[1, 2, 3, 4])?
                .page_ids_by_layer,
            vec![indexed_page_ids]
        );
        Ok(())
    }

    #[test]
    fn advances_publication_cursor_past_blocks_indexed_by_another_request() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 6, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        assert_eq!(finish_request(&mut cache, &[1, 2, 3, 4])?, 4);

        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        cache
            .cache
            .publish_completed_prefix(&mut cache.request_state, &[1, 2, 3, 4])?;
        assert_eq!(cache.request_state.accumulated_indexed_token_count(), 0);
        assert_eq!(
            cache
                .request_state
                .prefix_block_cursor()
                .matched_token_count(),
            4
        );

        cache.append(0, 4, &cache_tensor(4, 2)?, &cache_tensor(104, 2)?)?;
        cache
            .cache
            .publish_completed_prefix(&mut cache.request_state, &[1, 2, 3, 4, 5, 6])?;
        assert_eq!(
            cache
                .request_state
                .prefix_block_cursor()
                .matched_token_count(),
            6
        );
        assert_eq!(cache.request_state.accumulated_indexed_token_count(), 2);
        assert_eq!(
            cache
                .prefix_block_index
                .as_ref()
                .context("prefix caching should be enabled")?
                .block_count(),
            3
        );
        Ok(())
    }

    #[test]
    fn reports_prefix_cache_telemetry() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 2, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        assert_eq!(finish_request(&mut cache, &[1, 2, 3, 4])?, 4);

        assert_eq!(cache.restore_cached_prefix(&[1, 2, 9])?, 2);
        cache.reset_active_block_tables()?;
        assert_eq!(cache.restore_cached_prefix(&[9, 9])?, 0);
        cache.reset_active_block_tables()?;

        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;

        assert_eq!(
            cache.prefix_cache_telemetry(),
            Some(PrefixCacheTelemetry {
                token_capacity: 4,
                current_indexed_token_count: 2,
                cumulative_indexed_token_count: 4,
                cumulative_evicted_token_count: 2,
                cumulative_restored_token_count: 2,
                cumulative_lookup_count: 2,
                cumulative_hit_count: 1,
            })
        );
        Ok(())
    }

    #[test]
    fn omits_prefix_cache_telemetry_when_prefix_caching_is_disabled() -> Result<()> {
        let cache = paged_cache(1, 2, 2)?;

        assert_eq!(cache.prefix_cache_telemetry(), None);
        Ok(())
    }

    #[test]
    fn resumes_indexing_after_a_restored_prefix() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 4, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let prefix_page_ids = cache.page_ids(0).to_vec();
        finish_request(&mut cache, &[1, 2, 3, 4])?;

        assert_eq!(cache.restore_cached_prefix(&[1, 2, 3, 4, 5, 6])?, 4);
        cache.append(0, 4, &cache_tensor(4, 2)?, &cache_tensor(104, 2)?)?;
        let suffix_page_id = cache.page_ids(0)[2];
        finish_request(&mut cache, &[1, 2, 3, 4, 5, 6])?;

        let prefix_match = cache
            .prefix_block_index
            .as_ref()
            .context("prefix caching should be enabled")?
            .find_longest_cached_prefix(&[1, 2, 3, 4, 5, 6])?;
        assert_eq!(
            prefix_match.page_ids_by_layer,
            vec![vec![prefix_page_ids[0], prefix_page_ids[1], suffix_page_id]]
        );
        for page_id in prefix_match.page_ids_by_layer[0].iter().copied() {
            assert_eq!(
                cache.physical_page_pool.page_states[page_id.0],
                PageState::WrittenNotInUse
            );
        }
        Ok(())
    }

    #[test]
    fn evicts_an_inactive_prefix_leaf_when_appending_at_capacity() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 2, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let original_page_ids = cache.page_ids(0).to_vec();
        finish_request(&mut cache, &[1, 2, 3, 4])?;

        cache.append(0, 0, &cache_tensor(10, 2)?, &cache_tensor(110, 2)?)?;

        assert_eq!(cache.page_ids(0), vec![original_page_ids[1]]);
        assert_eq!(
            cache
                .prefix_block_index
                .as_ref()
                .context("prefix caching should be enabled")?
                .find_longest_cached_prefix(&[1, 2, 3, 4])?
                .page_ids_by_layer,
            vec![vec![original_page_ids[0]]]
        );
        Ok(())
    }

    #[test]
    fn evicts_multiple_prefix_leaves_when_an_append_needs_multiple_pages() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 3, true)?;
        cache.append(0, 0, &cache_tensor(0, 6)?, &cache_tensor(100, 6)?)?;
        let original_page_ids = cache.page_ids(0).to_vec();
        finish_request(&mut cache, &[1, 2, 3, 4, 5, 6])?;

        cache.append(0, 0, &cache_tensor(10, 4)?, &cache_tensor(110, 4)?)?;

        let active_page_ids = &cache.page_ids(0);
        assert_eq!(active_page_ids.len(), 2);
        assert!(active_page_ids.contains(&original_page_ids[1]));
        assert!(active_page_ids.contains(&original_page_ids[2]));
        assert_eq!(
            cache
                .prefix_block_index
                .as_ref()
                .context("prefix caching should be enabled")?
                .find_longest_cached_prefix(&[1, 2, 3, 4, 5, 6])?
                .page_ids_by_layer,
            vec![vec![original_page_ids[0]]]
        );
        Ok(())
    }

    #[test]
    fn does_not_evict_pages_attached_to_the_active_prefix() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 2, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        finish_request(&mut cache, &[1, 2, 3, 4])?;
        assert_eq!(cache.restore_cached_prefix(&[1, 2, 3, 4])?, 4);

        let error = cache
            .append(0, 4, &cache_tensor(4, 2)?, &cache_tensor(104, 2)?)
            .err()
            .context("append should fail while every cached page is active")?
            .to_string();

        assert!(error.contains("only 0 of 2 are available"), "{error}");
        assert_eq!(
            cache
                .prefix_block_index
                .as_ref()
                .context("prefix caching should be enabled")?
                .find_longest_cached_prefix(&[1, 2, 3, 4])?
                .page_ids_by_layer
                .first()
                .map_or(0, Vec::len),
            2
        );
        Ok(())
    }

    #[test]
    fn does_not_evict_a_prefix_attached_to_another_request() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 3, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        finish_request(&mut cache, &[1, 2, 3, 4])?;

        let mut other_request_state = RequestPagedCacheState::new(1, 2);
        assert_eq!(
            cache
                .cache
                .restore_cached_prefix(&mut other_request_state, &[1, 2, 3, 4],)?,
            4
        );

        let error = cache
            .append(0, 0, &cache_tensor(10, 4)?, &cache_tensor(110, 4)?)
            .err()
            .context("append should fail while another request uses the cached prefix")?
            .to_string();

        assert!(error.contains("only 1 of 3 are available"), "{error}");
        assert_eq!(
            cache
                .prefix_block_index
                .as_ref()
                .context("prefix caching should be enabled")?
                .find_longest_cached_prefix(&[1, 2, 3, 4])?
                .page_ids_by_layer
                .first()
                .map_or(0, Vec::len),
            2
        );
        Ok(())
    }

    #[test]
    fn rolls_back_retains_across_layers_when_prefix_attachment_fails() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(2, 2, 2, true)?;
        let allocated_page_id = cache.physical_page_pool.allocate_page()?;
        cache
            .prefix_block_index
            .as_mut()
            .context("prefix caching should be enabled")?
            .index_cached_sequence(
                &PrefixBlockCursor::at_root(),
                &[1, 2],
                [&[allocated_page_id][..], &[PageId(usize::MAX)][..]].into_iter(),
            )?;

        assert!(cache.restore_cached_prefix(&[1, 2]).is_err());
        assert_eq!(
            cache.physical_page_pool.page_states[allocated_page_id.0],
            PageState::PendingWrite
        );
        assert!(cache
            .request_state
            .page_ids_by_layer()
            .all(|page_ids| page_ids.is_empty()));
        Ok(())
    }

    #[test]
    fn retains_a_cached_page_until_its_prefix_entry_is_evicted() -> Result<()> {
        let mut cache = paged_cache(1, 2, 1)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let page_id = cache.page_ids(0)[0];

        cache.physical_page_pool.retain_allocated_page(page_id)?;
        assert_eq!(
            cache.physical_page_pool.page_states[page_id.0],
            PageState::WrittenInUse { reader_count: 1 }
        );
        cache.reset_active_block_tables()?;
        assert_eq!(
            cache.physical_page_pool.page_states[page_id.0],
            PageState::WrittenNotInUse
        );
        assert!(cache.physical_page_pool.free_page_ids.is_empty());

        cache.physical_page_pool.release_allocated_page(page_id)?;
        assert_eq!(
            cache.physical_page_pool.page_states[page_id.0],
            PageState::Unallocated
        );
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
            .unwrap_err();
        let error = format!("{error:#}");
        assert!(
            error.contains("cannot release an unallocated physical page"),
            "{error}"
        );
        assert_eq!(cache.physical_page_pool.free_page_ids, vec![page_id]);
        Ok(())
    }

    #[test]
    fn rejects_mutating_a_written_physical_page() -> Result<()> {
        let mut cache = paged_cache(1, 2, 1)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let page_id = cache.page_ids(0)[0];
        cache.physical_page_pool.retain_allocated_page(page_id)?;
        cache.truncate(1)?;

        let error = cache
            .append(0, 1, &cache_tensor(1, 1)?, &cache_tensor(101, 1)?)
            .err()
            .context("append should not mutate a written physical page")?
            .to_string();
        assert!(
            error.contains("cannot mutate written physical page"),
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
        assert_eq!(cache.cached_token_count(0), 0);
        Ok(())
    }
}
