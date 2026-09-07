use super::prefix_index::PrefixBlockCursor;
use super::prefix_index::PrefixBlockIndex;
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
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;
use thousands::Separable;

/// Maps one model layer's logical token order to pages in `PhysicalPagePool`.
#[derive(Default)]
struct LayerBlockTable {
    page_ids: Vec<PageId>,
    token_count: usize,
}

impl LayerCache for LayerBlockTable {
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
    layer_block_tables: Vec<LayerBlockTable>,
}

impl ActiveBlockTables {
    fn new(layer_count: usize) -> Self {
        Self {
            layer_block_tables: (0..layer_count)
                .map(|_| LayerBlockTable::default())
                .collect(),
        }
    }

    fn is_populated(&self) -> bool {
        self.layer_block_tables.iter().any(|layer_block_table| {
            layer_block_table.token_count != 0 || !layer_block_table.page_ids.is_empty()
        })
    }

    fn cached_token_count(&self) -> usize {
        self.layer_block_tables
            .first()
            .map_or(0, LayerCache::cached_token_count)
    }

    fn layer_count(&self) -> usize {
        self.layer_block_tables.len()
    }

    fn layer_block_table(&self, layer_index: usize) -> Option<&LayerBlockTable> {
        self.layer_block_tables.get(layer_index)
    }

    fn layer_cached_token_count(&self, layer_index: usize) -> Option<usize> {
        self.layer_block_table(layer_index)
            .map(LayerCache::cached_token_count)
    }

    fn validate_truncation(&self, target_token_count: usize) -> Result<()> {
        validate_truncation(&self.layer_block_tables, target_token_count)
    }

    fn append_page(&mut self, layer_index: usize, page_id: PageId) {
        self.layer_block_tables[layer_index].page_ids.push(page_id);
    }

    fn last_page_id(&self, layer_index: usize) -> Option<PageId> {
        self.layer_block_tables[layer_index]
            .page_ids
            .last()
            .copied()
    }

    fn record_appended_tokens(&mut self, layer_index: usize, token_count: usize) {
        self.layer_block_tables[layer_index].token_count += token_count;
    }

    fn page_ids_by_layer(&self) -> Vec<Vec<PageId>> {
        self.layer_block_tables
            .iter()
            .map(|layer_block_table| layer_block_table.page_ids.clone())
            .collect()
    }

    fn attach_cached_prefix(
        &mut self,
        page_ids_by_layer: Vec<Vec<PageId>>,
        matched_token_count: usize,
    ) {
        for (layer_block_table, page_ids) in
            self.layer_block_tables.iter_mut().zip(page_ids_by_layer)
        {
            layer_block_table.page_ids = page_ids;
            layer_block_table.token_count = matched_token_count;
        }
    }

    /// Truncates every virtual table and returns the physical pages it no longer references.
    fn truncate(&mut self, retained_page_count: usize, target_token_count: usize) -> Vec<PageId> {
        let mut released_page_ids = Vec::new();
        for layer_block_table in &mut self.layer_block_tables {
            released_page_ids.extend(layer_block_table.page_ids.split_off(retained_page_count));
            layer_block_table.token_count = target_token_count;
        }
        released_page_ids
    }

    /// Clears every virtual table and returns all physical pages it referenced.
    fn reset(&mut self) -> Vec<PageId> {
        let mut released_page_ids = Vec::new();
        for layer_block_table in &mut self.layer_block_tables {
            released_page_ids.extend(std::mem::take(&mut layer_block_table.page_ids));
            layer_block_table.token_count = 0;
        }
        released_page_ids
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

    /// Retains every page as one transaction.
    ///
    /// If retaining a page fails, previously retained pages are released and the remaining pages
    /// are left untouched.
    fn retain_allocated_pages_or_rollback<'a>(
        &mut self,
        page_ids: impl Clone + Iterator<Item = &'a PageId>,
    ) -> Result<()> {
        for (retained_page_count, &page_id) in page_ids.clone().enumerate() {
            if let Err(error) = self.retain_allocated_page(page_id) {
                for &retained_page_id in page_ids.clone().take(retained_page_count) {
                    self.release_allocated_page(retained_page_id)?;
                }
                return Err(error);
            }
        }
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
    layer_block_table: &LayerBlockTable,
    per_page_token_count: usize,
) -> Result<Tensor> {
    let mut remaining_token_count = layer_block_table.token_count;
    let mut page_slices = Vec::with_capacity(layer_block_table.page_ids.len());
    for &page_id in &layer_block_table.page_ids {
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

/// Orchestrates physical KV pages, active virtual block tables, and reusable prefix metadata.
pub(in crate::model_runner::server) struct PagedKvCache {
    physical_page_pool: PhysicalPagePool,
    active_block_tables: ActiveBlockTables,
    prefix_block_index: Option<PrefixBlockIndex>,
    prefix_block_cursor: Box<PrefixBlockCursor>,
}

// TODO: After allocation-pressure eviction is implemented, move `PhysicalPagePool` and
// `ActiveBlockTables` into dedicated files while keeping `PagedKvCache` as their orchestrator.

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
            prefix_block_cursor: Box::new(PrefixBlockCursor::at_root()),
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
            let page_offset = self
                .active_block_tables
                .layer_cached_token_count(layer_index)
                .context("KV-cache layer is missing from the active block tables")?
                % self.physical_page_pool.per_page_token_count;
            if page_offset == 0 {
                let page_id = self.physical_page_pool.allocate_page()?;
                self.active_block_tables.append_page(layer_index, page_id);
            }
            let page_id = self
                .active_block_tables
                .last_page_id(layer_index)
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
            self.active_block_tables
                .record_appended_tokens(layer_index, written_token_count);
            input_offset += written_token_count;
        }
        Ok(())
    }

    fn reconstruct_full_cache(&self, layer_index: usize) -> Result<CachedKeyValue> {
        let layer_block_table = self
            .active_block_tables
            .layer_block_table(layer_index)
            .context("KV-cache layer is missing from the active block tables")?;
        Ok(CachedKeyValue {
            key: reconstruct_contiguous_tensor(
                &self.physical_page_pool.key_pool,
                layer_block_table,
                self.physical_page_pool.per_page_token_count,
            )?,
            value: reconstruct_contiguous_tensor(
                &self.physical_page_pool.value_pool,
                layer_block_table,
                self.physical_page_pool.per_page_token_count,
            )?,
        })
    }

    /// Attaches the longest reusable prefix to the current request and returns its token count.
    pub(super) fn restore_cached_prefix(&mut self, input_token_ids: &[u32]) -> Result<usize> {
        let Some(prefix_block_index) = self.prefix_block_index.as_ref() else {
            return Ok(0);
        };
        let prefix_match = prefix_block_index.find_longest_cached_prefix(input_token_ids)?;
        if self.active_block_tables.is_populated() {
            bail!("cannot attach a cached prefix to non-empty active block tables");
        }
        let matched_token_count = prefix_match.cursor.matched_token_count();
        self.physical_page_pool
            .retain_allocated_pages_or_rollback(prefix_match.page_ids_by_layer.iter().flatten())?;
        self.active_block_tables
            .attach_cached_prefix(prefix_match.page_ids_by_layer, matched_token_count);
        *self.prefix_block_cursor = prefix_match.cursor;
        Ok(matched_token_count)
    }

    pub(super) fn retain_completed_blocks(&mut self, token_ids: &[u32]) -> Result<()> {
        let cached_token_count = self.active_block_tables.cached_token_count();
        let cached_token_ids = token_ids
            .get(..cached_token_count)
            .context("active KV cache contains more tokens than the completed request")?;
        if let Some(prefix_block_index) = self.prefix_block_index.as_mut() {
            let cached_page_ids_by_layer = self.active_block_tables.page_ids_by_layer();
            let indexing_result = prefix_block_index.index_cached_sequence(
                &self.prefix_block_cursor,
                cached_token_ids,
                &cached_page_ids_by_layer,
            );
            // This cursor describes only the request being finalized. Reset it so the next
            // request cannot accidentally resume indexing from the previous restored prefix.
            *self.prefix_block_cursor = PrefixBlockCursor::at_root();
            let newly_indexed_pages = indexing_result?;
            self.physical_page_pool
                .retain_allocated_pages_or_rollback(newly_indexed_pages.page_ids.iter())?;
        }
        Ok(())
    }

    pub(super) fn truncate(&mut self, target_token_count: usize) -> Result<()> {
        self.active_block_tables
            .validate_truncation(target_token_count)?;
        let retained_page_count =
            target_token_count.div_ceil(self.physical_page_pool.per_page_token_count);
        let released_page_ids = self
            .active_block_tables
            .truncate(retained_page_count, target_token_count);
        self.physical_page_pool
            .release_allocated_pages(released_page_ids)
    }

    pub(super) fn reset_active_block_tables(&mut self) -> Result<()> {
        *self.prefix_block_cursor = PrefixBlockCursor::at_root();
        let released_page_ids = self.active_block_tables.reset();
        self.physical_page_pool
            .release_allocated_pages(released_page_ids)
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.physical_page_pool.page_count / self.active_block_tables.layer_count()
            * self.physical_page_pool.per_page_token_count
    }

    pub(super) fn append(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        let Some(layer_block_table) = self.active_block_tables.layer_block_table(layer_index)
        else {
            bail!("invalid KV-cache layer {layer_index}");
        };
        let current_token_count = layer_block_table.token_count;
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
    ) -> Result<PagedKvCache> {
        let model_info = test_model_info(layer_count);
        let total_size_bytes =
            2 * per_pool_page_count * per_page_token_count * model_info.kv_cache_bytes_per_token();
        PagedKvCache::new(
            &model_info,
            ModelRole::Target,
            &Device::Cpu,
            per_page_token_count,
            enable_prefix_caching,
            total_size_bytes,
        )
    }

    fn retain_active_pages_for_prefix(cache: &mut PagedKvCache) -> Result<Vec<Vec<PageId>>> {
        let cached_page_ids_by_layer: Vec<_> = cache
            .active_block_tables
            .layer_block_tables
            .iter()
            .map(|layer_block_table| layer_block_table.page_ids.clone())
            .collect();
        for &page_id in cached_page_ids_by_layer.iter().flatten() {
            cache.physical_page_pool.retain_allocated_page(page_id)?;
        }
        Ok(cached_page_ids_by_layer)
    }

    fn finish_request(cache: &mut PagedKvCache, token_ids: &[u32]) -> Result<()> {
        cache.retain_completed_blocks(token_ids)?;
        cache.reset_active_block_tables()
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
                cache.active_block_tables.layer_block_tables[0]
                    .page_ids
                    .len(),
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

        assert_eq!(
            cache.active_block_tables.layer_block_tables[0]
                .page_ids
                .len(),
            2
        );
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
            cache.active_block_tables.layer_block_tables[0].page_ids,
            cache.active_block_tables.layer_block_tables[1].page_ids
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
        let mut original_page_ids = cache.active_block_tables.layer_block_tables[0]
            .page_ids
            .clone();

        cache.reset_active_block_tables()?;
        cache.append(0, 0, &key, &value)?;

        let mut reused_page_ids = cache.active_block_tables.layer_block_tables[0]
            .page_ids
            .clone();
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
                &cached_page_ids_by_layer,
            )?;
        cache.reset_active_block_tables()?;

        assert_eq!(cache.restore_cached_prefix(&[1, 2, 3, 4, 5])?, 4);
        for (layer_block_table, expected_page_ids) in cache
            .active_block_tables
            .layer_block_tables
            .iter()
            .zip(&cached_page_ids_by_layer)
        {
            assert_eq!(&layer_block_table.page_ids, expected_page_ids);
            assert_eq!(layer_block_table.token_count, 4);
            for page_id in expected_page_ids {
                assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 2);
            }
        }

        cache.reset_active_block_tables()?;
        for page_id in cached_page_ids_by_layer.iter().flatten() {
            assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 1);
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
                &cached_page_ids_by_layer,
            )?;
        cache.reset_active_block_tables()?;

        assert_eq!(cache.restore_cached_prefix(&[1, 2, 8, 9])?, 2);
        assert_eq!(
            cache.active_block_tables.layer_block_tables[0].page_ids,
            cached_page_ids_by_layer[0][..1]
        );
        assert_eq!(
            cache.active_block_tables.layer_block_tables[0].token_count,
            2
        );
        assert_eq!(
            cache.physical_page_pool.reference_counts[cached_page_ids_by_layer[0][0].0],
            2
        );
        assert_eq!(
            cache.physical_page_pool.reference_counts[cached_page_ids_by_layer[0][1].0],
            1
        );
        Ok(())
    }

    #[test]
    fn rejects_attaching_a_prefix_to_non_empty_active_block_tables() -> Result<()> {
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
                &cached_page_ids_by_layer,
            )?;

        let error = cache
            .restore_cached_prefix(&[1, 2])
            .unwrap_err()
            .to_string();
        assert_eq!(
            error,
            "cannot attach a cached prefix to non-empty active block tables"
        );
        assert_eq!(
            cache.physical_page_pool.reference_counts[cached_page_ids_by_layer[0][0].0],
            2
        );
        Ok(())
    }

    #[test]
    fn retains_complete_blocks_when_a_request_finishes() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(2, 2, 8, true)?;
        for layer_index in 0..2 {
            cache.append(layer_index, 0, &cache_tensor(0, 5)?, &cache_tensor(100, 5)?)?;
        }
        let active_page_ids_by_layer: Vec<_> = cache
            .active_block_tables
            .layer_block_tables
            .iter()
            .map(|layer_block_table| layer_block_table.page_ids.clone())
            .collect();

        // The final token ID has not been processed by the model and is not part of the active
        // KV cache. Request finalization indexes only the five tokens that have cached values.
        finish_request(&mut cache, &[1, 2, 3, 4, 5, 6])?;

        assert!(!cache.active_block_tables.is_populated());
        for layer_page_ids in &active_page_ids_by_layer {
            assert_eq!(
                cache.physical_page_pool.reference_counts[layer_page_ids[0].0],
                1
            );
            assert_eq!(
                cache.physical_page_pool.reference_counts[layer_page_ids[1].0],
                1
            );
            assert_eq!(
                cache.physical_page_pool.reference_counts[layer_page_ids[2].0],
                0
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
    fn does_not_retain_recomputed_pages_for_existing_prefix_blocks() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 4, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let indexed_page_ids = cache.active_block_tables.layer_block_tables[0]
            .page_ids
            .clone();
        finish_request(&mut cache, &[1, 2, 3, 4])?;

        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let recomputed_page_ids = cache.active_block_tables.layer_block_tables[0]
            .page_ids
            .clone();
        finish_request(&mut cache, &[1, 2, 3, 4])?;

        for page_id in &indexed_page_ids {
            assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 1);
        }
        for page_id in &recomputed_page_ids {
            assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 0);
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
    fn resumes_indexing_after_a_restored_prefix() -> Result<()> {
        let mut cache = paged_cache_with_prefix_caching(1, 2, 4, true)?;
        cache.append(0, 0, &cache_tensor(0, 4)?, &cache_tensor(100, 4)?)?;
        let prefix_page_ids = cache.active_block_tables.layer_block_tables[0]
            .page_ids
            .clone();
        finish_request(&mut cache, &[1, 2, 3, 4])?;

        assert_eq!(cache.restore_cached_prefix(&[1, 2, 3, 4, 5, 6])?, 4);
        cache.append(0, 4, &cache_tensor(4, 2)?, &cache_tensor(104, 2)?)?;
        let suffix_page_id = cache.active_block_tables.layer_block_tables[0].page_ids[2];
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
            assert_eq!(cache.physical_page_pool.reference_counts[page_id.0], 1);
        }
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
                &[vec![allocated_page_id], vec![PageId(usize::MAX)]],
            )?;

        assert!(cache.restore_cached_prefix(&[1, 2]).is_err());
        assert_eq!(
            cache.physical_page_pool.reference_counts[allocated_page_id.0],
            1
        );
        assert!(cache
            .active_block_tables
            .layer_block_tables
            .iter()
            .all(|layer_block_table| layer_block_table.page_ids.is_empty()));
        Ok(())
    }

    #[test]
    fn retains_a_cached_page_until_its_prefix_entry_is_evicted() -> Result<()> {
        let mut cache = paged_cache(1, 2, 1)?;
        cache.append(0, 0, &cache_tensor(0, 2)?, &cache_tensor(100, 2)?)?;
        let page_id = cache.active_block_tables.layer_block_tables[0].page_ids[0];

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
        let page_id = cache.active_block_tables.layer_block_tables[0].page_ids[0];
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
        assert_eq!(
            cache.active_block_tables.layer_block_tables[0].token_count,
            0
        );
        Ok(())
    }
}
