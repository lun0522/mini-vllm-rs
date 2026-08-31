use crate::model_loaders::CachedKeyValue;
use crate::model_loaders::KvCache;
use crate::model_loaders::ModelRole;
use crate::model_runner::KvCacheType;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::Tensor;

const TOKEN_DIMENSION: usize = 2;

#[derive(Default)]
struct ContiguousLayerCache {
    key: Option<Tensor>,
    value: Option<Tensor>,
    token_count: usize,
}

/// Stores one growing, contiguous key tensor and value tensor per transformer layer.
pub(super) struct ContiguousKvCache {
    layer_caches: Vec<ContiguousLayerCache>,
}

impl ContiguousKvCache {
    fn new(layer_count: usize) -> Self {
        Self {
            layer_caches: (0..layer_count)
                .map(|_| ContiguousLayerCache::default())
                .collect(),
        }
    }
}

impl KvCache for ContiguousKvCache {
    fn clear(&mut self) {
        for layer_cache in &mut self.layer_caches {
            *layer_cache = ContiguousLayerCache::default();
        }
    }

    fn append(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        let Some(layer_cache) = self.layer_caches.get_mut(layer_index) else {
            bail!("invalid KV-cache layer {layer_index}");
        };
        let appended_token_count = validate_cache_append(
            layer_cache.token_count,
            layer_index,
            start_position,
            key,
            value,
        )?;
        let cached_key = match &layer_cache.key {
            Some(cached_key) => Tensor::cat(&[cached_key, key], TOKEN_DIMENSION)?,
            None => key.clone(),
        };
        let cached_value = match &layer_cache.value {
            Some(cached_value) => Tensor::cat(&[cached_value, value], TOKEN_DIMENSION)?,
            None => value.clone(),
        };
        layer_cache.key = Some(cached_key.clone());
        layer_cache.value = Some(cached_value.clone());
        layer_cache.token_count += appended_token_count;
        Ok(CachedKeyValue {
            key: cached_key,
            value: cached_value,
        })
    }
}

#[derive(Default)]
struct PagedLayerCache {
    page_ids: Vec<usize>,
    token_count: usize,
}

/// Stores KV caches in reusable physical pages addressed through per-layer block tables.
///
/// The key and value pools are growing lists whose matching indices form a key/value page pair.
/// Each layer records the page IDs assigned to it, while the cache separately tracks page IDs that
/// were released and can be reassigned. Pages are allocated or reused as each layer's cache grows.
pub(super) struct PagedKvCache {
    page_token_count: usize,
    key_pages: Vec<Tensor>,
    value_pages: Vec<Tensor>,
    free_page_ids: Vec<usize>,
    layer_caches: Vec<PagedLayerCache>,
}

impl PagedKvCache {
    fn new(layer_count: usize, page_token_count: usize) -> Self {
        Self {
            page_token_count,
            key_pages: Vec::new(),
            value_pages: Vec::new(),
            free_page_ids: Vec::new(),
            layer_caches: (0..layer_count)
                .map(|_| PagedLayerCache::default())
                .collect(),
        }
    }

    /// Allocates or reuses a key/value page pair and returns their shared page ID.
    fn allocate_page(&mut self, key: &Tensor, value: &Tensor) -> Result<usize> {
        if let Some(page_id) = self.free_page_ids.pop() {
            return Ok(page_id);
        }

        let page_id = self.key_pages.len();
        self.key_pages
            .push(create_new_page(key, self.page_token_count)?);
        self.value_pages
            .push(create_new_page(value, self.page_token_count)?);
        Ok(page_id)
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
        let (batch_size, head_count, _, head_dimension) = self.key_pages[page_id].dims4()?;
        let page_ranges = [
            0..batch_size,
            0..head_count,
            page_offset..page_offset + token_count,
            0..head_dimension,
        ];
        self.key_pages[page_id] = self.key_pages[page_id].slice_assign(&page_ranges, &key_slice)?;
        self.value_pages[page_id] =
            self.value_pages[page_id].slice_assign(&page_ranges, &value_slice)?;
        Ok(())
    }

    fn reconstruct_full_cache(&self, layer_index: usize) -> Result<CachedKeyValue> {
        let layer_cache = &self.layer_caches[layer_index];
        Ok(CachedKeyValue {
            key: reconstruct_contiguous_tensor(
                &self.key_pages,
                layer_cache,
                self.page_token_count,
            )?,
            value: reconstruct_contiguous_tensor(
                &self.value_pages,
                layer_cache,
                self.page_token_count,
            )?,
        })
    }
}

impl KvCache for PagedKvCache {
    fn clear(&mut self) {
        for layer_cache in &mut self.layer_caches {
            self.free_page_ids.append(&mut layer_cache.page_ids);
            layer_cache.token_count = 0;
        }
    }

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
        let appended_token_count =
            validate_cache_append(current_token_count, layer_index, start_position, key, value)?;
        let mut input_offset = 0;
        while input_offset < appended_token_count {
            let page_offset = self.layer_caches[layer_index].token_count % self.page_token_count;
            if page_offset == 0 {
                let page_id = self.allocate_page(key, value)?;
                self.layer_caches[layer_index].page_ids.push(page_id);
            }
            let page_id = self.layer_caches[layer_index]
                .page_ids
                .last()
                .copied()
                .context("partial KV-cache page is missing from the block table")?;
            let written_token_count =
                (self.page_token_count - page_offset).min(appended_token_count - input_offset);
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
        self.reconstruct_full_cache(layer_index)
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
    let appended_token_count = key_dimensions.2;
    if appended_token_count == 0 {
        bail!("cannot append an empty KV-cache tensor");
    }
    Ok(appended_token_count)
}

fn create_new_page(source: &Tensor, page_token_count: usize) -> Result<Tensor> {
    // TODO: Allocate monolithic key/value tensor pools and suballocate page views from them.
    let (batch_size, head_count, _, head_dimension) = source.dims4()?;
    Ok(Tensor::zeros(
        (batch_size, head_count, page_token_count, head_dimension),
        source.dtype(),
        source.device(),
    )?)
}

/// Reads pages in logical block-table order and returns one contiguous attention tensor.
///
/// Every page except the last is full. The last page is narrowed to its valid token count before
/// concatenation so unwritten page capacity never reaches the attention calculation.
fn reconstruct_contiguous_tensor(
    pages: &[Tensor],
    layer_cache: &PagedLayerCache,
    page_token_count: usize,
) -> Result<Tensor> {
    let mut remaining_token_count = layer_cache.token_count;
    let mut page_slices = Vec::with_capacity(layer_cache.page_ids.len());
    for &page_id in &layer_cache.page_ids {
        let slice_token_count = page_token_count.min(remaining_token_count);
        page_slices.push(pages[page_id].narrow(TOKEN_DIMENSION, 0, slice_token_count)?);
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
    kv_cache_bytes_per_token: usize,
    layer_count: usize,
    model_role: ModelRole,
) -> Box<dyn KvCache> {
    match kv_cache_type {
        KvCacheType::Contiguous => Box::new(ContiguousKvCache::new(layer_count)),
        KvCacheType::Paged { page_token_count } => {
            let page_size_bytes = 2 * page_token_count * kv_cache_bytes_per_token;
            log::info!(
                "Creating {model_role} model paged KV cache with {page_token_count} tokens per page \
                ({page_size_bytes} bytes per virtual page)"
            );
            Box::new(PagedKvCache::new(layer_count, page_token_count))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    const PAGE_TOKEN_COUNT: usize = 16;

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

    #[test]
    fn allocates_pages_for_short_full_and_long_prompts() -> Result<()> {
        for (page_token_count, token_count, expected_page_count) in [
            (PAGE_TOKEN_COUNT, 15, 1),
            (PAGE_TOKEN_COUNT, 16, 1),
            (PAGE_TOKEN_COUNT, 17, 2),
            (2, 3, 2),
        ] {
            let mut cache = PagedKvCache::new(/* layer_count */ 1, page_token_count);
            let key = cache_tensor(0, token_count)?;
            let value = cache_tensor(100, token_count)?;
            let cached = cache.append(0, 0, &key, &value)?;
            assert_eq!(cache.layer_caches[0].page_ids.len(), expected_page_count);
            assert!(cache.key_pages.iter().all(|page| page
                .dim(TOKEN_DIMENSION)
                .is_ok_and(|size| size == page_token_count)));
            assert!(cache.value_pages.iter().all(|page| page
                .dim(TOKEN_DIMENSION)
                .is_ok_and(|size| size == page_token_count)));
            assert_eq!(tensor_values(&cached.key)?, tensor_values(&key)?);
            assert_eq!(tensor_values(&cached.value)?, tensor_values(&value)?);
        }
        Ok(())
    }

    #[test]
    fn contiguous_and_paged_caches_match_across_appends() -> Result<()> {
        let mut contiguous = ContiguousKvCache::new(/* layer_count */ 1);
        let mut paged = PagedKvCache::new(/* layer_count */ 1, /* page_token_count */ 2);

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
    fn appends_across_a_page_boundary() -> Result<()> {
        let mut cache = PagedKvCache::new(/* layer_count */ 1, PAGE_TOKEN_COUNT);
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
        let mut cache = PagedKvCache::new(/* layer_count */ 2, PAGE_TOKEN_COUNT);
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
        let mut cache = PagedKvCache::new(/* layer_count */ 1, PAGE_TOKEN_COUNT);
        let key = cache_tensor(0, 17)?;
        let value = cache_tensor(100, 17)?;
        cache.append(0, 0, &key, &value)?;
        let allocated_page_count = cache.key_pages.len();
        let mut original_page_ids = cache.layer_caches[0].page_ids.clone();
        cache.clear();
        cache.append(0, 0, &key, &value)?;
        let mut reused_page_ids = cache.layer_caches[0].page_ids.clone();
        original_page_ids.sort_unstable();
        reused_page_ids.sort_unstable();
        assert_eq!(cache.key_pages.len(), allocated_page_count);
        assert_eq!(reused_page_ids, original_page_ids);
        assert!(cache.free_page_ids.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_an_inconsistent_start_position() -> Result<()> {
        let mut cache = PagedKvCache::new(/* layer_count */ 1, PAGE_TOKEN_COUNT);
        let key = cache_tensor(0, 1)?;
        let value = cache_tensor(100, 1)?;
        cache.append(0, 0, &key, &value)?;
        assert!(cache.append(0, 0, &key, &value).is_err());
        Ok(())
    }
}
