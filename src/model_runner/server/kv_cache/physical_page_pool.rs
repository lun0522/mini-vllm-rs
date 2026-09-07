use super::utils::allocate_pool;
use super::utils::pool_page;
use super::utils::TOKEN_DIMENSION;
use crate::models::ModelInfo;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use candle_core::Tensor;
use thousands::Separable;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PageId(pub(super) usize);

/// Owns physical key/value tensors, page reference counts, and reusable page IDs.
///
/// "Physical" means these pages are actual storage locations in the preallocated tensor pools.
/// They exist independently of whichever active sequence or cached prefix refers to them. A page
/// becomes reusable only after all such owners release their references.
pub(super) struct PhysicalPagePool {
    pub(super) per_page_token_count: usize,
    pub(super) page_count: usize,
    pub(super) key_pool: Tensor,
    pub(super) value_pool: Tensor,
    pub(super) reference_counts: Vec<usize>,
    pub(super) free_page_ids: Vec<PageId>,
}

impl PhysicalPagePool {
    pub(super) fn new(
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
    pub(super) fn allocate_page(&mut self) -> Result<PageId> {
        let page_id = self
            .free_page_ids
            .pop()
            .context("paged KV cache has no free physical pages")?;
        self.reference_counts[page_id.0] = 1;
        Ok(page_id)
    }

    /// Adds another owner, such as a cached-prefix entry, to an allocated page.
    pub(super) fn retain_allocated_page(&mut self, page_id: PageId) -> Result<()> {
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
    pub(super) fn retain_allocated_pages_or_rollback<'a>(
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
    pub(super) fn release_allocated_page(&mut self, page_id: PageId) -> Result<()> {
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

    pub(super) fn release_allocated_pages(
        &mut self,
        page_ids: impl IntoIterator<Item = PageId>,
    ) -> Result<()> {
        for page_id in page_ids {
            self.release_allocated_page(page_id)?;
        }
        Ok(())
    }

    pub(super) fn write_page(
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

    pub(super) fn validate_append_capacity(
        &self,
        current_token_count: usize,
        appending_token_count: usize,
    ) -> Result<()> {
        let required_page_count =
            self.compute_required_page_count(current_token_count, appending_token_count);
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

    pub(super) fn compute_required_page_count(
        &self,
        current_token_count: usize,
        appending_token_count: usize,
    ) -> usize {
        let remaining_tokens_in_last_page = match current_token_count % self.per_page_token_count {
            0 => 0,
            page_offset => self.per_page_token_count - page_offset,
        };
        appending_token_count
            .saturating_sub(remaining_tokens_in_last_page)
            .div_ceil(self.per_page_token_count)
    }

    pub(super) fn free_page_count(&self) -> usize {
        self.free_page_ids.len()
    }
}
