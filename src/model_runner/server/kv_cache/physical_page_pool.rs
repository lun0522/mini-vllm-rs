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

/// Tracks whether a physical page may be written, reused, or evicted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PageState {
    /// Contains no useful KV values and has a page ID in the free list.
    Unallocated,
    /// Belongs to the active request and may still receive KV values.
    PendingWrite,
    /// Contains immutable cached KV values but is not used by an active request.
    /// It cannot be overwritten until eviction makes it `Unallocated` and it is allocated again.
    WrittenNotInUse,
    /// Contains immutable cached KV values used by one or more active requests.
    WrittenInUse { reader_count: u32 },
}

impl PageState {
    fn retain(&mut self) -> Result<()> {
        *self = match *self {
            Self::Unallocated => bail!("cannot retain an unallocated physical page"),
            Self::PendingWrite | Self::WrittenNotInUse => Self::WrittenInUse { reader_count: 1 },
            Self::WrittenInUse { reader_count } => Self::WrittenInUse {
                reader_count: reader_count
                    .checked_add(1)
                    .context("physical page reader count overflow")?,
            },
        };
        Ok(())
    }

    fn release(&mut self) -> Result<bool> {
        *self = match *self {
            Self::Unallocated => bail!("cannot release an unallocated physical page"),
            Self::PendingWrite | Self::WrittenNotInUse => Self::Unallocated,
            Self::WrittenInUse { reader_count: 1 } => Self::WrittenNotInUse,
            Self::WrittenInUse { reader_count } if reader_count > 1 => Self::WrittenInUse {
                reader_count: reader_count - 1,
            },
            Self::WrittenInUse { .. } => bail!("written physical page has no active readers"),
        };
        Ok(matches!(self, Self::Unallocated))
    }

    fn is_writable(self) -> bool {
        matches!(self, Self::PendingWrite)
    }
}

/// Owns physical key/value tensors, page ownership states, and reusable page IDs.
///
/// "Physical" means these pages are actual storage locations in the preallocated tensor pools.
/// They exist independently of whichever active sequence or cached prefix refers to them. A page
/// becomes reusable only after all such owners release it.
pub(super) struct PhysicalPagePool {
    pub(super) per_page_token_count: usize,
    pub(super) page_count: usize,
    pub(super) key_pool: Tensor,
    pub(super) value_pool: Tensor,
    pub(super) page_states: Vec<PageState>,
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
            page_states: vec![PageState::Unallocated; per_pool_page_count],
            free_page_ids: (0..per_pool_page_count).rev().map(PageId).collect(),
        })
    }

    /// Allocates a key/value page pair for the active block tables.
    ///
    /// The page becomes writable by the active block tables, so the caller must not
    /// immediately call `retain_allocated_page` for the same ownership.
    pub(super) fn allocate_page(&mut self) -> Result<PageId> {
        let page_id = self
            .free_page_ids
            .pop()
            .context("paged KV cache has no free physical pages")?;
        self.page_states[page_id.0] = PageState::PendingWrite;
        Ok(page_id)
    }

    /// Adds another owner, such as a cached-prefix entry, to an allocated page.
    pub(super) fn retain_allocated_page(&mut self, page_id: PageId) -> Result<()> {
        let page_state = self
            .page_states
            .get_mut(page_id.0)
            .with_context(|| format!("invalid physical page ID {}", page_id.0))?;
        page_state
            .retain()
            .with_context(|| format!("failed to retain physical page {}", page_id.0))
    }

    /// Retains every page as one transaction.
    ///
    /// If retaining a page fails, previously retained pages are released and the remaining pages
    /// are left untouched.
    pub(super) fn retain_allocated_pages_or_rollback<'a>(
        &mut self,
        page_ids: impl Iterator<Item = &'a PageId>,
    ) -> Result<()> {
        let mut previous_states: Vec<(PageId, PageState)> = Vec::new();
        for &page_id in page_ids {
            let Some(previous_state) = self.page_states.get(page_id.0).copied() else {
                for &(retained_page_id, previous_state) in &previous_states {
                    self.page_states[retained_page_id.0] = previous_state;
                }
                bail!("invalid physical page ID {}", page_id.0);
            };
            if let Err(error) = self.retain_allocated_page(page_id) {
                for (retained_page_id, previous_state) in previous_states {
                    self.page_states[retained_page_id.0] = previous_state;
                }
                return Err(error);
            }
            previous_states.push((page_id, previous_state));
        }
        Ok(())
    }

    /// Releases one owner and makes the page reusable after its final owner is removed.
    pub(super) fn release_allocated_page(&mut self, page_id: PageId) -> Result<()> {
        let page_state = self
            .page_states
            .get_mut(page_id.0)
            .with_context(|| format!("invalid physical page ID {}", page_id.0))?;
        if page_state
            .release()
            .with_context(|| format!("failed to release physical page {}", page_id.0))?
        {
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
        if !self.page_states[page_id.0].is_writable() {
            bail!("cannot mutate written physical page {}", page_id.0);
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
