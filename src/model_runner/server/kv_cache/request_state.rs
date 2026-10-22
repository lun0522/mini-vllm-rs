use super::physical_page_pool::PageId;
use super::prefix_index::PrefixBlockCursor;
use super::prefix_index::PrefixBlockMatch;
use super::prefix_index::PrefixIndexingResult;
use super::utils::validate_truncation;
use super::LayerCache;
use anyhow::Context;
use anyhow::Result;

/// Maps one model layer's logical token order to physical KV-cache pages.
#[derive(Default)]
struct LayerBlockTable {
    page_ids: Vec<PageId>,
    cached_token_count: usize,
}

impl LayerCache for LayerBlockTable {
    fn cached_token_count(&self) -> usize {
        self.cached_token_count
    }
}

#[derive(Clone, Copy)]
pub(super) struct LayerCacheView<'a> {
    pub(super) page_ids: &'a [PageId],
    pub(super) cached_token_count: usize,
}

/// Publication accounting and replaced pages whose request ownership must be released.
pub(super) struct PrefixPublicationResult {
    pub(super) newly_indexed_token_count: usize,
    pub(super) replaced_page_ids: Vec<PageId>,
}

/// Holds the virtual page mappings and prefix-publication progress for one request.
pub struct RequestPagedCacheState {
    per_page_token_count: usize,
    layer_block_tables: Vec<LayerBlockTable>,
    prefix_block_cursor: PrefixBlockCursor,
    accumulated_indexed_token_count: usize,
}

impl RequestPagedCacheState {
    pub fn new(layer_count: usize, per_page_token_count: usize) -> Self {
        Self {
            per_page_token_count,
            layer_block_tables: (0..layer_count)
                .map(|_| LayerBlockTable::default())
                .collect(),
            prefix_block_cursor: PrefixBlockCursor::at_root(),
            accumulated_indexed_token_count: 0,
        }
    }

    pub(super) fn prefix_block_cursor(&self) -> &PrefixBlockCursor {
        &self.prefix_block_cursor
    }

    pub(super) fn accumulated_indexed_token_count(&self) -> usize {
        self.accumulated_indexed_token_count
    }

    pub(super) fn is_block_table_populated(&self) -> bool {
        self.layer_block_tables.iter().any(|layer_block_table| {
            layer_block_table.cached_token_count != 0 || !layer_block_table.page_ids.is_empty()
        })
    }

    pub(super) fn cached_token_count(&self) -> usize {
        // Model execution appends and truncates every layer to the same sequence length, so the
        // first layer represents the request-wide cached token count.
        self.layer_block_tables
            .first()
            .map_or(0, LayerCache::cached_token_count)
    }

    pub(super) fn layer_cache(&self, layer_index: usize) -> Option<LayerCacheView<'_>> {
        self.layer_block_tables
            .get(layer_index)
            .map(|layer_block_table| LayerCacheView {
                page_ids: &layer_block_table.page_ids,
                cached_token_count: layer_block_table.cached_token_count,
            })
    }

    pub(super) fn page_ids_by_layer(&self) -> impl Clone + Iterator<Item = &[PageId]> {
        self.layer_block_tables
            .iter()
            .map(|layer_block_table| layer_block_table.page_ids.as_slice())
    }

    pub(super) fn attach_cached_prefix(&mut self, prefix_match: PrefixBlockMatch) {
        let matched_token_count = prefix_match.cursor.matched_token_count();
        for (layer_block_table, page_ids) in self
            .layer_block_tables
            .iter_mut()
            .zip(prefix_match.page_ids_by_layer)
        {
            layer_block_table.page_ids = page_ids;
            layer_block_table.cached_token_count = matched_token_count;
        }
        self.prefix_block_cursor = prefix_match.cursor;
    }

    /// Truncates every layer table and returns the physical pages no longer referenced.
    pub(super) fn truncate(&mut self, target_token_count: usize) -> Result<Vec<PageId>> {
        validate_truncation(&self.layer_block_tables, target_token_count)?;
        let retained_page_count = target_token_count.div_ceil(self.per_page_token_count);
        Ok(self.truncate_unchecked(retained_page_count, target_token_count))
    }

    pub(super) fn append_page(&mut self, layer_index: usize, page_id: PageId) {
        self.layer_block_tables[layer_index].page_ids.push(page_id);
    }

    pub(super) fn record_appended_tokens(
        &mut self,
        layer_index: usize,
        token_count: usize,
    ) -> Result<()> {
        self.layer_block_tables[layer_index].cached_token_count = self.layer_block_tables
            [layer_index]
            .cached_token_count
            .checked_add(token_count)
            .context("cached token count overflow")?;
        Ok(())
    }

    /// Adopts already-retained canonical pages and advances prefix-publication progress. Returns
    /// the newly indexed token count and the replaced pages whose ownership must be released.
    pub(super) fn record_prefix_publication(
        &mut self,
        publication: PrefixIndexingResult,
    ) -> Result<PrefixPublicationResult> {
        let newly_indexed_token_count = publication
            .block_count
            .checked_mul(self.per_page_token_count)
            .context("newly indexed prefix token count overflow")?;
        let accumulated_indexed_token_count = self
            .accumulated_indexed_token_count
            .checked_add(newly_indexed_token_count)
            .context("accumulated request prefix token count overflow")?;
        let replaced_page_ids = publication
            .page_replacements
            .into_iter()
            .map(|replacement| {
                std::mem::replace(
                    &mut self.layer_block_tables[replacement.layer_index].page_ids
                        [replacement.block_index],
                    replacement.page_id,
                )
            })
            .collect();
        self.accumulated_indexed_token_count = accumulated_indexed_token_count;
        self.prefix_block_cursor = publication.cursor;
        Ok(PrefixPublicationResult {
            newly_indexed_token_count,
            replaced_page_ids,
        })
    }

    /// Resets all request-local cache state and returns the physical pages it no longer references.
    pub(super) fn reset(&mut self) -> Vec<PageId> {
        self.prefix_block_cursor = PrefixBlockCursor::at_root();
        self.accumulated_indexed_token_count = 0;
        self.truncate_unchecked(
            /* retained_page_count */ 0, /* target_token_count */ 0,
        )
    }

    fn truncate_unchecked(
        &mut self,
        retained_page_count: usize,
        target_token_count: usize,
    ) -> Vec<PageId> {
        let mut released_page_ids = Vec::new();
        for layer_block_table in &mut self.layer_block_tables {
            released_page_ids.extend(layer_block_table.page_ids.split_off(retained_page_count));
            layer_block_table.cached_token_count = target_token_count;
        }
        released_page_ids
    }
}
