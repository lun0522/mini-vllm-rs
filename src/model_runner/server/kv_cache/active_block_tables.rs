use super::physical_page_pool::PageId;
use super::utils::validate_truncation;
use super::LayerCache;
use anyhow::Result;

/// Maps one model layer's logical token order to physical KV-cache pages.
#[derive(Default)]
pub(super) struct LayerBlockTable {
    pub(super) page_ids: Vec<PageId>,
    pub(super) cached_token_count: usize,
}

impl LayerCache for LayerBlockTable {
    fn cached_token_count(&self) -> usize {
        self.cached_token_count
    }
}

/// Holds the virtual block tables and token counts for the sequence being processed.
///
/// "Active" means these tables refer to physical pages attached to the current sequence. They do
/// not own tensor storage or manage page reference counts. Resetting the tables returns the page
/// IDs whose active references the paged-cache orchestrator must release.
pub(super) struct ActiveBlockTables {
    pub(super) layer_block_tables: Vec<LayerBlockTable>,
}

impl ActiveBlockTables {
    pub(super) fn new(layer_count: usize) -> Self {
        Self {
            layer_block_tables: (0..layer_count)
                .map(|_| LayerBlockTable::default())
                .collect(),
        }
    }

    pub(super) fn is_populated(&self) -> bool {
        self.layer_block_tables.iter().any(|layer_block_table| {
            layer_block_table.cached_token_count != 0 || !layer_block_table.page_ids.is_empty()
        })
    }

    pub(super) fn cached_token_count(&self) -> usize {
        self.layer_block_tables
            .first()
            .map_or(0, LayerCache::cached_token_count)
    }

    pub(super) fn layer_block_table(&self, layer_index: usize) -> Option<&LayerBlockTable> {
        self.layer_block_tables.get(layer_index)
    }

    pub(super) fn layer_cached_token_count(&self, layer_index: usize) -> Option<usize> {
        self.layer_block_table(layer_index)
            .map(LayerCache::cached_token_count)
    }

    pub(super) fn validate_truncation(&self, target_token_count: usize) -> Result<()> {
        validate_truncation(&self.layer_block_tables, target_token_count)
    }

    pub(super) fn append_page(&mut self, layer_index: usize, page_id: PageId) {
        self.layer_block_tables[layer_index].page_ids.push(page_id);
    }

    pub(super) fn last_page_id(&self, layer_index: usize) -> Option<PageId> {
        self.layer_block_tables[layer_index]
            .page_ids
            .last()
            .copied()
    }

    pub(super) fn record_appended_tokens(&mut self, layer_index: usize, token_count: usize) {
        self.layer_block_tables[layer_index].cached_token_count += token_count;
    }

    pub(super) fn page_ids_by_layer(&self) -> Vec<Vec<PageId>> {
        self.layer_block_tables
            .iter()
            .map(|layer_block_table| layer_block_table.page_ids.clone())
            .collect()
    }

    pub(super) fn attach_cached_prefix(
        &mut self,
        page_ids_by_layer: Vec<Vec<PageId>>,
        matched_token_count: usize,
    ) {
        for (layer_block_table, page_ids) in
            self.layer_block_tables.iter_mut().zip(page_ids_by_layer)
        {
            layer_block_table.page_ids = page_ids;
            layer_block_table.cached_token_count = matched_token_count;
        }
    }

    /// Truncates every virtual table and returns the physical pages it no longer references.
    pub(super) fn truncate(
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

    /// Clears every virtual table and returns all physical pages it referenced.
    pub(super) fn reset(&mut self) -> Vec<PageId> {
        self.truncate(
            /* retained_page_count */ 0, /* target_token_count */ 0,
        )
    }
}
