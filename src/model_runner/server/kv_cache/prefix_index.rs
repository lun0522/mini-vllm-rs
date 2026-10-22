use super::physical_page_pool::PageId;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PrefixBlockId(usize);

impl PrefixBlockId {
    fn next(&mut self) -> Result<Self> {
        let current = *self;
        self.0 = self.0.checked_add(1).context("prefix block ID overflow")?;
        Ok(current)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct AccessTimestamp(usize);

impl AccessTimestamp {
    fn begin_access(&mut self) -> Result<Self> {
        let current = *self;
        self.0 = self
            .0
            .checked_add(1)
            .context("prefix block access timestamp overflow")?;
        Ok(current)
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct PrefixBlockKey {
    parent_id: Option<PrefixBlockId>,
    token_ids: Box<[u32]>,
}

struct IndexedPrefixBlock {
    block_id: PrefixBlockId,
    parent_key: Option<PrefixBlockKey>,
    page_ids_by_layer: Box<[PageId]>,
    child_count: usize,
    last_access_timestamp: RefCell<AccessTimestamp>,
}

impl IndexedPrefixBlock {
    fn update_access_timestamp(&self, timestamp: AccessTimestamp) {
        *self.last_access_timestamp.borrow_mut() = timestamp;
    }
}

/// Immutable, token-only view of a prefix index for cache-affinity routing.
#[derive(Debug, Eq, PartialEq)]
pub struct PrefixCacheAffinitySnapshot {
    per_block_token_count: usize,
    blocks_map: HashMap<PrefixBlockKey, PrefixBlockId>,
}

impl PrefixCacheAffinitySnapshot {
    pub fn find_longest_prefix_token_count(&self, input_token_ids: &[u32]) -> usize {
        let mut parent_id = None;
        let mut matched_block_count = 0usize;
        for token_ids in input_token_ids.chunks_exact(self.per_block_token_count) {
            let key = PrefixBlockKey {
                parent_id,
                token_ids: token_ids.into(),
            };
            let Some(&block_id) = self.blocks_map.get(&key) else {
                break;
            };
            parent_id = Some(block_id);
            matched_block_count += 1;
        }
        matched_block_count
            .checked_mul(self.per_block_token_count)
            .expect("matched prefix token count must fit within cache capacity")
    }
}

pub(super) struct PrefixBlockMatch {
    pub(super) page_ids_by_layer: Vec<Vec<PageId>>,
    pub(super) cursor: PrefixBlockCursor,
}

#[derive(Clone, Debug)]
pub(super) struct PrefixBlockCursor {
    block_id: Option<PrefixBlockId>,
    block_key: Option<PrefixBlockKey>,
    matched_token_count: usize,
}

impl PrefixBlockCursor {
    pub(super) fn at_root() -> Self {
        Self {
            block_id: None,
            block_key: None,
            matched_token_count: 0,
        }
    }

    pub(super) fn matched_token_count(&self) -> usize {
        self.matched_token_count
    }
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct EvictedPrefixBlock {
    pub(super) page_ids_by_layer: Box<[PageId]>,
}

/// Replaces one request block-table entry with an already indexed canonical page when publication
/// finds a duplicate prefix block. The caller retains the canonical page before swapping the entry
/// and releasing the redundant page, so request ownership protects the publication cursor's block
/// from eviction and normal request cleanup releases that ownership.
#[derive(Debug)]
pub(super) struct PrefixPageReplacement {
    pub(super) layer_index: usize,
    pub(super) block_index: usize,
    pub(super) page_id: PageId,
}

#[derive(Debug)]
pub(super) struct PrefixIndexingResult {
    pub(super) newly_indexed_page_ids: Vec<PageId>,
    pub(super) page_replacements: Vec<PrefixPageReplacement>,
    pub(super) block_count: usize,
    pub(super) cursor: PrefixBlockCursor,
}

pub(super) struct PrefixBlockIndex {
    // This implementation intentionally uses one prefix block per physical KV-cache page. The
    // two sizes could differ with a more complex mapping, but keeping them equal gives every
    // indexed block exactly one independently retainable and evictable page per model layer.
    per_block_token_count: usize,
    blocks_map: HashMap<PrefixBlockKey, IndexedPrefixBlock>,
    leaf_block_keys: HashSet<PrefixBlockKey>,
    next_block_id: PrefixBlockId,
    current_timestamp: RefCell<AccessTimestamp>,
}

impl PrefixBlockIndex {
    pub(super) fn new(per_block_token_count: usize) -> Result<Self> {
        ensure!(
            per_block_token_count > 0,
            "prefix block token count must be greater than zero"
        );
        Ok(Self {
            per_block_token_count,
            blocks_map: HashMap::new(),
            leaf_block_keys: HashSet::new(),
            next_block_id: PrefixBlockId(0),
            current_timestamp: RefCell::new(AccessTimestamp(0)),
        })
    }

    pub(super) fn block_count(&self) -> usize {
        self.blocks_map.len()
    }

    pub(super) fn create_cache_affinity_snapshot(&self) -> PrefixCacheAffinitySnapshot {
        PrefixCacheAffinitySnapshot {
            per_block_token_count: self.per_block_token_count,
            blocks_map: self
                .blocks_map
                .iter()
                .map(|(key, block)| (key.clone(), block.block_id))
                .collect(),
        }
    }

    pub(super) fn find_longest_cached_prefix(
        &self,
        input_token_ids: &[u32],
    ) -> Result<PrefixBlockMatch> {
        let current_timestamp = self.current_timestamp.borrow_mut().begin_access()?;
        let mut cursor = PrefixBlockCursor::at_root();
        let mut matched_page_ids_by_layer = Vec::<Vec<PageId>>::new();
        for token_id_block in input_token_ids.chunks_exact(self.per_block_token_count) {
            let key = PrefixBlockKey {
                parent_id: cursor.block_id,
                token_ids: token_id_block.into(),
            };
            let Some(block) = self.blocks_map.get(&key) else {
                break;
            };
            block.update_access_timestamp(current_timestamp);
            if matched_page_ids_by_layer.is_empty() {
                matched_page_ids_by_layer.resize_with(block.page_ids_by_layer.len(), Vec::new);
            }
            // Each indexed block contributes one physical page to every model layer. Append each
            // page to that layer's ordered list of pages for the matched prefix.
            for (layer_index, &block_page_id) in block.page_ids_by_layer.iter().enumerate() {
                matched_page_ids_by_layer[layer_index].push(block_page_id);
            }
            cursor = PrefixBlockCursor {
                block_id: Some(block.block_id),
                block_key: Some(key),
                matched_token_count: matched_page_ids_by_layer[0]
                    .len()
                    .checked_mul(self.per_block_token_count)
                    .context("matched prefix token count overflow")?,
            };
        }
        Ok(PrefixBlockMatch {
            page_ids_by_layer: matched_page_ids_by_layer,
            cursor,
        })
    }

    pub(super) fn index_cached_sequence<'a>(
        &mut self,
        cursor: &PrefixBlockCursor,
        sequence_token_ids: &[u32],
        cached_page_ids_by_layer: impl Clone + Iterator<Item = &'a [PageId]>,
    ) -> Result<PrefixIndexingResult> {
        let complete_block_count = sequence_token_ids.len() / self.per_block_token_count;
        Self::validate_cached_page_counts(complete_block_count, cached_page_ids_by_layer.clone())?;
        let current_timestamp = self.current_timestamp.borrow_mut().begin_access()?;

        let starting_block_index = cursor.matched_token_count / self.per_block_token_count;
        ensure!(
            starting_block_index <= complete_block_count,
            "restored prefix is longer than the completed cached sequence"
        );
        let mut parent_id = cursor.block_id;
        let mut previous_block_key = cursor.block_key.clone();
        let mut is_appending_new_branch = false;
        let mut newly_indexed_page_ids = Vec::new();
        let mut page_replacements = Vec::new();
        let mut newly_indexed_block_count = 0usize;
        for (block_index, token_id_block) in sequence_token_ids
            .chunks_exact(self.per_block_token_count)
            .enumerate()
            .skip(starting_block_index)
        {
            let key = PrefixBlockKey {
                parent_id,
                token_ids: token_id_block.into(),
            };

            if !is_appending_new_branch {
                if let Some(block) = self.blocks_map.get(&key) {
                    block.update_access_timestamp(current_timestamp);
                    // The publication cursor will depend on this canonical block. Adopt its
                    // pages in the request's tables so their ownership also protects the cursor.
                    for (layer_index, (&page_id, cached_page_ids)) in block
                        .page_ids_by_layer
                        .iter()
                        .zip(cached_page_ids_by_layer.clone())
                        .enumerate()
                    {
                        if page_id != cached_page_ids[block_index] {
                            page_replacements.push(PrefixPageReplacement {
                                layer_index,
                                block_index,
                                page_id,
                            });
                        }
                    }
                    parent_id = Some(block.block_id);
                    previous_block_key = Some(key);
                    // Continue walking the cached prefix after finding this existing block.
                    continue;
                }

                is_appending_new_branch = true;
                // Only the existing parent at the branch point needs an additional lookup. Every
                // subsequent parent is part of the new branch and receives its child count when
                // it is inserted.
                if let Some(previous_block_key) = previous_block_key.as_ref() {
                    self.increment_block_child_count(previous_block_key)?;
                }
            }

            let block_id = self.next_block_id.next()?;
            let has_child = block_index + 1 < complete_block_count;
            let page_ids_by_layer: Box<_> = cached_page_ids_by_layer
                .clone()
                .map(|cached_page_ids| cached_page_ids[block_index])
                .collect();
            newly_indexed_page_ids.extend(page_ids_by_layer.iter().copied());
            newly_indexed_block_count = newly_indexed_block_count
                .checked_add(1)
                .context("newly indexed prefix block count overflow")?;
            self.blocks_map.insert(
                key.clone(),
                IndexedPrefixBlock {
                    block_id,
                    parent_key: previous_block_key,
                    page_ids_by_layer,
                    child_count: usize::from(has_child),
                    last_access_timestamp: RefCell::new(current_timestamp),
                },
            );
            if !has_child {
                self.leaf_block_keys.insert(key.clone());
            }
            parent_id = Some(block_id);
            previous_block_key = Some(key);
        }
        let indexed_token_count = complete_block_count
            .checked_mul(self.per_block_token_count)
            .context("indexed prefix token count overflow")?;
        Ok(PrefixIndexingResult {
            newly_indexed_page_ids,
            page_replacements,
            block_count: newly_indexed_block_count,
            cursor: PrefixBlockCursor {
                block_id: parent_id,
                block_key: previous_block_key,
                matched_token_count: indexed_token_count,
            },
        })
    }

    /// Removes the least recently used eligible leaf and returns its pages without modifying them.
    /// Returns `Ok(None)` when no leaf is eligible for eviction.
    pub(super) fn evict_least_recently_used_leaf(
        &mut self,
        is_eligible: impl Fn(&[PageId]) -> bool,
    ) -> Result<Option<EvictedPrefixBlock>> {
        let Some(key) = self.find_least_recently_used_leaf_key(is_eligible) else {
            return Ok(None);
        };
        self.leaf_block_keys.remove(&key);
        let evicted_block = self
            .blocks_map
            .remove(&key)
            .context("selected leaf block is missing from the prefix block index")?;
        if let Some(parent_key) = evicted_block.parent_key.as_ref() {
            self.decrement_block_child_count(parent_key)?;
        }
        Ok(Some(EvictedPrefixBlock {
            page_ids_by_layer: evicted_block.page_ids_by_layer,
        }))
    }

    fn find_least_recently_used_leaf_key(
        &self,
        is_eligible: impl Fn(&[PageId]) -> bool,
    ) -> Option<PrefixBlockKey> {
        self.leaf_block_keys
            .iter()
            .filter(|key| {
                let block = self
                    .blocks_map
                    .get(*key)
                    .expect("leaf block must exist in the prefix block index");
                is_eligible(&block.page_ids_by_layer)
            })
            .min_by_key(|key| {
                let block = self
                    .blocks_map
                    .get(*key)
                    .expect("leaf block must exist in the prefix block index");
                *block.last_access_timestamp.borrow()
            })
            .cloned()
    }

    fn validate_cached_page_counts<'a>(
        complete_block_count: usize,
        cached_page_ids_by_layer: impl Iterator<Item = &'a [PageId]>,
    ) -> Result<()> {
        if let Some((layer_index, cached_page_ids)) = cached_page_ids_by_layer
            .enumerate()
            .find(|(_, cached_page_ids)| cached_page_ids.len() < complete_block_count)
        {
            bail!(
                "cannot index sequence: complete block count {complete_block_count} exceeds \
                 cached page count {} for layer {layer_index}",
                cached_page_ids.len()
            );
        }
        Ok(())
    }

    fn increment_block_child_count(&mut self, block_key: &PrefixBlockKey) -> Result<()> {
        self.leaf_block_keys.remove(block_key);
        let block = self
            .blocks_map
            .get_mut(block_key)
            .context("prefix block parent is missing from the index")?;
        block.child_count = block
            .child_count
            .checked_add(1)
            .context("prefix block child count overflow")?;
        Ok(())
    }

    fn decrement_block_child_count(&mut self, block_key: &PrefixBlockKey) -> Result<()> {
        let block = self
            .blocks_map
            .get_mut(block_key)
            .context("leaf parent is missing from the prefix block index")?;
        block.child_count = block
            .child_count
            .checked_sub(1)
            .context("leaf parent does not reference the evicted child")?;
        if block.child_count == 0 {
            self.leaf_block_keys.insert(block_key.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(ids: &[&[usize]]) -> Vec<Vec<PageId>> {
        ids.iter()
            .map(|layer| layer.iter().copied().map(PageId).collect())
            .collect()
    }

    fn index_pages(
        index: &mut PrefixBlockIndex,
        cursor: &PrefixBlockCursor,
        token_ids: &[u32],
        page_ids_by_layer: Vec<Vec<PageId>>,
    ) -> Result<PrefixIndexingResult> {
        index.index_cached_sequence(
            cursor,
            token_ids,
            page_ids_by_layer.iter().map(Vec::as_slice),
        )
    }

    #[test]
    fn rejects_a_zero_token_block_size() {
        assert!(PrefixBlockIndex::new(0).is_err());
    }

    #[test]
    fn cache_affinity_snapshot_finds_the_longest_complete_cached_prefix() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 5, 6],
            pages(&[&[10, 12]]),
        )?;

        let snapshot = index.create_cache_affinity_snapshot();

        assert_eq!(snapshot.find_longest_prefix_token_count(&[1, 2, 3, 4]), 4);
        assert_eq!(snapshot.find_longest_prefix_token_count(&[1, 2, 5, 9]), 2);
        assert_eq!(snapshot.find_longest_prefix_token_count(&[1, 9, 3, 4]), 0);
        assert_eq!(snapshot.find_longest_prefix_token_count(&[1, 2, 3]), 2);
        Ok(())
    }

    #[test]
    fn cache_affinity_snapshot_is_unchanged_when_the_index_changes() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2],
            pages(&[&[10]]),
        )?;
        let old_snapshot = index.create_cache_affinity_snapshot();

        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[3, 4],
            pages(&[&[11]]),
        )?;
        let new_snapshot = index.create_cache_affinity_snapshot();

        assert_eq!(old_snapshot.find_longest_prefix_token_count(&[3, 4]), 0);
        assert_eq!(new_snapshot.find_longest_prefix_token_count(&[3, 4]), 2);
        Ok(())
    }

    #[test]
    fn rejects_fewer_cached_pages_than_complete_blocks() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;

        let error = index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11], &[20]]),
        )
        .unwrap_err();

        assert_eq!(
            error.to_string(),
            "cannot index sequence: complete block count 2 exceeds cached page count 1 for layer 1"
        );
        Ok(())
    }

    #[test]
    fn handles_an_empty_index() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;

        assert_eq!(
            index.find_longest_cached_prefix(&[1, 2])?.page_ids_by_layer,
            Vec::<Vec<PageId>>::new()
        );
        assert_eq!(index.evict_least_recently_used_leaf(|_| true)?, None);
        Ok(())
    }

    #[test]
    fn finds_an_exact_prefix_match_across_layers() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11], &[20, 21]]),
        )?;
        assert_eq!(index.leaf_block_keys.len(), 1);

        assert_eq!(
            index
                .find_longest_cached_prefix(&[1, 2, 3, 4])?
                .page_ids_by_layer,
            pages(&[&[10, 11], &[20, 21]])
        );
        Ok(())
    }

    #[test]
    fn finds_the_reusable_prefix_before_a_divergent_suffix() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;

        assert_eq!(
            index
                .find_longest_cached_prefix(&[1, 2, 8, 9])?
                .page_ids_by_layer,
            pages(&[&[10]])
        );
        Ok(())
    }

    #[test]
    fn distinguishes_identical_blocks_with_different_prefixes() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 7, 8],
            pages(&[&[10, 11]]),
        )?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[3, 4, 7, 8],
            pages(&[&[20, 21]]),
        )?;

        assert_eq!(
            index
                .find_longest_cached_prefix(&[3, 4, 7, 8])?
                .page_ids_by_layer,
            pages(&[&[20, 21]])
        );
        Ok(())
    }

    #[test]
    fn ignores_an_incomplete_final_sequence_block() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3],
            pages(&[&[10, 11]]),
        )?;

        assert_eq!(
            index
                .find_longest_cached_prefix(&[1, 2, 3])?
                .page_ids_by_layer,
            pages(&[&[10]])
        );
        assert_eq!(
            index.find_longest_cached_prefix(&[3])?.page_ids_by_layer,
            Vec::<Vec<PageId>>::new()
        );
        Ok(())
    }

    #[test]
    fn reindexes_an_existing_sequence_without_duplicate_blocks() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        let first_indexing = index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;
        assert_eq!(first_indexing.block_count, 2);

        let second_indexing = index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[20, 21]]),
        )?;
        assert_eq!(second_indexing.block_count, 0);

        assert_eq!(index.blocks_map.len(), 2);
        assert_eq!(index.next_block_id, PrefixBlockId(2));
        assert_eq!(index.leaf_block_keys.len(), 1);
        assert_eq!(
            index
                .find_longest_cached_prefix(&[1, 2, 3, 4])?
                .page_ids_by_layer,
            pages(&[&[10, 11]])
        );
        Ok(())
    }

    #[test]
    fn evicts_a_leaf_before_its_parent() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11], &[20, 21]]),
        )?;

        assert_eq!(
            index.evict_least_recently_used_leaf(|_| true)?,
            Some(EvictedPrefixBlock {
                page_ids_by_layer: Box::new([PageId(11), PageId(21)]),
            })
        );
        assert_eq!(index.leaf_block_keys.len(), 1);
        assert_eq!(
            index.evict_least_recently_used_leaf(|_| true)?,
            Some(EvictedPrefixBlock {
                page_ids_by_layer: Box::new([PageId(10), PageId(20)]),
            })
        );
        assert!(index.leaf_block_keys.is_empty());
        assert_eq!(index.evict_least_recently_used_leaf(|_| true)?, None);
        Ok(())
    }

    #[test]
    fn does_not_evict_an_ineligible_leaf() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;
        assert_eq!(index.evict_least_recently_used_leaf(|_| false)?, None);
        assert_eq!(index.blocks_map.len(), 2);
        Ok(())
    }

    #[test]
    fn stops_eviction_when_an_active_internal_block_becomes_a_leaf() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 5, 6],
            pages(&[&[10, 12]]),
        )?;
        assert!(index
            .evict_least_recently_used_leaf(|page_ids| page_ids != [PageId(10)])?
            .is_some());
        assert!(index
            .evict_least_recently_used_leaf(|page_ids| page_ids != [PageId(10)])?
            .is_some());
        assert_eq!(
            index.evict_least_recently_used_leaf(|page_ids| page_ids != [PageId(10)])?,
            None
        );
        assert_eq!(index.blocks_map.len(), 1);
        Ok(())
    }

    #[test]
    fn evicts_the_least_recently_used_leaf() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 5, 6],
            pages(&[&[10, 12]]),
        )?;
        assert_eq!(index.leaf_block_keys.len(), 2);

        // Touch the first branch, making the second branch the least recently used leaf.
        index.find_longest_cached_prefix(&[1, 2, 3, 4])?;

        assert_eq!(
            index.evict_least_recently_used_leaf(|_| true)?,
            Some(EvictedPrefixBlock {
                page_ids_by_layer: Box::new([PageId(12)]),
            })
        );
        assert_eq!(index.leaf_block_keys.len(), 1);
        assert_eq!(
            index
                .find_longest_cached_prefix(&[1, 2, 5, 6])?
                .page_ids_by_layer,
            pages(&[&[10]])
        );
        Ok(())
    }

    #[test]
    fn promotes_a_parent_only_after_evicting_its_final_child() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 5, 6],
            pages(&[&[10, 12]]),
        )?;

        assert_eq!(
            index.evict_least_recently_used_leaf(|_| true)?,
            Some(EvictedPrefixBlock {
                page_ids_by_layer: Box::new([PageId(11)]),
            })
        );
        assert_eq!(index.leaf_block_keys.len(), 1);
        assert_eq!(
            index.evict_least_recently_used_leaf(|_| true)?,
            Some(EvictedPrefixBlock {
                page_ids_by_layer: Box::new([PageId(12)]),
            })
        );
        assert_eq!(index.leaf_block_keys.len(), 1);
        assert_eq!(
            index.evict_least_recently_used_leaf(|_| true)?,
            Some(EvictedPrefixBlock {
                page_ids_by_layer: Box::new([PageId(10)]),
            })
        );
        Ok(())
    }

    #[test]
    fn reindexing_updates_lru_access_timestamps() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;
        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 5, 6],
            pages(&[&[10, 12]]),
        )?;

        index_pages(
            &mut index,
            &PrefixBlockCursor::at_root(),
            &[1, 2, 3, 4],
            pages(&[&[10, 11]]),
        )?;

        assert_eq!(
            index.evict_least_recently_used_leaf(|_| true)?,
            Some(EvictedPrefixBlock {
                page_ids_by_layer: Box::new([PageId(12)]),
            })
        );
        Ok(())
    }
}
