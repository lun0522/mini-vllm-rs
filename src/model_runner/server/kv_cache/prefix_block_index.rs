use super::PageId;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PrefixBlockId(usize);

impl PrefixBlockId {
    fn next(self) -> Result<Self> {
        Ok(Self(
            self.0.checked_add(1).context("prefix block ID overflow")?,
        ))
    }
}

#[derive(Eq, Hash, PartialEq)]
struct PrefixBlockKey {
    parent_id: Option<PrefixBlockId>,
    token_ids: Box<[u32]>,
}

struct IndexedPrefixBlock {
    block_id: PrefixBlockId,
    page_ids_by_layer: Box<[PageId]>,
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct PrefixBlockMatch {
    pub(super) page_ids_by_layer: Vec<Vec<PageId>>,
}

pub(super) struct PrefixBlockIndex {
    // This implementation intentionally uses one prefix block per physical KV-cache page. The
    // two sizes could differ with a more complex mapping, but keeping them equal gives every
    // indexed block exactly one independently retainable and evictable page per model layer.
    per_block_token_count: usize,
    blocks_map: HashMap<PrefixBlockKey, IndexedPrefixBlock>,
    next_block_id: PrefixBlockId,
}

impl PrefixBlockIndex {
    pub(super) fn new(per_block_token_count: usize) -> Result<Self> {
        if per_block_token_count == 0 {
            bail!("prefix block token count must be greater than zero");
        }
        Ok(Self {
            per_block_token_count,
            blocks_map: HashMap::new(),
            next_block_id: PrefixBlockId(0),
        })
    }

    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "prefix restoration will be connected separately")
    )]
    pub(super) fn find_longest_cached_prefix(&self, input_token_ids: &[u32]) -> PrefixBlockMatch {
        let mut parent_id = None;
        let mut matched_page_ids_by_layer = Vec::<Vec<PageId>>::new();
        for token_id_block in input_token_ids.chunks_exact(self.per_block_token_count) {
            let key = PrefixBlockKey {
                parent_id,
                token_ids: token_id_block.into(),
            };
            let Some(block) = self.blocks_map.get(&key) else {
                break;
            };
            if matched_page_ids_by_layer.is_empty() {
                matched_page_ids_by_layer.resize_with(block.page_ids_by_layer.len(), Vec::new);
            }
            // Each indexed block contributes one physical page to every model layer. Append each
            // page to that layer's ordered list of pages for the matched prefix.
            for (layer_index, &block_page_id) in block.page_ids_by_layer.iter().enumerate() {
                matched_page_ids_by_layer[layer_index].push(block_page_id);
            }
            parent_id = Some(block.block_id);
        }
        PrefixBlockMatch {
            page_ids_by_layer: matched_page_ids_by_layer,
        }
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "sequence indexing will be connected after the index exists"
        )
    )]
    pub(super) fn index_cached_sequence(
        &mut self,
        sequence_token_ids: &[u32],
        cached_page_ids_by_layer: &[Vec<PageId>],
    ) -> Result<()> {
        let complete_block_count = sequence_token_ids.len() / self.per_block_token_count;
        if let Some((layer_index, cached_page_ids)) = cached_page_ids_by_layer
            .iter()
            .enumerate()
            .find(|(_, cached_page_ids)| cached_page_ids.len() < complete_block_count)
        {
            bail!(
                "cannot index sequence: complete block count {complete_block_count} exceeds \
                 cached page count {} for layer {layer_index}",
                cached_page_ids.len()
            );
        }

        let mut parent_id = None;
        for (block_index, token_id_block) in sequence_token_ids
            .chunks_exact(self.per_block_token_count)
            .enumerate()
        {
            let key = PrefixBlockKey {
                parent_id,
                token_ids: token_id_block.into(),
            };
            // Follow or create the next trie node using the current block as the edge.
            parent_id = Some(match self.blocks_map.get(&key) {
                Some(block) => block.block_id,
                None => {
                    // TODO: Remove descendant entries when prefix-block eviction removes a parent.
                    let block_id = self.next_block_id;
                    self.next_block_id = self.next_block_id.next()?;
                    self.blocks_map.insert(
                        key,
                        IndexedPrefixBlock {
                            block_id,
                            page_ids_by_layer: cached_page_ids_by_layer
                                .iter()
                                .map(|cached_page_ids| cached_page_ids[block_index])
                                .collect(),
                        },
                    );
                    block_id
                }
            });
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

    #[test]
    fn rejects_a_zero_token_block_size() {
        assert!(PrefixBlockIndex::new(0).is_err());
    }

    #[test]
    fn finds_an_exact_prefix_match_across_layers() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index.index_cached_sequence(&[1, 2, 3, 4], &pages(&[&[10, 11], &[20, 21]]))?;

        assert_eq!(
            index.find_longest_cached_prefix(&[1, 2, 3, 4]),
            PrefixBlockMatch {
                page_ids_by_layer: pages(&[&[10, 11], &[20, 21]]),
            }
        );
        Ok(())
    }

    #[test]
    fn finds_the_reusable_prefix_before_a_divergent_suffix() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index.index_cached_sequence(&[1, 2, 3, 4], &pages(&[&[10, 11]]))?;

        assert_eq!(
            index.find_longest_cached_prefix(&[1, 2, 8, 9]),
            PrefixBlockMatch {
                page_ids_by_layer: pages(&[&[10]]),
            }
        );
        Ok(())
    }

    #[test]
    fn distinguishes_identical_blocks_with_different_prefixes() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index.index_cached_sequence(&[1, 2, 7, 8], &pages(&[&[10, 11]]))?;
        index.index_cached_sequence(&[3, 4, 7, 8], &pages(&[&[20, 21]]))?;

        assert_eq!(
            index.find_longest_cached_prefix(&[3, 4, 7, 8]),
            PrefixBlockMatch {
                page_ids_by_layer: pages(&[&[20, 21]]),
            }
        );
        Ok(())
    }

    #[test]
    fn ignores_an_incomplete_final_sequence_block() -> Result<()> {
        let mut index = PrefixBlockIndex::new(2)?;
        index.index_cached_sequence(&[1, 2, 3], &pages(&[&[10, 11]]))?;

        assert_eq!(
            index.find_longest_cached_prefix(&[1, 2, 3]),
            PrefixBlockMatch {
                page_ids_by_layer: pages(&[&[10]]),
            }
        );
        assert_eq!(
            index.find_longest_cached_prefix(&[3]),
            PrefixBlockMatch {
                page_ids_by_layer: Vec::new(),
            }
        );
        Ok(())
    }
}
