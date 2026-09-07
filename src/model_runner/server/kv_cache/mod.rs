use self::contiguous_cache::ContiguousKvCache;
use self::paged_cache::PagedKvCache;
use crate::model_runner::KvCacheType;
use crate::models::loaded_model::LoadedModel;
use crate::models::CachedKeyValue;
use crate::models::KvCache;
use crate::models::ModelRole;
use anyhow::Context;
use anyhow::Result;
use candle_core::Tensor;

mod contiguous_cache;
mod paged_cache;
mod prefix_index;
mod utils;

trait LayerCache {
    fn cached_token_count(&self) -> usize;
}

/// Model-runner-owned cache variants and their request lifecycle operations.
pub(super) enum KvCacheBackend {
    Contiguous(ContiguousKvCache),
    Paged(PagedKvCache),
}

impl KvCacheBackend {
    pub(super) fn token_capacity(&self) -> usize {
        match self {
            Self::Contiguous(cache) => cache.token_capacity(),
            Self::Paged(cache) => cache.token_capacity(),
        }
    }

    /// Restores reusable prefix pages and returns the number of restored tokens.
    pub(super) fn restore_cached_prefix(&mut self, token_ids: &[u32]) -> Result<usize> {
        match self {
            Self::Contiguous(_) => Ok(0),
            Self::Paged(cache) => cache.restore_cached_prefix(token_ids),
        }
    }

    pub(super) fn truncate(&mut self, target_token_count: usize) -> Result<()> {
        match self {
            Self::Contiguous(cache) => cache.truncate(target_token_count),
            Self::Paged(cache) => cache.truncate(target_token_count),
        }
    }

    pub(super) fn clear(&mut self) -> Result<()> {
        match self {
            Self::Contiguous(cache) => {
                cache.clear();
                Ok(())
            }
            Self::Paged(cache) => cache.reset_active_block_tables(),
        }
    }

    pub(super) fn finish_request(&mut self, token_ids: &[u32]) -> Result<()> {
        if let Self::Paged(cache) = self {
            cache.retain_completed_blocks(token_ids)?;
        }
        self.clear()
    }
}

impl KvCache for KvCacheBackend {
    fn append(
        &mut self,
        layer_index: usize,
        start_position: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        match self {
            Self::Contiguous(cache) => cache.append(layer_index, start_position, key, value),
            Self::Paged(cache) => cache.append(layer_index, start_position, key, value),
        }
    }
}

pub(super) fn create_kv_cache(
    kv_cache_type: KvCacheType,
    model: &LoadedModel,
    model_role: ModelRole,
    total_size_bytes: usize,
) -> Result<KvCacheBackend> {
    match kv_cache_type {
        KvCacheType::Contiguous => Ok(KvCacheBackend::Contiguous(
            ContiguousKvCache::new(model.info(), model_role, model.device(), total_size_bytes)
                .with_context(|| {
                    format!("failed to allocate {model_role} model contiguous KV-cache pools")
                })?,
        )),
        KvCacheType::Paged {
            per_page_token_count,
            enable_prefix_caching,
        } => Ok(KvCacheBackend::Paged(
            PagedKvCache::new(
                model.info(),
                model_role,
                model.device(),
                per_page_token_count,
                enable_prefix_caching,
                total_size_bytes,
            )
            .with_context(|| {
                format!("failed to allocate {model_role} model paged KV-cache pools")
            })?,
        )),
    }
}
