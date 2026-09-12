use self::contiguous_cache::ContiguousKvCache;
use self::paged_cache::PagedKvCache;
use crate::model_runner::KvCacheType;
use crate::models::loaded_model::LoadedModel;
use crate::models::ModelRole;
use anyhow::Context;
use anyhow::Result;

mod active_block_tables;
mod contiguous_cache;
mod manager;
mod paged_cache;
mod physical_page_pool;
mod prefix_index;
mod utils;

pub(super) use manager::KvCacheManager;

trait LayerCache {
    fn cached_token_count(&self) -> usize;
}

/// Model-runner-owned cache variants and their request lifecycle operations.
pub(super) enum KvCacheBackend {
    Contiguous(Box<ContiguousKvCache>),
    Paged(Box<PagedKvCache>),
}

impl KvCacheBackend {
    pub(super) fn supports_multiple_active_requests(&self) -> bool {
        matches!(self, Self::Paged(_))
    }

    pub(super) fn token_capacity(&self) -> usize {
        match self {
            Self::Contiguous(cache) => cache.token_capacity(),
            Self::Paged(cache) => cache.token_capacity(),
        }
    }

    pub(super) fn layer_count(&self) -> usize {
        match self {
            Self::Contiguous(cache) => cache.layer_count(),
            Self::Paged(cache) => cache.layer_count(),
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
        KvCacheType::Contiguous => Ok(KvCacheBackend::Contiguous(Box::new(
            ContiguousKvCache::new(model.info(), model_role, model.device(), total_size_bytes)
                .with_context(|| {
                    format!("failed to allocate {model_role} model contiguous KV-cache pools")
                })?,
        ))),
        KvCacheType::Paged {
            per_page_token_count,
            enable_prefix_caching,
        } => Ok(KvCacheBackend::Paged(Box::new(
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
        ))),
    }
}
