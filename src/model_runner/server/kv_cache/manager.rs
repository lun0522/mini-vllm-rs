use super::contiguous_cache::RequestContiguousCacheState;
use super::paged_cache::RequestPagedCacheState;
use super::KvCacheBackend;
use crate::models::CachedKeyValue;
use crate::models::ForwardContext;
use crate::models::KvCache;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use candle_core::Tensor;
use std::collections::HashMap;

enum RequestKvCacheState {
    Contiguous(RequestContiguousCacheState),
    Paged(RequestPagedCacheState),
}

/// Owns a model's shared cache backend and its request-specific cache states.
pub struct KvCacheManager {
    backend: KvCacheBackend,
    request_states: HashMap<u64, RequestKvCacheState>,
}

impl KvCacheManager {
    pub fn new(backend: KvCacheBackend) -> Self {
        Self {
            backend,
            request_states: HashMap::new(),
        }
    }

    pub fn start_request(&mut self, request_id: u64) -> Result<()> {
        if self.request_states.contains_key(&request_id) {
            bail!("KV cache request {request_id} already exists");
        }
        let request_state = match &self.backend {
            KvCacheBackend::Contiguous(_) => {
                if !self.request_states.is_empty() {
                    bail!("contiguous KV cache already has an active request");
                }
                RequestKvCacheState::Contiguous(RequestContiguousCacheState::new(
                    self.backend.layer_count(),
                ))
            }
            KvCacheBackend::Paged(_) => {
                RequestKvCacheState::Paged(RequestPagedCacheState::new(self.backend.layer_count()))
            }
        };
        self.request_states.insert(request_id, request_state);
        Ok(())
    }

    pub fn token_capacity(&self) -> usize {
        self.backend.token_capacity()
    }

    pub fn evicted_cached_token_count(&self) -> usize {
        self.backend.evicted_cached_token_count()
    }

    /// Restores reusable prefix pages and returns the number of restored tokens.
    pub fn restore_cached_prefix(&mut self, request_id: u64, token_ids: &[u32]) -> Result<usize> {
        self.with_request_state(
            request_id,
            /* handle_contiguous */
            |_, _| Ok(0),
            /* handle_paged */
            |cache, request_state| cache.restore_cached_prefix(request_state, token_ids),
        )
    }

    pub fn truncate(&mut self, request_id: u64, target_token_count: usize) -> Result<()> {
        self.with_request_state(
            request_id,
            /* handle_contiguous */
            |cache, request_state| cache.truncate(request_state, target_token_count),
            /* handle_paged */
            |cache, request_state| cache.truncate(request_state, target_token_count),
        )
    }

    pub fn finish_request(&mut self, request_id: u64, token_ids: &[u32]) -> Result<()> {
        self.with_request_state(
            request_id,
            /* handle_contiguous */
            |_, _| Ok(()),
            /* handle_paged */
            |cache, request_state| cache.retain_completed_blocks(request_state, token_ids),
        )?;
        self.remove_request(request_id)
    }

    /// Releases a request's active cache resources and removes its state.
    pub fn remove_request(&mut self, request_id: u64) -> Result<()> {
        self.with_request_state(
            request_id,
            /* handle_contiguous */
            |_, _| Ok(()),
            /* handle_paged */
            |cache, request_state| cache.reset_request(request_state),
        )?;
        self.request_states.remove(&request_id);
        Ok(())
    }

    fn with_request_state<T>(
        &mut self,
        request_id: u64,
        handle_contiguous: impl FnOnce(
            &mut super::contiguous_cache::ContiguousKvCache,
            &mut RequestContiguousCacheState,
        ) -> Result<T>,
        handle_paged: impl FnOnce(
            &mut super::paged_cache::PagedKvCache,
            &mut RequestPagedCacheState,
        ) -> Result<T>,
    ) -> Result<T> {
        let request_state = self
            .request_states
            .get_mut(&request_id)
            .with_context(|| format!("KV cache request {request_id} does not exist"))?;
        match (&mut self.backend, request_state) {
            (KvCacheBackend::Contiguous(cache), RequestKvCacheState::Contiguous(request_state)) => {
                handle_contiguous(cache, request_state)
            }
            (KvCacheBackend::Paged(cache), RequestKvCacheState::Paged(request_state)) => {
                handle_paged(cache, request_state)
            }
            _ => bail!("KV cache backend and request state do not match"),
        }
    }
}

impl KvCache for KvCacheManager {
    fn append(
        &mut self,
        context: &ForwardContext,
        layer_index: usize,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<CachedKeyValue> {
        self.with_request_state(
            context.request_id,
            /* handle_contiguous */
            |cache, request_state| {
                cache.append(
                    request_state,
                    layer_index,
                    context.start_position,
                    key,
                    value,
                )
            },
            /* handle_paged */
            |cache, request_state| {
                cache.append(
                    request_state,
                    layer_index,
                    context.start_position,
                    key,
                    value,
                )
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelInfo;
    use crate::models::ModelRole;
    use candle_core::DType;
    use candle_core::Device;

    fn paged_manager() -> Result<KvCacheManager> {
        let model_info = ModelInfo {
            layer_count: 1,
            kv_head_count: 1,
            head_dimension: 1,
            activation_dtype: DType::U32,
        };
        let cache = super::super::paged_cache::PagedKvCache::new(
            &model_info,
            ModelRole::Target,
            &Device::Cpu,
            2,
            true,
            16,
        )?;
        Ok(KvCacheManager::new(KvCacheBackend::Paged(Box::new(cache))))
    }

    fn contiguous_manager() -> Result<KvCacheManager> {
        let model_info = ModelInfo {
            layer_count: 1,
            kv_head_count: 1,
            head_dimension: 1,
            activation_dtype: DType::U32,
        };
        let cache = super::super::contiguous_cache::ContiguousKvCache::new(
            &model_info,
            ModelRole::Target,
            &Device::Cpu,
            16,
        )?;
        Ok(KvCacheManager::new(KvCacheBackend::Contiguous(Box::new(
            cache,
        ))))
    }

    #[test]
    fn stores_independent_paged_states_by_request_id() -> Result<()> {
        let mut manager = paged_manager()?;

        manager.start_request(1)?;
        manager.start_request(2)?;

        assert_eq!(manager.request_states.len(), 2);
        manager.remove_request(1)?;
        assert!(manager.request_states.contains_key(&2));
        manager.remove_request(2)?;
        assert!(manager.request_states.is_empty());
        Ok(())
    }

    #[test]
    fn rejects_duplicate_request_ids() -> Result<()> {
        let mut manager = paged_manager()?;
        manager.start_request(1)?;

        let error = manager.start_request(1).unwrap_err().to_string();

        assert_eq!(error, "KV cache request 1 already exists");
        Ok(())
    }

    #[test]
    fn contiguous_cache_rejects_a_second_active_request() -> Result<()> {
        let mut manager = contiguous_manager()?;
        manager.start_request(1)?;

        let error = manager.start_request(2).unwrap_err().to_string();

        assert_eq!(error, "contiguous KV cache already has an active request");
        manager.remove_request(1)?;
        manager.start_request(2)?;
        Ok(())
    }

    #[test]
    fn rejects_unknown_request_ids() -> Result<()> {
        let mut manager = paged_manager()?;

        let error = manager.remove_request(1).unwrap_err().to_string();

        assert_eq!(error, "KV cache request 1 does not exist");
        Ok(())
    }
}
