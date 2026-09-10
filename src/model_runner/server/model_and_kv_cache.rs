use crate::models::loaded_model::LoadedModel;
use crate::proto::model_runner::ModelMetadata;
use anyhow::Result;
use candle_core::Tensor;
use std::cell::RefCell;

use super::kv_cache::KvCacheBackend;
use super::kv_cache::KvCacheManager;

/// Couples one loaded model with the KV cache mutated by its forward passes.
pub(super) struct ModelAndKvCache {
    model: RefCell<LoadedModel>,
    kv_cache_manager: RefCell<KvCacheManager>,
}

impl ModelAndKvCache {
    pub(super) fn new(model: LoadedModel, kv_cache: KvCacheBackend) -> Self {
        Self {
            model: RefCell::new(model),
            kv_cache_manager: RefCell::new(KvCacheManager::new(kv_cache)),
        }
    }

    pub(super) fn start_request(&self, request_id: u64) -> Result<()> {
        self.kv_cache_manager.borrow_mut().start_request(request_id)
    }

    pub(super) fn forward(
        &self,
        request_id: u64,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache_manager = self.kv_cache_manager.borrow_mut();
        let mut kv_cache = kv_cache_manager
            .request_cache(request_id)
            .map_err(candle_core::Error::wrap)?;
        model.model().forward(input, start_position, &mut kv_cache)
    }

    pub(super) fn model_metadata(&self) -> ModelMetadata {
        self.model.borrow().metadata()
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.kv_cache_manager.borrow().token_capacity()
    }

    pub(super) fn create_input_tensor(&self, token_ids: &[u32]) -> candle_core::Result<Tensor> {
        Tensor::new(token_ids, self.model.borrow().device())?.unsqueeze(0)
    }

    pub(super) fn forward_for_speculative_verification(
        &self,
        request_id: u64,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache_manager = self.kv_cache_manager.borrow_mut();
        let mut kv_cache = kv_cache_manager
            .request_cache(request_id)
            .map_err(candle_core::Error::wrap)?;
        model
            .model()
            .forward_for_speculative_verification(input, start_position, &mut kv_cache)
    }

    /// Restores reusable prefix pages and returns the prefill start position.
    pub(super) fn restore_cached_prefix(
        &self,
        request_id: u64,
        token_ids: &[u32],
    ) -> Result<usize> {
        self.kv_cache_manager
            .borrow_mut()
            .restore_cached_prefix(request_id, token_ids)
    }

    pub(super) fn evicted_cached_token_count(&self) -> usize {
        self.kv_cache_manager.borrow().evicted_cached_token_count()
    }

    pub(super) fn truncate_cache(&self, request_id: u64, target_token_count: usize) -> Result<()> {
        self.kv_cache_manager
            .borrow_mut()
            .truncate(request_id, target_token_count)
    }

    pub(super) fn abort_request(&self, request_id: u64) -> Result<()> {
        self.kv_cache_manager
            .borrow_mut()
            .remove_request(request_id)
    }

    pub(super) fn finish_request(&self, request_id: u64, token_ids: &[u32]) -> Result<()> {
        self.kv_cache_manager
            .borrow_mut()
            .finish_request(request_id, token_ids)
    }
}
