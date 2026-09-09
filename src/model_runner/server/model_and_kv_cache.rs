use crate::models::loaded_model::LoadedModel;
use crate::proto::model_runner::ModelMetadata;
use anyhow::Result;
use candle_core::Tensor;
use std::cell::RefCell;

use super::kv_cache::KvCacheBackend;

/// Couples one loaded model with the KV cache mutated by its forward passes.
pub(super) struct ModelAndKvCache {
    model: RefCell<LoadedModel>,
    kv_cache: RefCell<KvCacheBackend>,
}

impl ModelAndKvCache {
    pub(super) fn new(model: LoadedModel, kv_cache: KvCacheBackend) -> Self {
        Self {
            model: RefCell::new(model),
            kv_cache: RefCell::new(kv_cache),
        }
    }

    pub(super) fn forward(
        &self,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache = self.kv_cache.borrow_mut();
        model.model().forward(input, start_position, &mut *kv_cache)
    }

    pub(super) fn model_metadata(&self) -> ModelMetadata {
        self.model.borrow().metadata()
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.kv_cache.borrow().token_capacity()
    }

    pub(super) fn create_input_tensor(&self, token_ids: &[u32]) -> candle_core::Result<Tensor> {
        Tensor::new(token_ids, self.model.borrow().device())?.unsqueeze(0)
    }

    pub(super) fn forward_for_speculative_verification(
        &self,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let mut model = self.model.borrow_mut();
        let mut kv_cache = self.kv_cache.borrow_mut();
        model
            .model()
            .forward_for_speculative_verification(input, start_position, &mut *kv_cache)
    }

    /// Restores reusable prefix pages and returns the prefill start position.
    pub(super) fn restore_cached_prefix(&self, token_ids: &[u32]) -> Result<usize> {
        self.kv_cache.borrow_mut().restore_cached_prefix(token_ids)
    }

    pub(super) fn evicted_cached_token_count(&self) -> usize {
        self.kv_cache.borrow().evicted_cached_token_count()
    }

    pub(super) fn truncate_cache(&self, target_token_count: usize) -> Result<()> {
        self.kv_cache.borrow_mut().truncate(target_token_count)
    }

    pub(super) fn clear_cache(&self) -> Result<()> {
        self.kv_cache.borrow_mut().clear()
    }

    pub(super) fn finish_request(&self, token_ids: &[u32]) -> Result<()> {
        self.kv_cache.borrow_mut().finish_request(token_ids)
    }
}
