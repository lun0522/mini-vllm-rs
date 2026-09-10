use crate::models::loaded_model::LoadedModel;
use crate::models::ForwardContext;
use crate::proto::model_runner::ModelMetadata;
use anyhow::Result;
use candle_core::Tensor;

use super::kv_cache::KvCacheBackend;
use super::kv_cache::KvCacheManager;

/// Owns one loaded model and the runtime KV-cache state used by that model instance.
pub(super) struct ModelInstance {
    model: LoadedModel,
    kv_cache_manager: KvCacheManager,
}

impl ModelInstance {
    pub(super) fn new(model: LoadedModel, kv_cache: KvCacheBackend) -> Self {
        Self {
            model,
            kv_cache_manager: KvCacheManager::new(kv_cache),
        }
    }

    pub(super) fn start_request(&mut self, request_id: u64) -> Result<()> {
        self.kv_cache_manager.start_request(request_id)
    }

    pub(super) fn forward(
        &mut self,
        request_id: u64,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let context = ForwardContext {
            request_id,
            start_position,
        };
        self.model
            .model()
            .forward(input, &context, &mut self.kv_cache_manager)
    }

    pub(super) fn model_metadata(&self) -> ModelMetadata {
        self.model.metadata()
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.kv_cache_manager.token_capacity()
    }

    pub(super) fn create_input_tensor(&self, token_ids: &[u32]) -> candle_core::Result<Tensor> {
        Tensor::new(token_ids, self.model.device())?.unsqueeze(0)
    }

    pub(super) fn forward_for_speculative_verification(
        &mut self,
        request_id: u64,
        input: &Tensor,
        start_position: usize,
    ) -> candle_core::Result<Tensor> {
        let context = ForwardContext {
            request_id,
            start_position,
        };
        self.model.model().forward_for_speculative_verification(
            input,
            &context,
            &mut self.kv_cache_manager,
        )
    }

    /// Restores reusable prefix pages and returns the prefill start position.
    pub(super) fn restore_cached_prefix(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
    ) -> Result<usize> {
        self.kv_cache_manager
            .restore_cached_prefix(request_id, token_ids)
    }

    pub(super) fn evicted_cached_token_count(&self) -> usize {
        self.kv_cache_manager.evicted_cached_token_count()
    }

    pub(super) fn truncate_cache(
        &mut self,
        request_id: u64,
        target_token_count: usize,
    ) -> Result<()> {
        self.kv_cache_manager
            .truncate(request_id, target_token_count)
    }

    pub(super) fn abort_request(&mut self, request_id: u64) -> Result<()> {
        self.kv_cache_manager.remove_request(request_id)
    }

    pub(super) fn finish_request(&mut self, request_id: u64, token_ids: &[u32]) -> Result<()> {
        self.kv_cache_manager.finish_request(request_id, token_ids)
    }
}
