use crate::model_loaders::KvCache;
use crate::model_runner::KvCacheType;
use candle_core::Tensor;

/// Stores one growing, contiguous key/value tensor pair per transformer layer.
pub(super) struct ContiguousKvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
}

impl ContiguousKvCache {
    pub(super) fn new(layer_count: usize) -> Self {
        Self {
            layers: vec![None; layer_count],
        }
    }
}

impl KvCache for ContiguousKvCache {
    fn clear(&mut self) {
        self.layers.fill(None);
    }

    fn layer(&mut self, layer_index: usize) -> &mut Option<(Tensor, Tensor)> {
        &mut self.layers[layer_index]
    }
}

pub(super) fn create_kv_cache(kv_cache_type: KvCacheType, layer_count: usize) -> Box<dyn KvCache> {
    match kv_cache_type {
        KvCacheType::Contiguous => Box::new(ContiguousKvCache::new(layer_count)),
    }
}
