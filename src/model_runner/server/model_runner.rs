use crate::model_runner::InferenceDevice;
use crate::model_runner::KvCacheType;
use crate::models::loaded_model::LoadedModel;
use crate::models::ModelInfo;
use crate::models::ModelRole;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::GetModelMetadataResponse;
use anyhow::Context;
use anyhow::Result;
use candle_core::Device;
use log::info;
use std::path::Path;

use super::kv_cache::create_kv_cache;
use super::model_instance::ModelInstance;
use super::text_generation;

/// Owns the loaded models and executes requests on the inference thread.
pub(super) struct ModelRunner {
    target: ModelInstance,
    draft: Option<ModelInstance>,
    draft_token_count: usize,
}

impl ModelRunner {
    pub(super) fn new(
        model_path: &Path,
        draft_model_path: Option<&Path>,
        draft_token_count: usize,
        inference_device: InferenceDevice,
        kv_cache_type: KvCacheType,
        target_kv_cache_size_bytes: usize,
    ) -> Result<Self> {
        let device = Self::get_inference_device(inference_device)?;
        let loaded_model = LoadedModel::new(model_path, device)?;
        let loaded_draft_model = draft_model_path
            .map(|draft_model_path| {
                LoadedModel::new(draft_model_path, loaded_model.device().clone())
            })
            .transpose()?;
        info!(
            "Selected inference device {:?} for quantized GGUF inference",
            loaded_model.device()
        );
        let target_kv_cache = create_kv_cache(
            kv_cache_type,
            &loaded_model,
            ModelRole::Target,
            target_kv_cache_size_bytes,
        )?;
        let target_kv_cache_token_capacity = target_kv_cache.token_capacity();
        let draft = loaded_draft_model
            .map(|model| {
                let draft_kv_cache_size_bytes =
                    compute_kv_cache_size_bytes(model.info(), target_kv_cache_token_capacity)?;
                let kv_cache = create_kv_cache(
                    kv_cache_type,
                    &model,
                    ModelRole::Draft,
                    draft_kv_cache_size_bytes,
                )?;
                Ok::<_, anyhow::Error>(ModelInstance::new(model, kv_cache))
            })
            .transpose()?;
        Ok(Self {
            target: ModelInstance::new(loaded_model, target_kv_cache),
            draft,
            draft_token_count,
        })
    }

    pub(super) fn model_metadata(&self) -> GetModelMetadataResponse {
        let target_model = self.target.model_metadata();
        let draft_model = self.draft.as_ref().map(ModelInstance::model_metadata);
        GetModelMetadataResponse {
            target_model: Some(target_model),
            draft_model,
        }
    }

    pub(super) fn token_capacity(&self) -> usize {
        self.target.token_capacity()
    }

    pub(super) fn evicted_cached_token_count(&self) -> usize {
        self.target.evicted_cached_token_count().saturating_add(
            self.draft
                .as_ref()
                .map_or(0, ModelInstance::evicted_cached_token_count),
        )
    }

    pub(super) fn generate_text(
        &mut self,
        request: &GenerateTextRequest,
        push_token: impl FnMut(u32) -> Result<()>,
        is_cancelled: impl FnMut() -> bool,
    ) -> Result<text_generation::TextGenerationResult> {
        self.start_request(request.request_id)?;
        let result = match (|| {
            // Restore only the prompt tokens before the final token. The final prompt token must
            // still pass through the model to produce the first generation logits, for both
            // regular and speculative decoding.
            let prompt_prefix = request
                .input_token_ids
                .split_last()
                .map_or(&[][..], |(_, prefix)| prefix);
            let prefill_start_positions = text_generation::PrefillStartPositions {
                target: self
                    .target
                    .restore_cached_prefix(request.request_id, prompt_prefix)?,
                draft: self
                    .draft
                    .as_mut()
                    .map(|draft| draft.restore_cached_prefix(request.request_id, prompt_prefix))
                    .transpose()?,
            };
            let result = text_generation::generate_text(
                &mut self.target,
                self.draft.as_mut(),
                self.draft_token_count,
                prefill_start_positions,
                request,
                push_token,
                is_cancelled,
            )?;
            Ok::<_, anyhow::Error>(result)
        })() {
            Ok(result) => result,
            Err(error) => {
                if let Err(cleanup_error) = self.abort_request(request.request_id) {
                    return Err(error.context(format!(
                        "additionally failed to clear KV caches: {cleanup_error:#}"
                    )));
                }
                return Err(error);
            }
        };
        if let Err(error) = self.finish_request(request.request_id, &result.token_ids) {
            if let Err(cleanup_error) = self.abort_request(request.request_id) {
                return Err(error.context(format!(
                    "additionally failed to clear KV caches: {cleanup_error:#}"
                )));
            }
            return Err(error);
        }
        Ok(result)
    }

    fn start_request(&mut self, request_id: u64) -> Result<()> {
        self.target.start_request(request_id)?;
        if let Some(draft) = &mut self.draft {
            if let Err(error) = draft.start_request(request_id) {
                self.target.abort_request(request_id)?;
                return Err(error);
            }
        }
        Ok(())
    }

    fn finish_request(&mut self, request_id: u64, token_ids: &[u32]) -> Result<()> {
        let target_result = self.target.finish_request(request_id, token_ids);
        let draft_result = self
            .draft
            .as_mut()
            .map(|draft| draft.finish_request(request_id, token_ids))
            .transpose();
        target_result?;
        draft_result?;
        Ok(())
    }

    fn abort_request(&mut self, request_id: u64) -> Result<()> {
        let target_result = self.target.abort_request(request_id);
        let draft_result = self
            .draft
            .as_mut()
            .map(|draft| draft.abort_request(request_id))
            .transpose();
        target_result?;
        draft_result?;
        Ok(())
    }

    fn get_inference_device(inference_device: InferenceDevice) -> Result<Device> {
        match inference_device {
            InferenceDevice::Cpu => Ok(Device::Cpu),
            InferenceDevice::Gpu => {
                Device::new_metal(0).context("failed to initialize the Metal device")
            }
        }
    }
}

fn compute_kv_cache_size_bytes(model_info: &ModelInfo, token_capacity: usize) -> Result<usize> {
    model_info
        .kv_cache_bytes_per_token()
        .checked_mul(model_info.layer_count)
        .and_then(|size| size.checked_mul(token_capacity))
        .and_then(|size| size.checked_mul(2))
        .context("KV-cache size exceeds usize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn derives_draft_cache_size_for_target_token_capacity() -> Result<()> {
        let draft_model_info = ModelInfo {
            layer_count: 4,
            kv_head_count: 2,
            head_dimension: 8,
            activation_dtype: DType::F32,
        };
        let target_token_capacity = 128;

        let size_bytes = compute_kv_cache_size_bytes(&draft_model_info, target_token_capacity)?;

        let derived_token_capacity = size_bytes
            / 2
            / draft_model_info.layer_count
            / draft_model_info.kv_cache_bytes_per_token();
        assert_eq!(derived_token_capacity, target_token_capacity);
        Ok(())
    }
}
