use crate::model_runner::ActivationDType;
use crate::model_runner::InferenceDevice;
use crate::model_runner::KvCacheConfig;
use crate::models::loaded_model::LoadedModel;
use crate::models::ModelInfo;
use crate::models::ModelRole;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::GetModelMetadataResponse;
use crate::proto::model_runner::PrefixCacheTelemetry;
use anyhow::bail;
use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use candle_core::Device;
use log::info;
use std::path::Path;
use std::path::PathBuf;

use super::draft_token_count::DraftTokenCountController;
use super::draft_token_count::DraftTokenCountPolicy;
use super::kv_cache::create_kv_cache;
use super::kv_cache::KvCacheGeometry;
use super::model_instance::ModelInstance;
use super::text_generation;

pub(super) struct ModelRunnerMetadata {
    pub model_metadata: GetModelMetadataResponse,
    /// Minimum target/draft KV-cache token capacity. Request validation uses this
    /// shared limit because speculative generation must keep every sequence in both caches.
    pub kv_cache_token_capacity: usize,
}

pub(super) struct DraftModelRunnerConfig {
    pub(super) model_path: PathBuf,
    pub(super) token_count_policy: DraftTokenCountPolicy,
}

struct DraftModel {
    model: ModelInstance,
    token_count_policy: DraftTokenCountPolicy,
}

struct NewlyIndexedTokenCounts {
    target: usize,
    draft: Option<usize>,
}

pub(super) struct ModelRunnerPrefixCacheTelemetry {
    pub(super) target: PrefixCacheTelemetry,
    pub(super) draft: Option<PrefixCacheTelemetry>,
}

/// Owns the loaded models and executes requests on the inference thread.
pub(super) struct ModelRunner {
    target: ModelInstance,
    draft: Option<DraftModel>,
}

impl ModelRunner {
    pub(super) fn new(
        model_path: &Path,
        draft_model_config: Option<DraftModelRunnerConfig>,
        inference_device: InferenceDevice,
        activation_dtype: ActivationDType,
        kv_cache_config: KvCacheConfig,
    ) -> Result<Self> {
        let device = Self::get_inference_device(inference_device)?;
        let loaded_model = LoadedModel::new(model_path, device, activation_dtype)?;
        let loaded_draft_model = draft_model_config
            .map(|config| {
                let model = LoadedModel::new(
                    &config.model_path,
                    loaded_model.device().clone(),
                    activation_dtype,
                )?;
                Ok::<_, Error>((model, config.token_count_policy))
            })
            .transpose()?;
        info!(
            "Selected inference device {:?} for quantized GGUF inference",
            loaded_model.device()
        );
        let target_kv_cache = create_kv_cache(
            kv_cache_config.kv_cache_type,
            &loaded_model,
            ModelRole::Target,
            kv_cache_config.target_kv_cache_size_bytes,
        )?;
        let target_kv_cache_token_capacity = target_kv_cache.token_capacity();
        let draft = loaded_draft_model
            .map(|(model, token_count_policy)| {
                let draft_kv_cache_size_bytes = resolve_draft_kv_cache_size_bytes(
                    kv_cache_config.draft_kv_cache_size_bytes,
                    model.info(),
                    target_kv_cache_token_capacity,
                )?;
                let kv_cache = create_kv_cache(
                    kv_cache_config.kv_cache_type,
                    &model,
                    ModelRole::Draft,
                    draft_kv_cache_size_bytes,
                )?;
                Ok::<_, Error>(DraftModel {
                    model: ModelInstance::new(model, kv_cache),
                    token_count_policy,
                })
            })
            .transpose()?;
        Ok(Self {
            target: ModelInstance::new(loaded_model, target_kv_cache),
            draft,
        })
    }

    pub(super) fn metadata(&self) -> ModelRunnerMetadata {
        let target_model = self.target.model_metadata();
        let draft_model = self
            .draft
            .as_ref()
            .map(|draft| draft.model.model_metadata());
        let kv_cache_token_capacity = self.kv_cache_geometry().token_capacity;
        ModelRunnerMetadata {
            model_metadata: GetModelMetadataResponse {
                target_model: Some(target_model),
                draft_model,
            },
            kv_cache_token_capacity,
        }
    }

    pub(super) fn supports_multiple_active_requests(&self) -> bool {
        self.target.supports_multiple_active_requests()
    }

    pub(super) fn prefix_cache_telemetry(&self) -> ModelRunnerPrefixCacheTelemetry {
        ModelRunnerPrefixCacheTelemetry {
            target: PrefixCacheTelemetry {
                token_capacity: self.target.kv_cache_geometry().token_capacity as u64,
            },
            draft: self.draft.as_ref().map(|draft| PrefixCacheTelemetry {
                token_capacity: draft.model.kv_cache_geometry().token_capacity as u64,
            }),
        }
    }

    pub(super) fn kv_cache_geometry(&self) -> KvCacheGeometry {
        let target_geometry = self.target.kv_cache_geometry();
        let Some(draft) = &self.draft else {
            return target_geometry;
        };
        let draft_geometry = draft.model.kv_cache_geometry();
        KvCacheGeometry {
            token_capacity: target_geometry
                .token_capacity
                .min(draft_geometry.token_capacity),
            page_token_count: target_geometry.page_token_count,
        }
    }

    pub(super) fn start_request(
        &mut self,
        request: GenerateTextRequest,
    ) -> Result<text_generation::RequestExecutionState> {
        let request_id = request.request_id;
        self.prepare_model_instances(request_id)?;
        match (|| {
            // Restore only the input tokens before the final token. The final input token must
            // still pass through the model to produce the first generation logits, for both
            // regular and speculative decoding.
            let input_prefix = request
                .input_token_ids
                .split_last()
                .map_or(&[][..], |(_, prefix)| prefix);
            let prefill_initial_positions = text_generation::PrefillStartPositions {
                target: self
                    .target
                    .restore_cached_prefix(request_id, input_prefix)?,
                draft: self
                    .draft
                    .as_mut()
                    .map(|draft| draft.model.restore_cached_prefix(request_id, input_prefix))
                    .transpose()?,
            };
            text_generation::RequestExecutionState::new(
                request,
                self.draft
                    .as_ref()
                    .map(|draft| DraftTokenCountController::new(draft.token_count_policy)),
                prefill_initial_positions,
            )
        })() {
            Ok(execution_state) => Ok(execution_state),
            Err(error) => {
                if let Err(cleanup_error) = self.abort_request(request_id) {
                    return Err(error.context(format!(
                        "additionally failed to clear KV caches: {cleanup_error:#}"
                    )));
                }
                Err(error)
            }
        }
    }

    pub(super) fn run_steps(
        &mut self,
        execution_batch: &mut text_generation::RequestExecutionBatch,
    ) -> Result<Vec<text_generation::GenerationStep>> {
        execution_batch.run_steps(
            &mut self.target,
            self.draft.as_mut().map(|draft| &mut draft.model),
        )
    }

    pub(super) fn finish_request(
        &mut self,
        execution_state: text_generation::RequestExecutionState,
    ) -> Result<text_generation::CompletedGeneration> {
        let request_id = execution_state.request_id();
        let result = match execution_state.into_completed_generation() {
            Ok(result) => result,
            Err(error) => {
                if let Err(cleanup_error) = self.abort_request(request_id) {
                    return Err(error.context(format!(
                        "additionally failed to clear KV caches: {cleanup_error:#}"
                    )));
                }
                return Err(error);
            }
        };
        let newly_indexed_token_counts =
            match self.finalize_model_instances(request_id, &result.cached_sequence_token_ids) {
                Ok(token_counts) => token_counts,
                Err(error) => {
                    if let Err(cleanup_error) = self.abort_request(request_id) {
                        return Err(error.context(format!(
                            "additionally failed to clear KV caches: {cleanup_error:#}"
                        )));
                    }
                    return Err(error);
                }
            };
        Ok(result.with_newly_indexed_token_counts(
            newly_indexed_token_counts.target,
            newly_indexed_token_counts.draft,
        ))
    }

    pub(super) fn abort_request(&mut self, request_id: u64) -> Result<()> {
        let target_result = self.target.abort_request(request_id);
        let draft_result = self
            .draft
            .as_mut()
            .map(|draft| draft.model.abort_request(request_id))
            .transpose();
        target_result?;
        draft_result?;
        Ok(())
    }

    fn prepare_model_instances(&mut self, request_id: u64) -> Result<()> {
        self.target.start_request(request_id)?;
        if let Some(draft) = &mut self.draft {
            if let Err(error) = draft.model.start_request(request_id) {
                self.target.abort_request(request_id)?;
                return Err(error);
            }
        }
        Ok(())
    }

    fn finalize_model_instances(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
    ) -> Result<NewlyIndexedTokenCounts> {
        let target_result = self.target.finish_request(request_id, token_ids);
        let draft_result = self
            .draft
            .as_mut()
            .map(|draft| draft.model.finish_request(request_id, token_ids))
            .transpose();
        Ok(NewlyIndexedTokenCounts {
            target: target_result?,
            draft: draft_result?,
        })
    }

    fn get_inference_device(inference_device: InferenceDevice) -> Result<Device> {
        match inference_device {
            InferenceDevice::Cpu => Ok(Device::Cpu),
            InferenceDevice::Gpu => {
                cfg_if::cfg_if! {
                    if #[cfg(target_os = "macos")] {
                        Device::new_metal(0).context("failed to initialize the Metal device")
                    } else {
                        Device::new_cuda(0).context("failed to initialize the CUDA device")
                    }
                }
            }
            InferenceDevice::Mixed => bail!("expecting a specific device type"),
        }
    }
}

fn resolve_draft_kv_cache_size_bytes(
    configured_size_bytes: Option<usize>,
    model_info: &ModelInfo,
    target_kv_cache_token_capacity: usize,
) -> Result<usize> {
    match configured_size_bytes {
        Some(size_bytes) => Ok(size_bytes),
        None => model_info
            .kv_cache_bytes_per_token()
            .checked_mul(model_info.layer_count)
            .and_then(|size| size.checked_mul(target_kv_cache_token_capacity))
            .and_then(|size| size.checked_mul(2))
            .context("KV-cache size exceeds usize"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn derives_draft_cache_size_for_target_token_capacity() -> Result<()> {
        let draft_model_info = ModelInfo {
            layer_count: 4,
            num_kv_heads: 2,
            head_dim: 8,
            activation_dtype: DType::F32,
            context_length: 128,
        };
        let target_kv_cache_token_capacity = 128;

        let size_bytes = resolve_draft_kv_cache_size_bytes(
            None,
            &draft_model_info,
            target_kv_cache_token_capacity,
        )?;

        let derived_kv_cache_token_capacity = size_bytes
            / 2
            / draft_model_info.layer_count
            / draft_model_info.kv_cache_bytes_per_token();
        assert_eq!(
            derived_kv_cache_token_capacity,
            target_kv_cache_token_capacity
        );
        Ok(())
    }

    #[test]
    fn respects_configured_draft_cache_size() -> Result<()> {
        let draft_model_info = ModelInfo {
            layer_count: 4,
            num_kv_heads: 2,
            head_dim: 8,
            activation_dtype: DType::F32,
            context_length: 128,
        };

        assert_eq!(
            resolve_draft_kv_cache_size_bytes(Some(1024), &draft_model_info, 128)?,
            1024
        );
        Ok(())
    }
}
