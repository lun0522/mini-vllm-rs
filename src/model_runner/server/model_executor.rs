use crate::model_runner::ActivationDType;
use crate::model_runner::InferenceDevice;
use crate::model_runner::KvCacheConfig;
use crate::model_runner::PrefixCachePublicationPolicy;
use crate::models::loaded_model::LoadedModel;
use crate::models::ModelInfo;
use crate::models::ModelRole;
use crate::proto::model_runner::GenerateTextRequest;
use crate::proto::model_runner::GetModelMetadataResponse;
#[allow(unused_imports)]
use anyhow::bail;
use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use candle_core::Device;
use log::info;
use std::path::Path;
use std::path::PathBuf;

use super::generation::draft_token_count::DraftTokenCountController;
use super::generation::draft_token_count::DraftTokenCountPolicy;
use super::generation::generation_stats::PrefixCacheGenerationStats;
use super::generation::generation_stats::PrefixCacheModelStats;
use super::generation::generation_stats::PrefixTokenCounts;
use super::generation::generation_stats::TextGenerationStats;
use super::generation::text_generation;
use super::kv_cache::create_kv_cache;
use super::kv_cache::KvCacheGeometry;
use super::kv_cache::PrefixCacheAffinitySnapshot;
use super::kv_cache::PrefixCacheTelemetry;
use super::model_instance::ModelInstance;
use super::BackendId;

pub(super) struct ModelExecutorMetadata {
    pub model_metadata: GetModelMetadataResponse,
    /// Minimum target/draft KV-cache token capacity. Request validation uses this
    /// shared limit because speculative generation must keep every sequence in both caches.
    pub kv_cache_token_capacity: usize,
}

pub(super) struct DraftModelConfig {
    pub(super) model_path: PathBuf,
    pub(super) token_count_policy: DraftTokenCountPolicy,
}

struct DraftModel {
    model: ModelInstance,
    token_count_policy: DraftTokenCountPolicy,
}

pub(super) struct ModelExecutorPrefixCacheTelemetry {
    pub(super) target: Option<PrefixCacheTelemetry>,
    pub(super) draft: Option<PrefixCacheTelemetry>,
}

/// Owns the loaded models and executes requests on the inference thread.
pub(super) struct ModelExecutor {
    target: ModelInstance,
    draft: Option<DraftModel>,
    tokenizer_vocabulary_size: usize,
    prefix_cache_publication_policy: PrefixCachePublicationPolicy,
}

impl ModelExecutor {
    pub(super) fn new(
        backend_id: BackendId,
        model_path: &Path,
        draft_model_config: Option<DraftModelConfig>,
        inference_device: InferenceDevice,
        gpu_device_index: usize,
        activation_dtype: ActivationDType,
        kv_cache_config: KvCacheConfig,
    ) -> Result<Self> {
        let prefix_cache_publication_policy = kv_cache_config.prefix_cache_publication_policy;
        let device = Self::get_inference_device(inference_device, gpu_device_index)?;
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
        let tokenizer_vocabulary_size =
            usize::try_from(loaded_model.metadata().tokenizer_vocabulary_size)
                .context("target tokenizer vocabulary size does not fit in usize")?;
        if let Some((draft_model, _)) = &loaded_draft_model {
            let draft_tokenizer_vocabulary_size =
                usize::try_from(draft_model.metadata().tokenizer_vocabulary_size)
                    .context("draft tokenizer vocabulary size does not fit in usize")?;
            anyhow::ensure!(
                draft_tokenizer_vocabulary_size == tokenizer_vocabulary_size,
                "target and draft GGUF tokenizers define different vocabulary sizes: \
                 {tokenizer_vocabulary_size} and {draft_tokenizer_vocabulary_size}"
            );
        }
        info!(
            "[Backend {backend_id}] Selected inference device {:?} for quantized GGUF inference",
            loaded_model.device()
        );
        let target_kv_cache = create_kv_cache(
            backend_id,
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
                    backend_id,
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
            tokenizer_vocabulary_size,
            prefix_cache_publication_policy,
        })
    }

    pub(super) fn metadata(&self) -> ModelExecutorMetadata {
        let target_model = self.target.model_metadata();
        let draft_model = self
            .draft
            .as_ref()
            .map(|draft| draft.model.model_metadata());
        let kv_cache_token_capacity = self.kv_cache_geometry().token_capacity;
        ModelExecutorMetadata {
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

    pub(super) fn prefix_cache_telemetry(&self) -> ModelExecutorPrefixCacheTelemetry {
        ModelExecutorPrefixCacheTelemetry {
            target: self.target.prefix_cache_telemetry(),
            draft: self
                .draft
                .as_ref()
                .and_then(|draft| draft.model.prefix_cache_telemetry()),
        }
    }

    /// Creates a routing snapshot from the target model's prefix cache.
    pub(super) fn create_cache_affinity_snapshot(&self) -> Option<PrefixCacheAffinitySnapshot> {
        self.target.create_cache_affinity_snapshot()
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
                self.tokenizer_vocabulary_size,
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
        let generation_steps = execution_batch.run_steps(
            &mut self.target,
            self.draft.as_mut().map(|draft| &mut draft.model),
        )?;
        // Incremental mode publishes complete prefix pages after every successful execution step.
        if self.prefix_cache_publication_policy == PrefixCachePublicationPolicy::Incremental {
            for execution_state in execution_batch.execution_states() {
                self.publish_completed_prefix(
                    execution_state.request_id(),
                    execution_state.committed_token_ids(),
                )?;
            }
        }
        Ok(generation_steps)
    }

    /// Publishes complete target and draft prefix pages without finalizing the request.
    ///
    /// Publication policies may call this internally while coordination policies call it through
    /// the inference worker. Repeated calls are safe: each call publishes only complete pages not
    /// previously published by the request.
    pub(super) fn publish_completed_prefix(
        &mut self,
        request_id: u64,
        token_ids: &[u32],
    ) -> Result<()> {
        self.target
            .publish_completed_prefix(request_id, token_ids)?;
        if let Some(draft) = &mut self.draft {
            draft
                .model
                .publish_completed_prefix(request_id, token_ids)?;
        }
        Ok(())
    }

    pub(super) fn finish_request(
        &mut self,
        execution_state: text_generation::RequestExecutionState,
    ) -> Result<TextGenerationStats> {
        let request_id = execution_state.request_id();
        let output = match execution_state.into_generation_output() {
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
            match self.finalize_model_instances(request_id, &output.committed_token_ids) {
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
        Ok(TextGenerationStats {
            execution: output.execution_stats,
            prefix_cache: PrefixCacheGenerationStats {
                target: PrefixCacheModelStats {
                    restored_token_count: output.restored_prefix_token_counts.target,
                    newly_indexed_token_count: newly_indexed_token_counts.target,
                },
                draft: output
                    .restored_prefix_token_counts
                    .draft
                    .map(|restored_token_count| PrefixCacheModelStats {
                        restored_token_count,
                        newly_indexed_token_count: newly_indexed_token_counts.draft.expect(
                            "draft prefix restoration requires draft KV-cache finalization",
                        ),
                    }),
            },
        })
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
    ) -> Result<PrefixTokenCounts> {
        let target_result = self.target.finish_request(request_id, token_ids);
        let draft_result = self
            .draft
            .as_mut()
            .map(|draft| draft.model.finish_request(request_id, token_ids))
            .transpose();
        Ok(PrefixTokenCounts {
            target: target_result?,
            draft: draft_result?,
        })
    }

    fn get_inference_device(
        inference_device: InferenceDevice,
        #[allow(unused_variables)] gpu_device_index: usize,
    ) -> Result<Device> {
        match inference_device {
            InferenceDevice::Cpu => Ok(Device::Cpu),
            InferenceDevice::Gpu => {
                cfg_if::cfg_if! {
                    if #[cfg(all(target_os = "macos", feature = "metal"))] {
                        Device::new_metal(gpu_device_index).with_context(|| {
                            format!("failed to initialize Metal device {gpu_device_index}")
                        })
                    } else if #[cfg(all(target_os = "macos", not(feature = "metal")))] {
                        bail!(
                            "GPU inference on macOS requires the `metal` Cargo feature; \
                             rebuild with `--features metal`"
                        )
                    } else if #[cfg(all(target_os = "linux", feature = "cuda"))] {
                        Device::new_cuda(gpu_device_index).with_context(|| {
                            format!("failed to initialize CUDA device {gpu_device_index}")
                        })
                    } else if #[cfg(all(target_os = "linux", not(feature = "cuda")))] {
                        bail!(
                            "GPU inference on Linux requires the `cuda` Cargo feature; \
                             rebuild with `--features cuda`"
                        )
                    } else {
                        bail!("GPU inference is unsupported on this operating system")
                    }
                }
            }
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
    use crate::model_runner::KvCacheType;
    use crate::models::CausalLanguageModel;
    use crate::models::ForwardInput;
    use crate::models::ForwardOutput;
    use crate::models::KvCache;
    use candle_core::DType;
    use candle_core::Tensor;

    #[cfg(all(target_os = "macos", not(feature = "metal")))]
    #[test]
    fn gpu_request_explains_missing_metal_feature() {
        let error = ModelExecutor::get_inference_device(InferenceDevice::Gpu, 0)
            .expect_err("CPU-only builds must reject GPU inference");
        assert!(
            error
                .to_string()
                .contains("requires the `metal` Cargo feature"),
            "unexpected error: {error:#}"
        );
    }

    #[cfg(all(target_os = "linux", not(feature = "cuda")))]
    #[test]
    fn gpu_request_explains_missing_cuda_feature() {
        let error = ModelExecutor::get_inference_device(InferenceDevice::Gpu, 0)
            .expect_err("CPU-only builds must reject GPU inference");
        assert!(
            error
                .to_string()
                .contains("requires the `cuda` Cargo feature"),
            "unexpected error: {error:#}"
        );
    }

    struct TestModel {
        info: ModelInfo,
        next_token: u32,
    }

    impl CausalLanguageModel for TestModel {
        fn info(&self) -> &ModelInfo {
            &self.info
        }

        fn forward_with_speculative_verification(
            &mut self,
            generation_inputs: &[ForwardInput],
            verification_inputs: &[ForwardInput],
            kv_cache: &mut dyn KvCache,
        ) -> Result<ForwardOutput> {
            let generation_logits = generation_inputs
                .iter()
                .map(|input| {
                    append_test_cache_and_create_logits(input, kv_cache, self.next_token, false)
                })
                .collect::<Result<Vec<_>>>()?;
            let verification_logits = verification_inputs
                .iter()
                .map(|input| {
                    append_test_cache_and_create_logits(input, kv_cache, self.next_token, true)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(ForwardOutput {
                generation_logits,
                verification_logits,
            })
        }
    }

    fn append_test_cache_and_create_logits(
        input: &ForwardInput,
        kv_cache: &mut dyn KvCache,
        next_token: u32,
        return_sequence_logits: bool,
    ) -> Result<Tensor> {
        let token_count = input.input.dim(1)?;
        let cache_values = input.input.reshape((1, 1, token_count, 1))?;
        kv_cache.append_new_key_value(
            input.request_id,
            /* layer_index */ 0,
            input.start_position,
            &cache_values,
            &cache_values,
        )?;
        let mut logits = vec![0.0f32; 2];
        logits[next_token as usize] = 1.0;
        let logits = Tensor::new(logits, &Device::Cpu)?;
        if return_sequence_logits {
            Ok(logits.broadcast_as((token_count, 2))?)
        } else {
            Ok(logits)
        }
    }

    fn test_model_instance(next_token: u32) -> Result<ModelInstance> {
        let model = LoadedModel::for_test(Box::new(TestModel {
            info: ModelInfo {
                layer_count: 1,
                num_kv_heads: 1,
                head_dim: 1,
                activation_dtype: DType::U32,
                context_length: 16,
            },
            next_token,
        }));
        let cache = create_kv_cache(
            BackendId::from_index(0),
            KvCacheType::Paged {
                per_page_token_count: 2,
                enable_prefix_caching: true,
            },
            &model,
            ModelRole::Target,
            /* total_size_bytes */ 64,
        )?;
        Ok(ModelInstance::new(model, cache))
    }

    fn test_model_executor(policy: PrefixCachePublicationPolicy) -> Result<ModelExecutor> {
        Ok(ModelExecutor {
            target: test_model_instance(/* next_token */ 1)?,
            draft: None,
            tokenizer_vocabulary_size: 2,
            prefix_cache_publication_policy: policy,
        })
    }

    fn speculative_test_model_executor() -> Result<ModelExecutor> {
        Ok(ModelExecutor {
            target: test_model_instance(/* next_token */ 1)?,
            draft: Some(DraftModel {
                model: test_model_instance(/* next_token */ 0)?,
                token_count_policy: DraftTokenCountPolicy::Fixed {
                    draft_token_count: 2,
                },
            }),
            tokenizer_vocabulary_size: 2,
            prefix_cache_publication_policy: PrefixCachePublicationPolicy::Incremental,
        })
    }

    fn test_request() -> GenerateTextRequest {
        GenerateTextRequest {
            request_id: 1,
            input_token_ids: vec![1, 2, 3, 4, 5],
            max_new_tokens: 1,
            repeat_penalty: 1.0,
            repeat_last_n: 16,
            ..Default::default()
        }
    }

    fn current_indexed_token_count(model_executor: &ModelExecutor) -> usize {
        model_executor
            .target
            .prefix_cache_telemetry()
            .expect("test cache should have prefix caching enabled")
            .current_indexed_token_count
    }

    fn draft_current_indexed_token_count(model_executor: &ModelExecutor) -> usize {
        model_executor
            .draft
            .as_ref()
            .expect("test model executor should have a draft model")
            .model
            .prefix_cache_telemetry()
            .expect("test cache should have prefix caching enabled")
            .current_indexed_token_count
    }

    #[test]
    fn completion_policy_publishes_only_when_finishing() -> Result<()> {
        let mut model_executor =
            test_model_executor(PrefixCachePublicationPolicy::OnRequestCompletion)?;
        let execution_state = model_executor.start_request(test_request())?;
        let mut batch = text_generation::RequestExecutionBatch::with_capacity(1);
        batch.push(Box::new(execution_state), /* token_budget */ 5);

        model_executor.run_steps(&mut batch)?;
        assert_eq!(current_indexed_token_count(&model_executor), 0);

        let execution_state = *batch
            .into_execution_states()
            .next()
            .expect("test batch should contain one request");
        let stats = model_executor.finish_request(execution_state)?;
        assert_eq!(stats.prefix_cache.target.newly_indexed_token_count, 4);
        assert_eq!(current_indexed_token_count(&model_executor), 4);
        Ok(())
    }

    #[test]
    fn incremental_policy_publishes_new_complete_pages_after_each_step() -> Result<()> {
        let mut model_executor = test_model_executor(PrefixCachePublicationPolicy::Incremental)?;
        let execution_state = model_executor.start_request(test_request())?;
        let mut batch = text_generation::RequestExecutionBatch::with_capacity(1);
        batch.push(Box::new(execution_state), /* token_budget */ 2);

        model_executor.run_steps(&mut batch)?;
        assert_eq!(current_indexed_token_count(&model_executor), 2);

        model_executor.run_steps(&mut batch)?;
        assert_eq!(current_indexed_token_count(&model_executor), 4);

        // The final one-token prefill fragment does not complete another two-token page.
        model_executor.run_steps(&mut batch)?;
        assert_eq!(current_indexed_token_count(&model_executor), 4);

        let execution_state = *batch
            .into_execution_states()
            .next()
            .expect("test batch should contain one request");
        let stats = model_executor.finish_request(execution_state)?;
        assert_eq!(stats.prefix_cache.target.newly_indexed_token_count, 4);
        assert_eq!(current_indexed_token_count(&model_executor), 4);
        Ok(())
    }

    #[test]
    fn incremental_policy_publishes_after_speculative_rollback() -> Result<()> {
        let mut model_executor = speculative_test_model_executor()?;
        let mut request = test_request();
        request.max_new_tokens = 1;
        let execution_state = model_executor.start_request(request)?;
        let mut batch = text_generation::RequestExecutionBatch::with_capacity(1);
        batch.push(Box::new(execution_state), /* token_budget */ 4);

        // Speculative prefill caches and publishes the first four input tokens in both models.
        model_executor.run_steps(&mut batch)?;
        assert_eq!(current_indexed_token_count(&model_executor), 4);
        assert_eq!(draft_current_indexed_token_count(&model_executor), 4);

        // The draft proposes token 0 while the target selects token 1. Verification temporarily
        // appends the rejected proposal, then truncates both caches to five tokens. Publication
        // must therefore leave the four-token complete prefix unchanged.
        model_executor.run_steps(&mut batch)?;
        assert_eq!(current_indexed_token_count(&model_executor), 4);
        assert_eq!(draft_current_indexed_token_count(&model_executor), 4);

        let execution_state = *batch
            .into_execution_states()
            .next()
            .expect("test batch should contain one request");
        let stats = model_executor.finish_request(execution_state)?;
        assert_eq!(stats.prefix_cache.target.newly_indexed_token_count, 4);
        assert_eq!(
            stats
                .prefix_cache
                .draft
                .expect("speculative generation should report draft stats")
                .newly_indexed_token_count,
            4
        );
        Ok(())
    }

    #[test]
    fn publishes_target_and_draft_prefixes_at_their_independent_progress() -> Result<()> {
        let mut model_executor = speculative_test_model_executor()?;
        let request_id = 1;
        model_executor.prepare_model_instances(request_id)?;

        let target_input = ForwardInput {
            request_id,
            input: Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu)?,
            start_position: 0,
        };
        model_executor.target.forward(&[target_input])?;
        let draft_input = ForwardInput {
            request_id,
            input: Tensor::new(&[[1u32, 2]], &Device::Cpu)?,
            start_position: 0,
        };
        model_executor
            .draft
            .as_mut()
            .expect("test model runner should have a draft model")
            .model
            .forward(&[draft_input])?;

        model_executor.publish_completed_prefix(request_id, &[1, 2, 3, 4])?;

        assert_eq!(current_indexed_token_count(&model_executor), 4);
        assert_eq!(draft_current_indexed_token_count(&model_executor), 2);
        model_executor.abort_request(request_id)?;
        Ok(())
    }

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
