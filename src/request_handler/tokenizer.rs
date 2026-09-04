use crate::model_loaders::ModelRole;
use crate::proto::GetModelMetadataResponse;
use crate::proto::ModelMetadata;
use anyhow::Context;
use anyhow::Error;
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Clone, Copy)]
pub(crate) enum ModelArchitecture {
    Llama,
    Qwen2,
}

impl ModelArchitecture {
    pub(crate) fn format_chat_prompt(self, prompt: &str) -> String {
        match self {
            Self::Llama => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\
                 {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            Self::Qwen2 => {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
        }
    }

    pub(crate) fn end_of_sequence_tokens(self) -> &'static [&'static str] {
        match self {
            Self::Llama => &["<|eot_id|>", "<|end_of_text|>"],
            Self::Qwen2 => &["<|im_end|>", "<|endoftext|>"],
        }
    }
}

pub(crate) fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(Error::msg)
        .context("failed to load the tokenizer")
}

pub(super) fn load_and_validate_tokenizer(
    tokenizer_path: &Path,
    draft_tokenizer_path: Option<&Path>,
    model_metadata: &GetModelMetadataResponse,
) -> Result<Tokenizer> {
    let tokenizer =
        load_tokenizer(tokenizer_path).context("failed to load the target tokenizer")?;
    validate_model_vocabulary(
        &tokenizer,
        model_metadata
            .target_model
            .as_ref()
            .context("model runner did not report target model metadata")?,
        ModelRole::Target,
    )?;

    match (draft_tokenizer_path, &model_metadata.draft_model) {
        (Some(path), Some(metadata)) => {
            let draft_tokenizer =
                load_tokenizer(path).context("failed to load the draft tokenizer")?;
            validate_model_vocabulary(&tokenizer, metadata, ModelRole::Draft)?;
            validate_tokenizer_compatibility(&tokenizer, &draft_tokenizer)?;
            // Generation uses the target tokenizer after confirming that both tokenizers assign
            // the same IDs, so we drop the second tokenizer after validation.
        }
        (None, None) => {}
        (Some(_), None) => anyhow::bail!("a draft tokenizer was provided without a draft model"),
        (None, Some(_)) => anyhow::bail!("the draft model does not have a tokenizer"),
    }
    Ok(tokenizer)
}

/// Ensures target and draft tokenizers assign the same ID to every token.
pub(crate) fn validate_tokenizer_compatibility(
    target_tokenizer: &Tokenizer,
    draft_tokenizer: &Tokenizer,
) -> Result<()> {
    let target_vocabulary = target_tokenizer.get_vocab(true);
    let draft_vocabulary = draft_tokenizer.get_vocab(true);
    if target_vocabulary == draft_vocabulary {
        return Ok(());
    }

    let tokens: BTreeSet<_> = target_vocabulary
        .keys()
        .chain(draft_vocabulary.keys())
        .collect();
    let mismatched_token = tokens
        .into_iter()
        .find(|token| target_vocabulary.get(*token) != draft_vocabulary.get(*token));
    let mismatch = mismatched_token.map_or_else(
        || "no individual mismatch found".to_owned(),
        |token| {
            format!(
                "token {token:?} maps to {:?} in the target tokenizer and {:?} in the draft tokenizer",
                target_vocabulary.get(token),
                draft_vocabulary.get(token),
            )
        },
    );
    anyhow::bail!(
        "draft tokenizer is incompatible with the target tokenizer: target vocabulary has {} entries, \
         draft vocabulary has {} entries; {mismatch}",
        target_vocabulary.len(),
        draft_vocabulary.len(),
    )
}

/// Checks the tokenizer's token-ID range against the model's input and output vocabulary sizes.
pub(crate) fn validate_model_vocabulary(
    tokenizer: &Tokenizer,
    model_metadata: &ModelMetadata,
    model_role: ModelRole,
) -> Result<()> {
    let required_size = tokenizer
        .get_vocab(true)
        .values()
        .copied()
        .max()
        .map_or(0, |maximum_id| u64::from(maximum_id) + 1);
    if model_metadata.input_vocabulary_size < required_size {
        anyhow::bail!(
            "{model_role} model input vocabulary has {} entries but the shared tokenizer requires \
             token IDs through {}",
            model_metadata.input_vocabulary_size,
            required_size.saturating_sub(1),
        );
    }
    if model_metadata.output_vocabulary_size < required_size {
        anyhow::bail!(
            "{model_role} model output vocabulary has {} entries but the shared tokenizer requires \
             token IDs through {}",
            model_metadata.output_vocabulary_size,
            required_size.saturating_sub(1),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer(vocabulary: &[(&str, u32)]) -> Tokenizer {
        let vocabulary = vocabulary
            .iter()
            .map(|(token, id)| format!(r#""{token}":{id}"#))
            .collect::<Vec<_>>()
            .join(",");
        Tokenizer::from_bytes(format!(
            r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{{{vocabulary}}},"unk_token":"[UNK]"}}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn accepts_identical_tokenizers_and_reports_id_mismatches() {
        let target = tokenizer(&[("a", 0), ("b", 1)]);
        let compatible = tokenizer(&[("a", 0), ("b", 1)]);
        assert!(validate_tokenizer_compatibility(&target, &compatible).is_ok());

        let incompatible = tokenizer(&[("a", 1), ("b", 0)]);
        let error = validate_tokenizer_compatibility(&target, &incompatible)
            .unwrap_err()
            .to_string();
        assert!(error.contains("token \"a\" maps to Some(0)"), "{error}");
        assert!(error.contains("draft tokenizer"), "{error}");
    }

    #[test]
    fn reports_tokens_missing_from_either_tokenizer() {
        let target = tokenizer(&[("a", 0), ("b", 1)]);
        let missing_from_draft = tokenizer(&[("a", 0)]);
        let error = validate_tokenizer_compatibility(&target, &missing_from_draft)
            .unwrap_err()
            .to_string();
        assert!(error.contains("target vocabulary has 2"), "{error}");
        assert!(error.contains("draft vocabulary has 1"), "{error}");
        assert!(
            error.contains("token \"b\" maps to Some(1) in the target tokenizer and None"),
            "{error}"
        );

        let target = tokenizer(&[("a", 0)]);
        let extra_in_draft = tokenizer(&[("a", 0), ("b", 1)]);
        let error = validate_tokenizer_compatibility(&target, &extra_in_draft)
            .unwrap_err()
            .to_string();
        assert!(error.contains("target vocabulary has 1"), "{error}");
        assert!(error.contains("draft vocabulary has 2"), "{error}");
        assert!(
            error.contains("token \"b\" maps to None in the target tokenizer and Some(1)"),
            "{error}"
        );
    }

    #[test]
    fn validates_model_input_and_output_vocabulary_sizes() {
        let tokenizer = tokenizer(&[("a", 0), ("b", 2)]);
        let metadata = ModelMetadata {
            architecture: 0,
            input_vocabulary_size: 3,
            output_vocabulary_size: 3,
        };
        assert!(validate_model_vocabulary(&tokenizer, &metadata, ModelRole::Target).is_ok());

        let input_error = validate_model_vocabulary(
            &tokenizer,
            &ModelMetadata {
                input_vocabulary_size: 2,
                ..metadata
            },
            ModelRole::Target,
        )
        .unwrap_err()
        .to_string();
        assert!(input_error.contains("target model input vocabulary has 2 entries"));

        let output_error = validate_model_vocabulary(
            &tokenizer,
            &ModelMetadata {
                output_vocabulary_size: 2,
                ..metadata
            },
            ModelRole::Draft,
        )
        .unwrap_err()
        .to_string();
        assert!(output_error.contains("draft model output vocabulary has 2 entries"));
    }
}
