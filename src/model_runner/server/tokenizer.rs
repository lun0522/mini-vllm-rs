use anyhow::Context;
use anyhow::Result;
use std::collections::BTreeSet;
use std::path::Path;
use tokenizers::Tokenizer;

pub(super) fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(anyhow::Error::msg)
        .context("failed to load the tokenizer")
}

pub(super) fn validate_tokenizer_compatibility(
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
}
