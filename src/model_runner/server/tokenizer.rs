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
