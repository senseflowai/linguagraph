//! Stable BM25-style sparse tokenization, kept byte-for-byte compatible
//! with qlink's `src/bm25.rs`.
//!
//! qlink writes each `_canonical` point with a named sparse vector
//! ([`SPARSE_VECTOR`]) built by tokenizing the canonical text; the Qdrant
//! collection is created with the IDF modifier, so the store computes the
//! IDF term itself and the client only supplies term ids + term
//! frequencies. When linguagraph queries that collection directly (the
//! client-side hybrid grounding path), it **must** tokenize the query
//! exactly the way qlink tokenized the documents — otherwise the term ids
//! diverge and the lexical (BM25) branch silently returns nothing. This
//! module is therefore a faithful port of qlink's tokenizer, including the
//! FNV-1a hash and the Russian stopword set; the tests lock the same
//! vectors qlink locks.

use std::collections::BTreeMap;

/// Name of the BM25 sparse vector inside a hybrid `_canonical` collection.
/// The dense vector stays unnamed (the collection default).
pub const SPARSE_VECTOR: &str = "text_bm25";

/// Tokenize and hash `text` into Qdrant sparse-vector parts
/// `(term_ids, term_frequencies)`, term ids sorted ascending.
pub fn to_sparse_parts(text: &str) -> (Vec<u32>, Vec<f32>) {
    let mut term_freqs = BTreeMap::<u32, f32>::new();
    for token in tokenize(text) {
        *term_freqs.entry(hash_token(&token)).or_default() += 1.0;
    }
    term_freqs.into_iter().unzip()
}

/// Lowercase, split on non-alphanumeric boundaries, drop empty tokens and
/// Russian stopwords. Fixed (not configurable): ingest and query must
/// agree byte-for-byte.
fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|ch: char| !ch.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .filter(|token| !is_russian_stopword(token))
}

/// Stable FNV-1a 32-bit hash of a token. Deterministic across builds,
/// platforms and Rust versions (unlike `DefaultHasher`).
fn hash_token(token: &str) -> u32 {
    const FNV_OFFSET: u32 = 0x811c_9dc5;
    const FNV_PRIME: u32 = 0x0100_0193;
    let mut hash = FNV_OFFSET;
    for &byte in token.as_bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn is_russian_stopword(token: &str) -> bool {
    matches!(
        token,
        "а" | "без"
            | "более"
            | "бы"
            | "был"
            | "была"
            | "были"
            | "было"
            | "быть"
            | "в"
            | "вам"
            | "вас"
            | "весь"
            | "во"
            | "вот"
            | "все"
            | "всего"
            | "всех"
            | "вы"
            | "где"
            | "да"
            | "даже"
            | "для"
            | "до"
            | "его"
            | "ее"
            | "если"
            | "есть"
            | "еще"
            | "же"
            | "за"
            | "здесь"
            | "и"
            | "из"
            | "или"
            | "им"
            | "их"
            | "к"
            | "как"
            | "ко"
            | "когда"
            | "кто"
            | "ли"
            | "либо"
            | "мне"
            | "может"
            | "мы"
            | "на"
            | "над"
            | "надо"
            | "наш"
            | "не"
            | "него"
            | "нее"
            | "нет"
            | "ни"
            | "них"
            | "но"
            | "ну"
            | "о"
            | "об"
            | "однако"
            | "он"
            | "она"
            | "они"
            | "оно"
            | "от"
            | "очень"
            | "по"
            | "под"
            | "при"
            | "с"
            | "со"
            | "так"
            | "также"
            | "такой"
            | "там"
            | "те"
            | "тем"
            | "то"
            | "того"
            | "тоже"
            | "той"
            | "только"
            | "том"
            | "ты"
            | "у"
            | "уже"
            | "хотя"
            | "чего"
            | "чей"
            | "чем"
            | "что"
            | "чтобы"
            | "чье"
            | "эта"
            | "эти"
            | "это"
            | "я"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_stable_fnv1a_32() {
        // Canonical FNV-1a/32 test vectors — identical to qlink's, so the
        // persisted index and our live queries hash tokens the same way.
        assert_eq!(hash_token("a"), 0xe40c_292c);
        assert_eq!(hash_token("foobar"), 0xbf9c_f968);
    }

    #[test]
    fn tokenizer_lowercases_splits_and_counts_tf() {
        let (ids, vals) = to_sparse_parts("Алматы, алматы!");
        assert_eq!(ids.len(), 1);
        assert_eq!(vals, vec![2.0]);
    }

    #[test]
    fn stopwords_are_dropped() {
        let (ids, _) = to_sparse_parts("магазин и в на Алматы");
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn empty_text_yields_empty_sparse_vector() {
        let (ids, vals) = to_sparse_parts("   ,.! и в ");
        assert!(ids.is_empty());
        assert!(vals.is_empty());
    }

    #[test]
    fn english_word_matches_expected_hash() {
        // A single english word -> one term id (its FNV-1a hash), tf 1.
        let (ids, vals) = to_sparse_parts("freedom");
        assert_eq!(ids, vec![hash_token("freedom")]);
        assert_eq!(vals, vec![1.0]);
    }
}
