//! Lexical similarity helpers used as a second-signal gate next to
//! dense embedding similarity.
//!
//! Embedding top-1 is too permissive a signal on its own — different
//! entities with similar property shapes embed close together. Jaro-
//! Winkler over the *primary name line* gives an independent surface-
//! form signal that catches the obvious "totally different name, same
//! type" false-positive.
//!
//! Caveats:
//!  * ASCII case-folded via `to_lowercase()`. Non-ASCII chars are
//!    compared as Unicode scalars (codepoints), which is enough for the
//!    Cyrillic fixtures currently in the test suite but not full NFC.
//!  * We deliberately don't pull in `unicode-normalization` for v1; the
//!    upgrade path is a one-file change here.

/// Jaro-Winkler similarity ∈ [0.0, 1.0].
///
/// Equal strings return 1.0; entirely disjoint strings return 0.0.
/// The implementation follows the standard Jaro algorithm with the
/// Winkler common-prefix boost (scaling factor 0.1, capped at 4
/// characters of prefix), comparing Unicode scalar values directly.
pub(super) fn jaro_winkler(a: &str, b: &str) -> f64 {
    if a == b {
        return if a.is_empty() { 0.0 } else { 1.0 };
    }
    let jaro = jaro(a, b);
    if jaro == 0.0 {
        return 0.0;
    }
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let prefix = a_chars
        .iter()
        .zip(b_chars.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count() as f64;
    jaro + prefix * 0.1 * (1.0 - jaro)
}

fn jaro(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let alen = a.len();
    let blen = b.len();
    if alen == 0 || blen == 0 {
        return 0.0;
    }
    let match_window = alen.max(blen) / 2;
    let match_window = match_window.saturating_sub(1);

    let mut a_matches = vec![false; alen];
    let mut b_matches = vec![false; blen];
    let mut matches = 0usize;
    for (i, ca) in a.iter().enumerate() {
        let start = i.saturating_sub(match_window);
        let end = (i + match_window + 1).min(blen);
        for j in start..end {
            if b_matches[j] {
                continue;
            }
            if *ca != b[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }
    if matches == 0 {
        return 0.0;
    }
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..alen {
        if !a_matches[i] {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a[i] != b[k] {
            transpositions += 1;
        }
        k += 1;
    }
    let m = matches as f64;
    let t = (transpositions / 2) as f64;
    (m / alen as f64 + m / blen as f64 + (m - t) / m) / 3.0
}

/// Score the lexical similarity of an incoming entity's key text
/// against a candidate's same field. For `_canonical`-keyed entities,
/// we extract the primary-name line on both sides before comparing —
/// see [`primary_name_of`] for the rationale.
pub(super) fn lexical_score(incoming: &str, hit_canonical: &str) -> f64 {
    let a = primary_name_of(incoming);
    let b = primary_name_of(hit_canonical);
    jaro_winkler(&a.to_lowercase(), &b.to_lowercase())
}

/// Extract the "primary identifier" line from a `_canonical` text.
///
/// `_canonical` is multi-line `type: X\nkey: value\n...`. Comparing
/// the whole blob dilutes name similarity with noise from other
/// properties (a Person with matching `name: Elon Musk` but differing
/// `age` would score lower than its name surface-form deserves).
///
/// Rules:
///  1. If first non-empty line starts with `type: `, look for a
///     `name: ...` line and return everything after `name: `.
///  2. Otherwise return the first non-`type:` line's value.
///  3. If the text has no `type:` prefix at all (the explicit
///     legacy raw-name path), return the whole text — that's already
///     just a name.
pub(super) fn primary_name_of(canonical: &str) -> &str {
    let mut lines = canonical.lines().filter(|l| !l.trim().is_empty());
    let Some(first) = lines.next() else {
        return canonical;
    };
    if !first.starts_with("type:") {
        return canonical;
    }
    // Look for `name: ...` first.
    for line in canonical.lines() {
        if let Some(rest) = line.strip_prefix("name: ") {
            return rest;
        }
    }
    // Fall back to the first non-`type:` line, stripped of its key prefix.
    for line in canonical.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        if let Some((_, value)) = line.split_once(": ") {
            return value;
        }
        return line;
    }
    first
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jaro_winkler_reference_pairs() {
        // From the Jaro-Winkler reference paper.
        let martha = jaro_winkler("MARTHA", "MARHTA");
        assert!(
            (martha - 0.961).abs() < 0.01,
            "MARTHA/MARHTA expected ~0.961, got {martha}"
        );
        let dixon = jaro_winkler("DIXON", "DICKSONX");
        assert!(
            (dixon - 0.813).abs() < 0.01,
            "DIXON/DICKSONX expected ~0.813, got {dixon}"
        );
    }

    #[test]
    fn equal_strings_score_one() {
        assert_eq!(jaro_winkler("alice", "alice"), 1.0);
    }

    #[test]
    fn disjoint_strings_score_zero() {
        assert_eq!(jaro_winkler("abc", "xyz"), 0.0);
    }

    #[test]
    fn empty_string_pair_scores_zero() {
        assert_eq!(jaro_winkler("", ""), 0.0);
        assert_eq!(jaro_winkler("", "abc"), 0.0);
        assert_eq!(jaro_winkler("abc", ""), 0.0);
    }

    #[test]
    fn unicode_codepoint_comparison_matches_cyrillic_fixture() {
        // Matches the existing Russian-language test fixture in
        // soft_merge integration tests — must score high enough to
        // pass a reasonable lexical gate.
        let s = jaro_winkler("общественное согласие", "общественное соглас.");
        assert!(s >= 0.85, "Cyrillic near-duplicate expected ≥0.85, got {s}");
    }

    #[test]
    fn primary_name_of_extracts_name_line() {
        let canonical = "type: Person\nage: 42\nname: Elon Musk\nrole: CEO";
        assert_eq!(primary_name_of(canonical), "Elon Musk");
    }

    #[test]
    fn primary_name_of_falls_back_to_first_non_type_line() {
        // No `name:` line — use the first key's value as the
        // identifier signal.
        let canonical = "type: Organization\ncountry: US";
        assert_eq!(primary_name_of(canonical), "US");
    }

    #[test]
    fn primary_name_of_passes_through_non_canonical_text() {
        // Legacy raw-name path: value is already a name, not
        // multi-line canonical text. Return as-is.
        assert_eq!(primary_name_of("Alice"), "Alice");
    }

    #[test]
    fn lexical_score_is_case_insensitive_for_ascii() {
        assert_eq!(lexical_score("alice", "ALICE"), 1.0);
    }

    #[test]
    fn lexical_score_unwraps_canonical_on_both_sides() {
        let a = "type: Person\nname: Elon Musk";
        let b = "type: Person\nname: Elon Mask";
        let s = lexical_score(a, b);
        // Both names share 8 of 9 chars; should be quite high.
        assert!(s >= 0.85, "expected ≥0.85 for Elon Musk / Elon Mask, got {s}");
    }
}
