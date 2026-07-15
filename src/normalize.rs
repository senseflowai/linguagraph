//! Unicode-aware string normalization used for case-insensitive matching.
//!
//! Memgraph's `toLower()` is ASCII-oriented in practice for our Cyrillic
//! fixtures, so matching must not depend on database-side case folding.

use crate::ast::query::Literal;

pub(crate) const NORMALIZED_PROPERTY_PREFIX: &str = "_lg_norm_";

pub(crate) fn normalized_property_name(prop: &str) -> String {
    format!("{NORMALIZED_PROPERTY_PREFIX}{prop}")
}

pub(crate) fn normalize_for_match(s: &str) -> String {
    s.trim().to_lowercase()
}

pub(crate) fn normalize_literal_for_match(value: Literal) -> Literal {
    match value {
        Literal::String(s) => Literal::String(normalize_for_match(&s)),
        Literal::List(items) => {
            Literal::List(items.into_iter().map(normalize_literal_for_match).collect())
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_cyrillic_case() {
        assert_eq!(normalize_for_match(" Медеуский район "), "медеуский район");
    }
}
