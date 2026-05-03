//! Capability flags advertised by [`TypeHandler`]s.
//!
//! Capabilities exist for three reasons:
//!
//! 1. **Validation**: the DSL parser uses them to reject `{"type": "Keyword",
//!    "op": "search"}` early instead of letting it crash inside an
//!    `emit` somewhere.
//! 2. **DSL generation**: the prompt advertises only ops a type actually
//!    supports, so the LLM can't ask for impossible things.
//! 3. **Pipeline introspection**: callers can ask "does this query touch
//!    a type that needs an embedder?" without instantiating one
//!    speculatively.
//!
//! We use a hand-rolled bitset rather than `bitflags` to keep the
//! dependency graph small. The flag set is tiny and unlikely to grow
//! beyond what fits in a `u16`.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::TypedOp;

/// Bitset of capabilities a type advertises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Capabilities(u16);

impl Capabilities {
    pub const NONE: Self = Capabilities(0);
    pub const INGEST: Self = Capabilities(1 << 0);
    pub const EXACT_MATCH: Self = Capabilities(1 << 1);
    pub const SEMANTIC_SEARCH: Self = Capabilities(1 << 2);
    pub const HYBRID_SEARCH: Self = Capabilities(1 << 3);
    pub const GEO_SEARCH: Self = Capabilities(1 << 4);
    pub const RANGE: Self = Capabilities(1 << 5);
    pub const CONTAINS: Self = Capabilities(1 << 6);

    pub const fn empty() -> Self {
        Capabilities(0)
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn union(self, other: Self) -> Self {
        Capabilities(self.0 | other.0)
    }

    pub fn iter(self) -> impl Iterator<Item = Capabilities> {
        const ALL: &[Capabilities] = &[
            Capabilities::INGEST,
            Capabilities::EXACT_MATCH,
            Capabilities::SEMANTIC_SEARCH,
            Capabilities::HYBRID_SEARCH,
            Capabilities::GEO_SEARCH,
            Capabilities::RANGE,
            Capabilities::CONTAINS,
        ];
        ALL.iter().copied().filter(move |c| self.contains(*c))
    }

    /// Default DSL ops implied by a capability set. Handlers can return
    /// a subset/superset by overriding [`super::TypeHandler::supported_ops`].
    pub fn default_ops(self) -> Vec<TypedOp> {
        let mut out = Vec::new();
        if self.contains(Self::EXACT_MATCH) {
            out.extend([TypedOp::Eq, TypedOp::Neq, TypedOp::In]);
        }
        if self.contains(Self::RANGE) {
            out.extend([TypedOp::Gt, TypedOp::Gte, TypedOp::Lt, TypedOp::Lte]);
        }
        if self.contains(Self::CONTAINS) {
            out.extend([TypedOp::Contains, TypedOp::StartsWith, TypedOp::EndsWith]);
        }
        if self.contains(Self::SEMANTIC_SEARCH) {
            out.push(TypedOp::Search);
        }
        if self.contains(Self::HYBRID_SEARCH) {
            out.push(TypedOp::HybridSearch);
        }
        if self.contains(Self::GEO_SEARCH) {
            out.push(TypedOp::Near);
        }
        out
    }
}

impl std::ops::BitOr for Capabilities {
    type Output = Capabilities;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for Capabilities {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl fmt::Display for Capabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&'static str> = self.iter().map(name).collect();
        if names.is_empty() {
            f.write_str("(none)")
        } else {
            f.write_str(&names.join(" | "))
        }
    }
}

fn name(c: Capabilities) -> &'static str {
    match c {
        Capabilities::INGEST => "ingest",
        Capabilities::EXACT_MATCH => "exact_match",
        Capabilities::SEMANTIC_SEARCH => "semantic_search",
        Capabilities::HYBRID_SEARCH => "hybrid_search",
        Capabilities::GEO_SEARCH => "geo_search",
        Capabilities::RANGE => "range",
        Capabilities::CONTAINS => "contains",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_basics() {
        let c = Capabilities::INGEST | Capabilities::SEMANTIC_SEARCH;
        assert!(c.contains(Capabilities::INGEST));
        assert!(c.contains(Capabilities::SEMANTIC_SEARCH));
        assert!(!c.contains(Capabilities::EXACT_MATCH));
    }

    #[test]
    fn iter_yields_only_set_flags() {
        let c = Capabilities::EXACT_MATCH | Capabilities::HYBRID_SEARCH;
        let v: Vec<_> = c.iter().collect();
        assert_eq!(v.len(), 2);
        assert!(v.contains(&Capabilities::EXACT_MATCH));
        assert!(v.contains(&Capabilities::HYBRID_SEARCH));
    }

    #[test]
    fn default_ops_match_capability_set() {
        let c = Capabilities::EXACT_MATCH | Capabilities::SEMANTIC_SEARCH;
        let ops = c.default_ops();
        assert!(ops.contains(&TypedOp::Eq));
        assert!(ops.contains(&TypedOp::Search));
        assert!(!ops.contains(&TypedOp::HybridSearch));
    }
}
