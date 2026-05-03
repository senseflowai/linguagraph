//! Operations a typed predicate can carry.
//!
//! `TypedOp` is the closed set of *names* that may appear after `"op": …`
//! in a typed DSL filter. The handler decides which subset it accepts;
//! the parser only checks that the name is registered on the type.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypedOp {
    // ── Exact / range / containment ─────────────────────────────────
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Contains,
    StartsWith,
    EndsWith,
    // ── Semantic / hybrid / geo ─────────────────────────────────────
    /// Pure vector search: `qlink.search` with the input embedded.
    Search,
    /// Hybrid: combine an exact equality with a vector score.
    #[serde(alias = "hybrid")]
    HybridSearch,
    /// Geo proximity (reserved for future GeoLocation handler).
    Near,
}

impl TypedOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Neq => "neq",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lt => "lt",
            Self::Lte => "lte",
            Self::In => "in",
            Self::Contains => "contains",
            Self::StartsWith => "starts_with",
            Self::EndsWith => "ends_with",
            Self::Search => "search",
            Self::HybridSearch => "hybrid_search",
            Self::Near => "near",
        }
    }
}

impl std::fmt::Display for TypedOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
