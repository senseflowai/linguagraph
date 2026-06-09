//! Operations a typed predicate can carry.
//!
//! `TypedOp` is the closed set of *names* that may appear after `"op": …`
//! in a typed DSL filter. The handler decides which subset it accepts;
//! the parser only checks that the name is registered on the type.

use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, strum::Display,
    strum::IntoStaticStr,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
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

    SearchReranked,
    /// Hybrid: combine an exact equality with a vector score.
    #[serde(alias = "hybrid")]
    HybridSearch,
    /// Geo proximity (reserved for future GeoLocation handler).
    Near,
}

impl TypedOp {
    /// The op as a static snake_case string (e.g. `"starts_with"`).
    pub fn as_str(&self) -> &'static str {
        (*self).into()
    }
}
