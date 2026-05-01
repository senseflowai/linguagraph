//! Internal query AST.
//!
//! The AST is the *single source of truth* for every downstream consumer
//! (Cypher builder, future planners, explain printers). It is intentionally
//! decoupled from the JSON DSL so we can evolve either side independently.

pub mod from_dsl;
pub mod query;

pub use from_dsl::AstError;
pub use query::*;
