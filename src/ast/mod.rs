//! Internal query AST.
//!
//! The AST is the *single source of truth* for every downstream consumer
//! (Cypher builder, future planners, explain printers). It is intentionally
//! decoupled from the JSON DSL so we can evolve either side independently.
//!
//! DSL → AST resolution lives in [`crate::resolve`]. Existing callers
//! that referenced `ast::from_dsl` continue to work via a small
//! compatibility shim.

pub mod from_dsl;
pub mod query;

pub use crate::resolve::AstError;
pub use query::*;
