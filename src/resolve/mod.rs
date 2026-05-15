//! DSL → AST resolution.
//!
//! The resolver is the only place stringly-typed DSL concepts
//! (`alias.property` field refs, JSON literals, type-name strings,
//! op-name strings) cross into the typed AST domain. By the time
//! [`lower_full`] returns, every alias resolves, every literal is
//! typed, every filter has been matched to either a built-in op or a
//! registered [`crate::types::TypeHandler`], and aggregation rules
//! (`Find` vs `Aggregate`, `group_by` requirements) are enforced.
//!
//! This used to live as `ast/from_dsl.rs`; the move keeps the AST
//! itself a pure data model and gives this validation/lowering pass
//! its own home where future passes (alias interning, normalization,
//! source-projection injection) can grow alongside it.

mod ast;

pub use ast::{lower, lower_full, lower_with_registry, AstError};
