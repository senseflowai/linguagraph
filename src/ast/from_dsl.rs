//! Backwards-compatibility shim for the DSL → AST lowering.
//!
//! The implementation moved to [`crate::resolve`]. This module
//! re-exports the same public surface so existing callers
//! (`linguagraph::ast::from_dsl::lower`, `linguagraph::ast::AstError`)
//! keep working unchanged.
//!
//! New code should import directly from `linguagraph::resolve`.

pub use crate::resolve::{lower, lower_full, lower_with_registry, AstError};
