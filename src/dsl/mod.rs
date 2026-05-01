//! JSON DSL: the externally visible surface for graph queries.
//!
//! The DSL is intentionally narrow — it exists so an LLM can emit something
//! we can validate cheaply and reject before it ever touches the database.
//! All structural rules live in this module; semantic rules (alias resolution,
//! property typing) live in [`crate::ast`].

pub mod parser;
pub mod schema;

pub use parser::{parse, parse_str, DslError};
pub use schema::*;
