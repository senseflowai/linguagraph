//! AST → Cypher compilation.
//!
//! The builder is intentionally split into single-purpose pieces
//! ([`match_part`], [`where_part`], [`return_part`], …). Each one writes into
//! a shared [`Cursor`] which holds the growing query string and a parameter
//! map. Values **never** make it into the string — they are bound as `$pN`
//! parameters that the database driver renders safely.

mod cursor;
mod cypher;
mod insert;
mod match_part;
mod return_part;
mod where_part;

pub use cursor::CypherQuery;
pub use cypher::{build, build_read, build_read_with, compile, compile_with, BuilderError};
pub use insert::{build_insert, InsertError};
pub use where_part::WhereError;
