//! Crate-wide error type.
//!
//! Each layer (`dsl`, `ast`, `builder`, `db`, `config`) exposes its own typed
//! error. They all converge into [`Error`] at the orchestration boundary so
//! the CLI and library callers only need to match a single enum.

use thiserror::Error;

/// Top-level result alias used across the public API.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("DSL error: {0}")]
    Dsl(#[from] crate::dsl::DslError),

    #[error("AST error: {0}")]
    Ast(#[from] crate::ast::AstError),

    #[error("Cypher builder error: {0}")]
    Builder(#[from] crate::builder::BuilderError),

    #[error("Insert builder error: {0}")]
    InsertBuilder(#[from] crate::builder::InsertError),

    #[error("Mapper error: {0}")]
    Mapper(#[from] crate::mapper::MapperError),

    #[error("Graph build error: {0}")]
    GraphBuild(#[from] crate::graph::GraphBuildError),

    #[error("Ingest error: {0}")]
    Ingest(#[from] crate::ingest::IngestError),

    #[error("Database error: {0}")]
    Db(#[from] crate::db::DbError),

    #[error("Configuration error: {0}")]
    Config(#[from] crate::config::ConfigError),

    #[error("Ontology error: {0}")]
    Ontology(#[from] crate::graph::OntologyError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}
