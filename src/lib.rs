//! `linguagraph` — convert JSON-DSL graph queries into Cypher and run them
//! against a Memgraph database.
//!
//! The crate is organized as a stack of single-responsibility layers:
//!
//! | Layer        | Module               | Purpose                                    |
//! |--------------|----------------------|--------------------------------------------|
//! | DSL          | [`dsl`]              | parse + structurally validate JSON         |
//! | AST          | [`ast`]              | typed query model                          |
//! | Builder      | [`builder`]          | AST → parameterized Cypher                 |
//! | DB           | [`db`]               | execute against Memgraph (or a mock)       |
//! | Config       | [`config`]           | TOML + env-var configuration               |
//! | Prompt       | [`prompt`]           | schema-aware system prompts for LLMs       |
//! | Core         | [`core`]              | wires the layers together                  |
//! | CLI          | [`cli`]              | command-line entrypoints                   |
//! | API          | [`api`]              | stable public surface for the REST service |
//!
//! Anything user-facing (the CLI, integration tests) goes through
//! [`core::Pipeline`]; the layers below it are reusable on their own.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod api;
pub mod ast;
pub mod builder;
pub mod cli;
pub mod config;
pub mod core;
pub mod db;
pub mod dsl;
pub mod e2e;
pub mod embeddings;
pub mod error;
pub mod graph;
pub mod ingest;
pub mod llm;
pub mod prompt;
pub mod resolve;
pub mod types;

pub use error::{Error, Result};
