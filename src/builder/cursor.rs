//! Output buffer + parameter table threaded through every builder stage.

use std::collections::BTreeMap;

use crate::ast::query::Literal;

/// A finished Cypher query with its bound parameters.
///
/// `BTreeMap` is used so the textual representation is deterministic, which
/// makes snapshot tests stable.
#[derive(Debug, Clone, PartialEq)]
pub struct CypherQuery {
    pub text: String,
    pub params: BTreeMap<String, Literal>,
}

impl CypherQuery {
    pub fn new(text: String, params: BTreeMap<String, Literal>) -> Self {
        Self { text, params }
    }
}

/// Mutable builder state. Only the [`cypher`](super::cypher) entrypoint
/// constructs and finalizes one of these.
pub(super) struct Cursor {
    pub buf: String,
    pub params: BTreeMap<String, Literal>,
    next_id: usize,
}

impl Cursor {
    pub fn new() -> Self {
        Self { buf: String::new(), params: BTreeMap::new(), next_id: 0 }
    }

    /// Bind `value` to a fresh `$pN` placeholder and return the placeholder
    /// reference (with leading `$`).
    pub fn bind(&mut self, value: Literal) -> String {
        let name = format!("p{}", self.next_id);
        self.next_id += 1;
        self.params.insert(name.clone(), value);
        format!("${name}")
    }

    pub fn finish(self) -> CypherQuery {
        CypherQuery::new(self.buf, self.params)
    }
}
