//! Output buffer + parameter table threaded through every builder stage.

use std::collections::BTreeMap;

use crate::ast::query::Literal;
use crate::types::context::ParamBinder;

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
///
/// Beyond the main `buf`/`params`, the cursor accumulates *contributions*
/// from type handlers:
///
/// * `pre_match` — lines spliced **before** the user's MATCH (e.g. a
///   `CALL qlink.search(...) YIELD id, score` that a SemanticText
///   handler asks for).
/// * `post_match` — lines spliced **after** the user's WHERE (e.g.
///   the hybrid-scoring CALL chain).
/// * `extra_order_by` — sort keys appended **after** the user's
///   ORDER BY so explicit user sort still wins.
///
/// Handlers populate these through the [`crate::types::EmitCtx`]
/// wrapper rather than touching the cursor directly; this keeps the
/// cursor's invariants (deterministic param naming, no inline literals)
/// intact.
pub(super) struct Cursor {
    pub buf: String,
    pub params: BTreeMap<String, Literal>,
    pub pre_match: Vec<String>,
    pub post_match: Vec<String>,
    pub extra_order_by: Vec<(String, crate::types::context::OrderDir)>,
    next_id: usize,
}

impl Cursor {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            params: BTreeMap::new(),
            pre_match: Vec::new(),
            post_match: Vec::new(),
            extra_order_by: Vec::new(),
            next_id: 0,
        }
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

    /// Run a handler's `emit` against this cursor.
    ///
    /// Threads the cursor in as a [`ParamBinder`] (so handlers bind
    /// values into the same param table the rest of the builder uses)
    /// and drains the resulting prelude/with/order-by contributions
    /// into the cursor's accumulators. Returns the boolean expression
    /// the handler wants AND-ed into WHERE, if any.
    pub fn run_handler<F>(&mut self, f: F) -> Result<Option<String>, crate::types::TypeError>
    where
        F: FnOnce(&mut crate::types::EmitCtx<'_>) -> Result<(), crate::types::TypeError>,
    {
        use crate::types::context::CypherContribution;
        let mut contrib = CypherContribution::default();
        let result;
        {
            let binder: &mut dyn ParamBinder = self;
            let mut ctx = crate::types::EmitCtx::new(&mut contrib, binder);
            result = f(&mut ctx);
        }
        let where_expr = contrib.where_inline.clone();
        self.pre_match.extend(contrib.pre_match);
        self.post_match.extend(contrib.post_match);
        self.extra_order_by.extend(contrib.order_by);
        result.map(|_| where_expr)
    }
}

/// `ParamBinder` impl so type handlers can use the same parameter
/// table the rest of the builder writes into.
impl ParamBinder for Cursor {
    fn bind(&mut self, value: Literal) -> String {
        Cursor::bind(self, value)
    }
}
