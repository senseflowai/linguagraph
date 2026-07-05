//! Id-scoped node / relationship lookups.
//!
//! The graph DSL selects by *type and property*, never by a database
//! internal id, so the "open this exact node / edge" inspector actions a
//! `{nodes, edges}` UI needs can't be expressed as a [`DslQuery`]. These
//! methods issue small, parameterized Cypher lookups directly against the
//! pipeline's client instead. They return the raw [`QueryResult`]; shaping
//! into DTOs lives in [`crate::service::convert`].

use std::collections::BTreeMap;

use crate::ast::query::Literal;
use crate::builder::CypherQuery;
use crate::core::Pipeline;
use crate::db::QueryResult;
use crate::error::Result;
use crate::graph::{MENTION_REL, PART_OF_REL, SOURCE_LABEL};

impl Pipeline {
    /// Fetch a single entity by its database internal id, together with
    /// its immediate relationships and reachable `:Source` documents.
    ///
    /// The result has one row (or none, when the id is unknown) with the
    /// columns `id`, `labels`, `props`, `sources`, and `relations` (a
    /// list of relationship maps). Uses pattern comprehensions so a node
    /// with no relationships yields an empty list rather than a row of
    /// nulls.
    pub async fn entity_detail(&self, id: i64) -> Result<QueryResult> {
        let text = format!(
            "MATCH (n) WHERE id(n) = $id\n\
             RETURN id(n) AS id, labels(n) AS labels, n {{.*}} AS props,\n\
             \x20 [(n)-[:{MENTION_REL}|{PART_OF_REL}]->(__s:{SOURCE_LABEL}) | __s {{.*}}] AS sources,\n\
             \x20 [(n)-[r]-(m) | {{id: id(r), type: type(r), from: id(startNode(r)), \
             to: id(endNode(r)), other_id: id(m), other_labels: labels(m), \
             other_props: m {{.*}}, props: r {{.*}}}}] AS relations"
        );
        self.execute_by_id(text, id).await
    }

    /// Fetch a single relationship by its database internal id, with both
    /// endpoints' ids, labels and properties. Direction is `from` →
    /// `to` (start → end node). At most one row.
    pub async fn relation_detail(&self, id: i64) -> Result<QueryResult> {
        let text = "MATCH ()-[r]-() WHERE id(r) = $id\n\
             WITH r LIMIT 1\n\
             RETURN id(r) AS id, type(r) AS type, \
             id(startNode(r)) AS from, id(endNode(r)) AS to, \
             labels(startNode(r)) AS from_labels, startNode(r) {.*} AS from_props, \
             labels(endNode(r)) AS to_labels, endNode(r) {.*} AS to_props, \
             r {.*} AS props"
            .to_string();
        self.execute_by_id(text, id).await
    }

    async fn execute_by_id(&self, text: String, id: i64) -> Result<QueryResult> {
        let mut params = BTreeMap::new();
        params.insert("id".to_string(), Literal::Int(id));
        let query = CypherQuery::new(text, params);
        Ok(self.client().execute(&query).await?)
    }
}
