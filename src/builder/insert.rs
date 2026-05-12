//! Compile an [`InsertQuery`] into one [`CypherQuery`] per batch.
//!
//! Every batch is rendered as a single `UNWIND $rows AS row …` statement.
//! All values are bound as a single object/list parameter — labels and
//! property names appear inline in the template but are validated against
//! a strict identifier grammar so a hostile mapping cannot smuggle Cypher.

use std::collections::BTreeMap;

use thiserror::Error;

use crate::ast::query::*;

use super::cursor::CypherQuery;

#[derive(Debug, Error)]
pub enum InsertError {
    #[error("invalid Cypher identifier '{0}': labels and property names must match [A-Za-z_][A-Za-z0-9_]*")]
    InvalidIdentifier(String),

    #[error("node row missing primary-key value for label '{label}'")]
    MissingId { label: String },

    #[error("node batch '{0}' has zero rows; planner should not emit empty batches")]
    EmptyBatch(String),
}

/// Render every batch in `query`. Empty inputs yield an empty `Vec`.
pub fn build_insert(query: &InsertQuery) -> Result<Vec<CypherQuery>, InsertError> {
    let mut out = Vec::with_capacity(query.node_batches.len() + query.relation_batches.len());

    for batch in &query.node_batches {
        out.push(render_node_batch(batch)?);
    }
    for batch in &query.relation_batches {
        out.push(render_relation_batch(batch)?);
    }

    Ok(out)
}

fn render_node_batch(batch: &NodeBatch) -> Result<CypherQuery, InsertError> {
    if batch.rows.is_empty() {
        return Err(InsertError::EmptyBatch(batch.label.clone()));
    }
    check_ident(&batch.label)?;
    check_ident(&batch.merge_on)?;

    // Each row becomes `{ id: <pk>, props: { ...other props... } }`. The
    // builder is the single place that knows the row layout — the planner
    // only fills it.
    let mut rows = Vec::with_capacity(batch.rows.len());
    for row in &batch.rows {
        if matches!(row.id, Literal::Null) {
            return Err(InsertError::MissingId {
                label: batch.label.clone(),
            });
        }
        for k in row.props.keys() {
            check_ident(k)?;
        }
        let mut entry: BTreeMap<String, Literal> = BTreeMap::new();
        entry.insert("id".to_string(), row.id.clone());
        entry.insert("props".to_string(), Literal::Object(row.props.clone()));
        rows.push(Literal::Object(entry));
    }

    let text = format!(
        "UNWIND $rows AS row\n\
         MERGE (n:{label} {{{key}: row.id}})\n\
         SET n += row.props",
        label = batch.label,
        key = batch.merge_on,
    );

    let mut params = BTreeMap::new();
    params.insert("rows".to_string(), Literal::List(rows));

    Ok(CypherQuery::new(text, params))
}

fn render_relation_batch(batch: &RelationBatch) -> Result<CypherQuery, InsertError> {
    if batch.rows.is_empty() {
        return Err(InsertError::EmptyBatch(batch.rel_type.clone()));
    }
    check_ident(&batch.rel_type)?;
    check_ident(&batch.from_label)?;
    check_ident(&batch.to_label)?;
    check_ident(&batch.from_key)?;
    check_ident(&batch.to_key)?;

    let mut rows = Vec::with_capacity(batch.rows.len());
    for row in &batch.rows {
        let mut entry: BTreeMap<String, Literal> = BTreeMap::new();
        for k in row.props.keys() {
            check_ident(k)?;
        }
        entry.insert("from".to_string(), row.from_id.clone());
        entry.insert("to".to_string(), row.to_id.clone());
        entry.insert("props".to_string(), Literal::Object(row.props.clone()));
        rows.push(Literal::Object(entry));
    }

    let text = format!(
        "UNWIND $rels AS rel\n\
         MATCH (a:{from_label} {{{from_key}: rel.from}})\n\
         MATCH (b:{to_label} {{{to_key}: rel.to}})\n\
         MERGE (a)-[r:{rel_type}]->(b)\n\
         SET r += rel.props",
        from_label = batch.from_label,
        from_key = batch.from_key,
        to_label = batch.to_label,
        to_key = batch.to_key,
        rel_type = batch.rel_type,
    );

    let mut params = BTreeMap::new();
    params.insert("rels".to_string(), Literal::List(rows));

    Ok(CypherQuery::new(text, params))
}

/// Cypher identifiers we render inline (labels, property names, relation
/// types) must match a conservative grammar. Anything else gets rejected
/// here so a malformed mapping can never produce executable SQL/Cypher.
fn check_ident(s: &str) -> Result<(), InsertError> {
    let mut chars = s.chars();
    let first = chars.next();
    let ok = matches!(first, Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok {
        return Err(InsertError::InvalidIdentifier(s.to_string()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> Literal {
        Literal::String(v.into())
    }

    #[test]
    fn renders_node_batch() {
        let mut props = BTreeMap::new();
        props.insert("name".into(), s("cam-1"));
        props.insert("state".into(), s("active"));

        let batch = NodeBatch {
            label: "Camera".into(),
            merge_on: "id".into(),
            rows: vec![NodeRow { id: s("c1"), props }],
        };

        let out = render_node_batch(&batch).unwrap();
        assert!(out.text.contains("UNWIND $rows AS row"));
        assert!(out.text.contains("MERGE (n:Camera {id: row.id})"));
        assert!(out.text.contains("SET n += row.props"));

        // The full row payload is one parameter; nothing leaks inline.
        assert!(out.params.contains_key("rows"));
        assert!(!out.text.contains("cam-1"));
        assert!(!out.text.contains("active"));
    }

    #[test]
    fn renders_relation_batch() {
        let batch = RelationBatch {
            rel_type: "LOCATED_IN".into(),
            from_label: "Camera".into(),
            from_key: "id".into(),
            to_label: "Place".into(),
            to_key: "id".into(),
            rows: vec![RelationRow {
                from_id: s("c1"),
                to_id: s("p1"),
                props: BTreeMap::new(),
            }],
        };
        let out = render_relation_batch(&batch).unwrap();
        assert!(out.text.contains("UNWIND $rels AS rel"));
        assert!(out.text.contains("MATCH (a:Camera {id: rel.from})"));
        assert!(out.text.contains("MATCH (b:Place {id: rel.to})"));
        assert!(out.text.contains("MERGE (a)-[r:LOCATED_IN]->(b)"));
        assert!(out.text.contains("SET r += rel.props"));
    }

    #[test]
    fn rejects_bad_label() {
        let batch = NodeBatch {
            label: "1Bad".into(),
            merge_on: "id".into(),
            rows: vec![NodeRow {
                id: s("x"),
                props: BTreeMap::new(),
            }],
        };
        assert!(matches!(
            render_node_batch(&batch),
            Err(InsertError::InvalidIdentifier(_))
        ));
    }

    #[test]
    fn rejects_injection_attempt_in_label() {
        let batch = NodeBatch {
            label: "Camera) MATCH (x".into(),
            merge_on: "id".into(),
            rows: vec![NodeRow {
                id: s("x"),
                props: BTreeMap::new(),
            }],
        };
        assert!(matches!(
            render_node_batch(&batch),
            Err(InsertError::InvalidIdentifier(_))
        ));
    }

    #[test]
    fn rejects_null_id() {
        let batch = NodeBatch {
            label: "Camera".into(),
            merge_on: "id".into(),
            rows: vec![NodeRow {
                id: Literal::Null,
                props: BTreeMap::new(),
            }],
        };
        assert!(matches!(
            render_node_batch(&batch),
            Err(InsertError::MissingId { .. })
        ));
    }

    #[test]
    fn empty_query_yields_no_batches() {
        let q = InsertQuery {
            node_batches: vec![],
            relation_batches: vec![],
        };
        assert!(build_insert(&q).unwrap().is_empty());
    }
}
