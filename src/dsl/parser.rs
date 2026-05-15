//! Parse JSON into a [`DslQuery`] and run cheap structural checks.
//!
//! Validation here is purely syntactic: identifier shape, no duplicate
//! aliases, sane depth ranges. Anything that needs to know about the wider
//! query (alias resolution, group_by/aggregate consistency) is the job of
//! the lowering step in [`crate::ast::from_dsl`].

use std::collections::HashSet;
use std::path::Path;

use thiserror::Error;
use tokio::fs;

use super::schema::*;

#[derive(Debug, Error)]
pub enum DslError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),

    #[error("I/O error reading DSL: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid identifier '{0}': must match [A-Za-z_][A-Za-z0-9_]*")]
    InvalidIdentifier(String),

    #[error("invalid qualified field '{0}': expected `<alias>.<property>` or `<alias>`")]
    InvalidFieldRef(String),

    #[error("duplicate alias '{0}'")]
    DuplicateAlias(String),

    #[error("invalid depth range {min}..{max}: min must be >= 1 and max >= min")]
    InvalidDepth { min: u32, max: u32 },

    #[error("limit must be > 0")]
    InvalidLimit,

    #[error("`return` clause must contain at least one item")]
    EmptyReturn,

    #[error("unknown plain filter op '{0}'; if this is a typed op, set the filter's `type` field")]
    UnknownOp(String),
}

/// Parse and validate a DSL document from a file path.
pub async fn parse(path: &Path) -> Result<DslQuery, DslError> {
    let raw = fs::read_to_string(path).await?;
    parse_str(&raw)
}

/// Parse and validate a DSL document from an in-memory string.
pub fn parse_str(raw: &str) -> Result<DslQuery, DslError> {
    let query: DslQuery = serde_json::from_str(raw)?;
    validate(&query)?;
    Ok(query)
}

fn validate(q: &DslQuery) -> Result<(), DslError> {
    let mut aliases: HashSet<&str> = HashSet::new();

    check_identifier(&q.start.label)?;
    check_identifier(&q.start.alias)?;
    insert_alias(&mut aliases, &q.start.alias)?;

    // A query-wide `prefix_label` is inlined as an extra Cypher label
    // on every node pattern, so it must match the same conservative
    // identifier grammar as a regular label. An empty string is a
    // no-op (treated as None by downstream consumers).
    if let Some(prefix) = q.prefix_label.as_deref() {
        if !prefix.is_empty() {
            check_identifier(prefix)?;
        }
    }

    for t in &q.traversals {
        if let Some(from) = &t.from {
            check_identifier(from)?;
        }
        check_identifier(&t.edge.label)?;
        check_identifier(&t.edge.alias)?;
        insert_alias(&mut aliases, &t.edge.alias)?;

        check_identifier(&t.target.label)?;
        check_identifier(&t.target.alias)?;
        insert_alias(&mut aliases, &t.target.alias)?;

        if let Some(d) = t.depth {
            if d.min < 1 || d.max < d.min {
                return Err(DslError::InvalidDepth {
                    min: d.min,
                    max: d.max,
                });
            }
        }
    }

    for f in &q.filters {
        check_field_ref(&f.field)?;
        // Op validation is deferred to the lowering step, which has
        // both the registry and the graph specification at hand. The
        // parser only checks the op is identifier-shaped — anything
        // beyond that requires knowing whether the filter ends up
        // typed (handler ops) or untyped (plain ops). The lowerer
        // returns a precise `UnknownPlainOp` / `UnsupportedTypedOp`
        // when the resolved op is invalid.
        if f.op.is_empty() || !f.op.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(DslError::UnknownOp(f.op.clone()));
        }
        if let Some(ty) = &f.field_type {
            check_identifier(ty)?;
        }
    }

    if q.return_.is_empty() {
        return Err(DslError::EmptyReturn);
    }
    for item in &q.return_ {
        match item {
            ReturnItem::Field { field, alias } => {
                check_field_ref(field)?;
                if let Some(a) = alias {
                    check_identifier(a)?;
                }
            }
            ReturnItem::Aggregate { field, alias, .. } => {
                check_field_ref(field)?;
                if let Some(a) = alias {
                    check_identifier(a)?;
                }
            }
        }
    }

    for g in &q.group_by {
        check_field_ref(g)?;
    }
    for s in &q.sort {
        check_identifier_or_field(&s.field)?;
    }
    if let Some(0) = q.limit {
        return Err(DslError::InvalidLimit);
    }

    Ok(())
}

fn insert_alias<'a>(set: &mut HashSet<&'a str>, alias: &'a str) -> Result<(), DslError> {
    if !set.insert(alias) {
        return Err(DslError::DuplicateAlias(alias.to_string()));
    }
    Ok(())
}

fn check_identifier(s: &str) -> Result<(), DslError> {
    let mut chars = s.chars();
    let first = chars.next();
    let ok = matches!(first, Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ok {
        return Err(DslError::InvalidIdentifier(s.to_string()));
    }
    Ok(())
}

/// Field references are either `alias` or `alias.property`. Both halves must
/// be valid identifiers.
fn check_field_ref(s: &str) -> Result<(), DslError> {
    let mut parts = s.split('.');
    let alias = parts.next().unwrap_or("");
    let prop = parts.next();
    if parts.next().is_some() || alias.is_empty() {
        return Err(DslError::InvalidFieldRef(s.to_string()));
    }
    check_identifier(alias)?;
    if let Some(p) = prop {
        check_identifier(p)?;
    }
    Ok(())
}

/// Sort can reference either a return alias or a qualified field.
fn check_identifier_or_field(s: &str) -> Result<(), DslError> {
    if s.contains('.') {
        check_field_ref(s)
    } else {
        check_identifier(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_find() {
        let json = r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "return": [{ "field": "p.name" }]
        }"#;
        let q = parse_str(json).unwrap();
        assert_eq!(q.action, Action::Find);
        assert_eq!(q.start.label, "Person");
    }

    #[test]
    fn rejects_duplicate_alias() {
        let json = r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "traversals": [{
                "edge": { "label": "KNOWS", "alias": "r", "direction": "out" },
                "target": { "label": "Person", "alias": "p" }
            }],
            "return": [{ "field": "p.name" }]
        }"#;
        let err = parse_str(json).unwrap_err();
        assert!(matches!(err, DslError::DuplicateAlias(_)));
    }

    #[test]
    fn rejects_bad_identifier() {
        let json = r#"{
            "action": "find",
            "start": { "label": "1Person", "alias": "p" },
            "return": [{ "field": "p.name" }]
        }"#;
        assert!(matches!(
            parse_str(json),
            Err(DslError::InvalidIdentifier(_))
        ));
    }

    #[test]
    fn rejects_bad_depth() {
        let json = r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "traversals": [{
                "edge": { "label": "KNOWS", "alias": "r", "direction": "out" },
                "target": { "label": "Person", "alias": "p2" },
                "depth": { "min": 0, "max": 3 }
            }],
            "return": [{ "field": "p.name" }]
        }"#;
        assert!(matches!(
            parse_str(json),
            Err(DslError::InvalidDepth { .. })
        ));
    }

    fn multiple_traversal() {
        let json = r#"{
          "action": "aggregate",
          "start": { "label": "Camera", "alias": "c" },
          "traversals": [
            { "edge": { "label": "HAS_STORAGE", "alias": "s", "direction": "out" }, "target": { "label": "Storage", "alias": "st" } },
            { "edge": { "label": "LOCATED_IN", "alias": "l", "direction": "out" }, "target": { "label": "Place", "alias": "p" } }
          ],
          "filters": [
            { "field": "c.state", "op": "eq", "value": "active" },
            { "field": "st.depth", "op": "eq", "value": 30 },
            { "field": "p.name", "op": "eq", "value": "Office" }
          ],
          "return": [
            { "aggregate": "count", "field": "c.id", "alias": "count" }
          ],
          "limit": 1
        }"#;

        let _q = parse_str(json).unwrap();
    }

    #[test]
    fn rejects_empty_return() {
        let json = r#"{
            "action": "find",
            "start": { "label": "Person", "alias": "p" },
            "return": []
        }"#;
        assert!(matches!(parse_str(json), Err(DslError::EmptyReturn)));
    }
}
