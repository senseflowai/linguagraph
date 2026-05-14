//! MATCH clause emission.
//!
//! Each traversal either *chains* onto the previous one (continuing the
//! same path expression) or starts a *new* MATCH that references its
//! `from_alias` by name only. Two siblings of the start node thus
//! correctly become two MATCH clauses, not one impossible chained path.

use std::fmt::Write;

use crate::ast::query::*;

use super::cursor::Cursor;

pub(super) fn write_match(cur: &mut Cursor, q: &ReadQuery) {
    cur.buf.push_str("MATCH ");
    write_node(cur, &q.start);

    // Track the alias we just emitted as the rightmost endpoint. When the
    // next traversal's `from_alias` matches it, we keep extending the same
    // pattern; otherwise we close the current MATCH and open a new one
    // that re-uses the bound alias by name.
    let mut current_endpoint = q.start.alias.clone();

    for t in &q.traversals {
        if t.optional {
            continue;
        }
        if t.from_alias != current_endpoint {
            cur.buf.push_str("\nMATCH ");
            // Reference the already-bound alias without repeating its
            // label — Cypher resolves it from the surrounding scope.
            let _ = write!(cur.buf, "({})", t.from_alias);
        }
        write_traversal_tail(cur, t);
        current_endpoint = t.target.alias.clone();
    }
}

pub(super) fn write_optional_matches(cur: &mut Cursor, q: &ReadQuery) {
    for t in q.traversals.iter().filter(|t| t.optional) {
        cur.buf.push_str("\nOPTIONAL MATCH ");
        let _ = write!(cur.buf, "({})", t.from_alias);
        write_traversal_tail(cur, t);
    }
}

fn write_node(cur: &mut Cursor, node: &Node) {
    // An empty `label` means "any node" — used by `TraversalQuery`
    // when the caller doesn't pin the entity type. In Cypher,
    // `MATCH (e)` matches a node of any label, so we just drop the
    // `:Label` suffix instead of emitting `(:)` (which would be a
    // syntax error).
    if node.label.is_empty() {
        let _ = write!(cur.buf, "({})", node.alias);
    } else {
        let _ = write!(cur.buf, "({}:{})", node.alias, node.label);
    }
}

/// The `-[edge]->(target)` portion of a traversal — written *after* the
/// source node (which the caller has already emitted, either as the
/// labeled start node or as the bare alias reference).
fn write_traversal_tail(cur: &mut Cursor, t: &EdgeTraversal) {
    let depth = match t.depth {
        Some(d) if d.min == 1 && d.max == 1 => String::new(),
        Some(d) => format!("*{}..{}", d.min, d.max),
        None => String::new(),
    };
    let body = format!("[{}:{}{}]", t.edge_alias, t.edge_label, depth);
    let arrow = match t.direction {
        Direction::Out => format!("-{}->", body),
        Direction::In => format!("<-{}-", body),
        Direction::Both => format!("-{}-", body),
    };
    cur.buf.push_str(&arrow);
    write_node(cur, &t.target);
}
