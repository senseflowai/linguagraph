//! MATCH clause emission.

use std::fmt::Write;

use crate::ast::query::*;

use super::cursor::Cursor;

pub(super) fn write_match(cur: &mut Cursor, q: &Query) {
    cur.buf.push_str("MATCH ");
    write_node(cur, &q.start);
    for t in &q.traversals {
        write_traversal(cur, t);
    }
}

fn write_node(cur: &mut Cursor, node: &Node) {
    let _ = write!(cur.buf, "({}:{})", node.alias, node.label);
}

fn write_traversal(cur: &mut Cursor, t: &EdgeTraversal) {
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
