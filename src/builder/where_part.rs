//! WHERE clause emission. Every value is bound as a parameter.

use crate::ast::query::*;

use super::cursor::Cursor;

pub(super) fn write_where(cur: &mut Cursor, expr: &FilterExpression) {
    cur.buf.push_str("\nWHERE ");
    write_expr(cur, expr);
}

fn write_expr(cur: &mut Cursor, expr: &FilterExpression) {
    match expr {
        FilterExpression::Predicate(p) => write_predicate(cur, p),
        FilterExpression::And(parts) => write_joined(cur, parts, " AND "),
        FilterExpression::Or(parts) => write_joined(cur, parts, " OR "),
        FilterExpression::Not(inner) => {
            cur.buf.push_str("NOT (");
            write_expr(cur, inner);
            cur.buf.push(')');
        }
    }
}

fn write_joined(cur: &mut Cursor, parts: &[FilterExpression], sep: &str) {
    if parts.is_empty() {
        return;
    }
    cur.buf.push('(');
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            cur.buf.push_str(sep);
        }
        write_expr(cur, p);
    }
    cur.buf.push(')');
}

fn write_predicate(cur: &mut Cursor, p: &Predicate) {
    let lhs = render_property(&p.field);
    let placeholder = cur.bind(p.value.clone());
    let rendered = match p.op {
        ComparisonOp::Eq => format!("{lhs} = {placeholder}"),
        ComparisonOp::Neq => format!("{lhs} <> {placeholder}"),
        ComparisonOp::Gt => format!("{lhs} > {placeholder}"),
        ComparisonOp::Gte => format!("{lhs} >= {placeholder}"),
        ComparisonOp::Lt => format!("{lhs} < {placeholder}"),
        ComparisonOp::Lte => format!("{lhs} <= {placeholder}"),
        ComparisonOp::In => format!("{lhs} IN {placeholder}"),
        ComparisonOp::Contains => format!("{lhs} CONTAINS {placeholder}"),
        ComparisonOp::StartsWith => format!("{lhs} STARTS WITH {placeholder}"),
        ComparisonOp::EndsWith => format!("{lhs} ENDS WITH {placeholder}"),
    };
    cur.buf.push_str(&rendered);
}

pub(super) fn render_property(p: &PropertyRef) -> String {
    match &p.property {
        Some(prop) => format!("{}.{}", p.alias, prop),
        None => p.alias.to_string(),
    }
}
