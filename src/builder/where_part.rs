//! WHERE clause emission. Every value is bound as a parameter.
//!
//! Typed predicates (carried in [`FilterExpression::Typed`]) are routed
//! through the [`crate::types::TypeRegistry`]; the matching handler
//! decides what Cypher fragments to emit and what boolean expression
//! to splice here. The builder itself never inspects the type id.

use crate::ast::query::*;
use crate::types::TypeRegistry;

use super::cursor::Cursor;
use super::cypher::BuilderError;

pub(super) fn write_where(
    cur: &mut Cursor,
    expr: &FilterExpression,
    registry: &TypeRegistry,
) -> Result<(), BuilderError> {
    // Lower the expression into a string buffer first so typed
    // predicates can contribute prelude/with/order_by *before* we lay
    // down the WHERE keyword. Without this two-pass approach, a
    // handler's prelude would land in the middle of a WHERE clause.
    let mut s = String::new();
    write_expr(cur, expr, registry, &mut s)?;
    if !s.is_empty() {
        cur.buf.push_str("\nWHERE ");
        cur.buf.push_str(&s);
    }
    Ok(())
}

fn write_expr(
    cur: &mut Cursor,
    expr: &FilterExpression,
    registry: &TypeRegistry,
    out: &mut String,
) -> Result<(), BuilderError> {
    match expr {
        FilterExpression::Predicate(p) => {
            write_predicate(cur, p, out);
            Ok(())
        }
        FilterExpression::Typed(t) => {
            let handler = registry.get(&t.type_id)?;
            let where_expr = cur.run_handler(|ctx| handler.emit(ctx, t))?;
            if let Some(w) = where_expr {
                out.push_str(&w);
            } else {
                // Handler emitted only prelude / with / order_by — nothing
                // for WHERE. Use a trivially-true placeholder so the
                // boolean structure (AND/OR/NOT) stays valid.
                out.push_str("true");
            }
            Ok(())
        }
        FilterExpression::And(parts) => write_joined(cur, parts, " AND ", registry, out),
        FilterExpression::Or(parts) => write_joined(cur, parts, " OR ", registry, out),
        FilterExpression::Not(inner) => {
            out.push_str("NOT (");
            write_expr(cur, inner, registry, out)?;
            out.push(')');
            Ok(())
        }
    }
}

fn write_joined(
    cur: &mut Cursor,
    parts: &[FilterExpression],
    sep: &str,
    registry: &TypeRegistry,
    out: &mut String,
) -> Result<(), BuilderError> {
    if parts.is_empty() {
        return Ok(());
    }
    if parts.len() == 1 {
        return write_expr(cur, &parts[0], registry, out);
    }
    out.push('(');
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        write_expr(cur, p, registry, out)?;
    }
    out.push(')');
    Ok(())
}

fn write_predicate(cur: &mut Cursor, p: &Predicate, out: &mut String) {
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
    out.push_str(&rendered);
}

pub(super) fn render_property(p: &PropertyRef) -> String {
    match &p.property {
        Some(prop) => format!("{}.{}", p.alias, prop),
        None => p.alias.to_string(),
    }
}
