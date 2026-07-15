//! WHERE clause emission. Every value is bound as a parameter.
//!
//! Typed predicates (carried in [`FilterExpression::Typed`]) are routed
//! through the [`crate::types::TypeRegistry`]; the matching handler
//! decides what Cypher fragments to emit and what boolean expression
//! to splice here. The builder itself never inspects the type id.

use std::collections::HashSet;

use crate::ast::query::*;
use crate::normalize::{normalize_literal_for_match, normalized_property_name};
use crate::types::TypeRegistry;

use super::cursor::Cursor;
use super::cypher::BuilderError;

/// Every alias referenced by a field in `expr` — via plain or typed
/// predicates, recursing through `And`/`Or`/`Not`. Used by the builder to
/// detect when a filter needs an alias that's only bound by an OPTIONAL
/// MATCH, so it can defer the WHERE clause until after that match.
pub(super) fn collect_referenced_aliases<'a>(
    expr: &'a FilterExpression,
    out: &mut HashSet<&'a str>,
) {
    match expr {
        FilterExpression::Predicate(p) => {
            out.insert(p.field.alias.as_str());
        }
        FilterExpression::Typed(t) => {
            out.insert(t.field.alias.as_str());
        }
        FilterExpression::And(parts) | FilterExpression::Or(parts) => {
            for part in parts {
                collect_referenced_aliases(part, out);
            }
        }
        FilterExpression::Not(inner) => collect_referenced_aliases(inner, out),
    }
}

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
    let case_insensitive = matches!(
        p.op,
        ComparisonOp::EqCi
            | ComparisonOp::NeqCi
            | ComparisonOp::InCi
            | ComparisonOp::ContainsCi
            | ComparisonOp::StartsWithCi
            | ComparisonOp::EndsWithCi
    );
    let lhs = if case_insensitive {
        render_normalized_property(&p.field)
    } else {
        render_property(&p.field)
    };
    let value = if case_insensitive {
        normalize_literal_for_match(p.value.clone())
    } else {
        p.value.clone()
    };
    let placeholder = cur.bind(value);
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
        ComparisonOp::EqCi => format!("{lhs} = {placeholder}"),
        ComparisonOp::NeqCi => format!("{lhs} <> {placeholder}"),
        ComparisonOp::InCi => format!("{lhs} IN {placeholder}"),
        ComparisonOp::ContainsCi => format!("{lhs} CONTAINS {placeholder}"),
        ComparisonOp::StartsWithCi => format!("{lhs} STARTS WITH {placeholder}"),
        ComparisonOp::EndsWithCi => format!("{lhs} ENDS WITH {placeholder}"),
    };
    out.push_str(&rendered);
}

pub(super) fn render_property(p: &PropertyRef) -> String {
    match &p.property {
        Some(prop) => format!("{}.{}", p.alias, prop),
        None => p.alias.to_string(),
    }
}

fn render_normalized_property(p: &PropertyRef) -> String {
    match &p.property {
        Some(prop) => format!("{}.{}", p.alias, normalized_property_name(prop)),
        None => p.alias.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pref(alias: &str, prop: &str) -> PropertyRef {
        PropertyRef {
            alias: Alias::new(alias),
            property: Some(prop.to_string()),
        }
    }

    #[test]
    fn collects_aliases_across_and_or_not() {
        let expr = FilterExpression::And(vec![
            FilterExpression::Predicate(Predicate {
                field: pref("p", "age"),
                op: ComparisonOp::Gt,
                value: Literal::Int(18),
            }),
            FilterExpression::Not(Box::new(FilterExpression::Or(vec![
                FilterExpression::Predicate(Predicate {
                    field: pref("c", "revenue"),
                    op: ComparisonOp::Lt,
                    value: Literal::Int(100),
                }),
                FilterExpression::Predicate(Predicate {
                    field: pref("c", "active"),
                    op: ComparisonOp::Eq,
                    value: Literal::Bool(true),
                }),
            ]))),
        ]);

        let mut aliases = HashSet::new();
        collect_referenced_aliases(&expr, &mut aliases);
        assert_eq!(aliases, HashSet::from(["p", "c"]));
    }
}
