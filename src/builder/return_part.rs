//! RETURN, ORDER BY and LIMIT.

use std::fmt::Write;

use crate::ast::query::*;

use super::cursor::Cursor;
use super::where_part::render_property;

pub(super) fn write_return(cur: &mut Cursor, q: &ReadQuery) {
    cur.buf.push_str(if q.distinct {
        "\nRETURN DISTINCT "
    } else {
        "\nRETURN "
    });
    let parts: Vec<String> = q.returns.iter().map(render_return).collect();
    cur.buf.push_str(&parts.join(", "));
}

pub(super) fn render_group_key(key: &GroupByKey) -> String {
    let base = render_property(&key.field);
    match key.transform {
        None => base,
        Some(GroupByTransform::DatePart(part)) => render_date_part_expr(&base, part),
    }
}

fn render_date_part_expr(base: &str, part: DatePart) -> String {
    // Date/time properties are stored as ISO-8601 strings by the ingest
    // layer, and older datasets may contain date-only values (`YYYY-MM-DD`).
    // Neo4j's `datetime()` rejects those date-only strings, so date-part
    // grouping extracts fixed-width ISO components from `toString(value)`
    // instead. This also works for native temporal values because Neo4j
    // renders them as ISO-shaped strings.
    let value = format!("toString({base})");
    match part {
        DatePart::Year => render_numeric_component(base, &value, 0, 4, 4),
        DatePart::Month => render_numeric_component(base, &value, 5, 2, 7),
        DatePart::Day => render_numeric_component(base, &value, 8, 2, 10),
        DatePart::Hour => render_numeric_component(base, &value, 11, 2, 13),
        DatePart::Quarter => {
            let month = format!("toInteger(substring({value}, 5, 2))");
            format!(
                "CASE WHEN {base} IS NULL OR size({value}) < 7 THEN NULL ELSE toInteger(ceil({month} / 3.0)) END"
            )
        }
    }
}

fn render_numeric_component(
    base: &str,
    value: &str,
    start: usize,
    length: usize,
    min_size: usize,
) -> String {
    format!(
        "CASE WHEN {base} IS NULL OR size({value}) < {min_size} THEN NULL ELSE toInteger(substring({value}, {start}, {length})) END"
    )
}

#[allow(dead_code)]
pub(super) fn write_order_by(cur: &mut Cursor, sort: &[SortKey]) {
    if sort.is_empty() {
        return;
    }
    cur.buf.push_str("\nORDER BY ");
    let parts: Vec<String> = sort.iter().map(format_sort_key).collect();
    cur.buf.push_str(&parts.join(", "));
}

/// Like [`write_order_by`] but also flushes any handler-contributed
/// `extra_order_by` keys, *after* the user's explicit sort. This means
/// an explicit `sort` clause in the DSL is always respected first;
/// type-handler sorts (semantic score, geo distance) act as
/// tie-breakers.
pub(super) fn write_order_by_with_extra(cur: &mut Cursor, sort: &[SortKey]) {
    let extras: Vec<(String, crate::types::context::OrderDir)> =
        cur.extra_order_by.drain(..).collect();
    if sort.is_empty() && extras.is_empty() {
        return;
    }
    cur.buf.push_str("\nORDER BY ");
    let mut parts: Vec<String> = sort.iter().map(format_sort_key).collect();
    for (key, dir) in extras {
        parts.push(format!("{key} {}", dir.as_str()));
    }
    cur.buf.push_str(&parts.join(", "));
}

fn format_sort_key(s: &SortKey) -> String {
    let key = match &s.key {
        SortRef::Projected(name) => name.clone(),
        SortRef::Property(p) => render_property(p),
    };
    let dir = match s.order {
        SortOrder::Asc => "ASC",
        SortOrder::Desc => "DESC",
    };
    format!("{key} {dir}")
}

pub(super) fn write_limit(cur: &mut Cursor, limit: u32) {
    let _ = write!(cur.buf, "\nLIMIT {limit}");
}

fn render_return(item: &ReturnClause) -> String {
    match item {
        ReturnClause::Field { field, alias } => {
            let base = render_property(field);
            match alias {
                Some(a) => format!("{base} AS {a}"),
                None => base,
            }
        }
        ReturnClause::GroupKey { key, alias } => {
            let base = render_group_key(key);
            format!("{base} AS {alias}")
        }
        ReturnClause::Aggregate { func, field, alias } => {
            let inner = render_property(field);
            let call = match func {
                // AggregateFn::Count => {
                //     let v = inner.split('.').next();
                //     format!("count({})", v.unwrap_or(inner.as_str()))
                // },
                AggregateFn::Count => {
                    let v = inner.split('.').next();
                    format!("count({})", v.unwrap_or(inner.as_str()))
                }
                AggregateFn::Sum => format!("sum({inner})"),
                AggregateFn::Avg => format!("avg({inner})"),
                AggregateFn::Min => format!("min({inner})"),
                AggregateFn::Max => format!("max({inner})"),
            };
            match alias {
                Some(a) => format!("{call} AS {a}"),
                None => call,
            }
        }
    }
}
