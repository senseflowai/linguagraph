//! RETURN, ORDER BY and LIMIT.

use std::fmt::Write;

use crate::ast::query::*;

use super::cursor::Cursor;
use super::where_part::render_property;

pub(super) fn write_return(cur: &mut Cursor, q: &ReadQuery) {
    cur.buf.push_str("\nRETURN ");
    let parts: Vec<String> = q.returns.iter().map(render_return).collect();
    cur.buf.push_str(&parts.join(", "));
}

pub(super) fn render_group_key(key: &GroupByKey) -> String {
    let base = render_property(&key.field);
    match key.transform {
        None => base,
        Some(GroupByTransform::DatePart(part)) => {
            format!("{}.{}", base, render_date_part(part))
        }
    }
}

fn render_date_part(part: DatePart) -> &'static str {
    match part {
        DatePart::Year => "year",
        DatePart::Quarter => "quarter",
        DatePart::Month => "month",
        DatePart::Day => "day",
        DatePart::Hour => "hour",
    }
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
