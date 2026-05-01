//! RETURN, ORDER BY and LIMIT.

use std::fmt::Write;

use crate::ast::query::*;

use super::cursor::Cursor;
use super::where_part::render_property;

pub(super) fn write_return(cur: &mut Cursor, q: &Query) {
    cur.buf.push_str("\nRETURN ");
    let parts: Vec<String> = q.returns.iter().map(render_return).collect();
    cur.buf.push_str(&parts.join(", "));
}

pub(super) fn write_order_by(cur: &mut Cursor, sort: &[SortKey]) {
    if sort.is_empty() {
        return;
    }
    cur.buf.push_str("\nORDER BY ");
    let parts: Vec<String> = sort
        .iter()
        .map(|s| {
            let key = match &s.key {
                SortRef::Projected(name) => name.clone(),
                SortRef::Property(p) => render_property(p),
            };
            let dir = match s.order {
                SortOrder::Asc => "ASC",
                SortOrder::Desc => "DESC",
            };
            format!("{key} {dir}")
        })
        .collect();
    cur.buf.push_str(&parts.join(", "));
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
        ReturnClause::Aggregate { func, field, alias } => {
            let inner = render_property(field);
            let call = match func {
                AggregateFn::Count => format!("count({inner})"),
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
