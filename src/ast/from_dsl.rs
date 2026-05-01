//! Lower a [`DslQuery`] into the strongly-typed [`Query`] AST.
//!
//! This is the only place where DSL-level concepts (`Action::Aggregate`,
//! qualified field strings, JSON literals) are translated into the typed
//! domain. Anything downstream may assume the AST is consistent: aliases
//! resolve, aggregations are only present in `Aggregate` queries, and so on.

use std::collections::HashMap;

use thiserror::Error;

use crate::ast::query::*;
use crate::dsl::schema as d;

#[derive(Debug, Error)]
pub enum AstError {
    #[error("alias '{0}' is not bound in the MATCH pattern")]
    UnknownAlias(String),

    #[error("field '{0}' does not parse as `<alias>` or `<alias>.<property>`")]
    BadField(String),

    #[error("unsupported literal value (objects/non-finite numbers are not allowed)")]
    UnsupportedLiteral,

    #[error("`find` queries may not contain aggregations; use action='aggregate'")]
    AggregateInFind,

    #[error("`aggregate` queries must include `group_by` when projecting non-aggregate fields")]
    MissingGroupBy,

    #[error("sort key '{0}' is neither a projection alias nor a known property")]
    BadSortKey(String),

    #[error("traversal depth {got} exceeds configured max {max}")]
    DepthTooLarge { got: u32, max: u32 },
}

/// Lower a validated DSL query into the AST.
///
/// `max_depth` is enforced here so the AST itself can advertise that any
/// traversal it carries is within limits; downstream code never has to check.
pub fn lower(dsl: d::DslQuery, max_depth: u32) -> Result<ReadQuery, AstError> {
    let mut bound: HashMap<String, ()> = HashMap::new();
    bound.insert(dsl.start.alias.clone(), ());

    let start = Node {
        label: dsl.start.label.clone(),
        alias: Alias::new(dsl.start.alias.clone()),
    };

    let mut traversals = Vec::with_capacity(dsl.traversals.len());
    for t in dsl.traversals {
        if let Some(depth) = t.depth {
            if depth.max > max_depth {
                return Err(AstError::DepthTooLarge { got: depth.max, max: max_depth });
            }
        }
        bound.insert(t.edge.alias.clone(), ());
        bound.insert(t.target.alias.clone(), ());
        traversals.push(EdgeTraversal {
            edge_label: t.edge.label,
            edge_alias: Alias::new(t.edge.alias),
            direction: lower_direction(t.edge.direction),
            target: Node {
                label: t.target.label,
                alias: Alias::new(t.target.alias),
            },
            depth: t.depth.map(|r| Depth { min: r.min, max: r.max }),
        });
    }

    let filter = lower_filters(&dsl.filters, &bound)?;
    let returns = lower_returns(&dsl.return_, &bound)?;

    let action = match dsl.action {
        d::Action::Find => Action::Find,
        d::Action::Aggregate => Action::Aggregate,
    };

    enforce_aggregation_rules(action, &returns, &dsl.group_by)?;

    let group_by = dsl
        .group_by
        .iter()
        .map(|s| resolve_property(s, &bound))
        .collect::<Result<Vec<_>, _>>()?;

    let sort = lower_sort(&dsl.sort, &returns, &bound)?;

    Ok(ReadQuery {
        action,
        start,
        traversals,
        filter,
        returns,
        group_by,
        sort,
        limit: dsl.limit,
    })
}

fn lower_direction(d: d::Direction) -> Direction {
    match d {
        d::Direction::Out => Direction::Out,
        d::Direction::In => Direction::In,
        d::Direction::Both => Direction::Both,
    }
}

fn lower_filters(
    filters: &[d::Filter],
    bound: &HashMap<String, ()>,
) -> Result<Option<FilterExpression>, AstError> {
    if filters.is_empty() {
        return Ok(None);
    }
    let mut preds = Vec::with_capacity(filters.len());
    for f in filters {
        preds.push(FilterExpression::Predicate(Predicate {
            field: resolve_property(&f.field, bound)?,
            op: lower_op(f.op),
            value: Literal::from_json(&f.value).ok_or(AstError::UnsupportedLiteral)?,
        }));
    }
    Ok(Some(if preds.len() == 1 {
        preds.pop().expect("len checked")
    } else {
        FilterExpression::And(preds)
    }))
}

fn lower_op(op: d::FilterOp) -> ComparisonOp {
    match op {
        d::FilterOp::Eq => ComparisonOp::Eq,
        d::FilterOp::Neq => ComparisonOp::Neq,
        d::FilterOp::Gt => ComparisonOp::Gt,
        d::FilterOp::Gte => ComparisonOp::Gte,
        d::FilterOp::Lt => ComparisonOp::Lt,
        d::FilterOp::Lte => ComparisonOp::Lte,
        d::FilterOp::In => ComparisonOp::In,
        d::FilterOp::Contains => ComparisonOp::Contains,
        d::FilterOp::StartsWith => ComparisonOp::StartsWith,
        d::FilterOp::EndsWith => ComparisonOp::EndsWith,
    }
}

fn lower_returns(
    items: &[d::ReturnItem],
    bound: &HashMap<String, ()>,
) -> Result<Vec<ReturnClause>, AstError> {
    items
        .iter()
        .map(|item| match item {
            d::ReturnItem::Field { field, alias } => Ok(ReturnClause::Field {
                field: resolve_property(field, bound)?,
                alias: alias.clone(),
            }),
            d::ReturnItem::Aggregate { aggregate, field, alias } => Ok(ReturnClause::Aggregate {
                func: lower_agg(*aggregate),
                field: resolve_property(field, bound)?,
                alias: alias.clone(),
            }),
        })
        .collect()
}

fn lower_agg(a: d::AggregateFn) -> AggregateFn {
    match a {
        d::AggregateFn::Count => AggregateFn::Count,
        d::AggregateFn::Sum => AggregateFn::Sum,
        d::AggregateFn::Avg => AggregateFn::Avg,
        d::AggregateFn::Min => AggregateFn::Min,
        d::AggregateFn::Max => AggregateFn::Max,
    }
}

fn lower_sort(
    items: &[d::SortItem],
    returns: &[ReturnClause],
    bound: &HashMap<String, ()>,
) -> Result<Vec<SortKey>, AstError> {
    let projection_aliases: Vec<&str> = returns
        .iter()
        .filter_map(|r| match r {
            ReturnClause::Field { alias, .. } => alias.as_deref(),
            ReturnClause::Aggregate { alias, .. } => alias.as_deref(),
        })
        .collect();

    items
        .iter()
        .map(|s| {
            let order = match s.order {
                d::SortOrder::Asc => SortOrder::Asc,
                d::SortOrder::Desc => SortOrder::Desc,
            };
            let key = if s.field.contains('.') {
                SortRef::Property(resolve_property(&s.field, bound)?)
            } else if projection_aliases.iter().any(|a| *a == s.field) {
                SortRef::Projected(s.field.clone())
            } else if bound.contains_key(&s.field) {
                // Sorting by an entity is unusual but legal — fall through as
                // a property ref with no property part.
                SortRef::Property(PropertyRef { alias: Alias::new(s.field.clone()), property: None })
            } else {
                return Err(AstError::BadSortKey(s.field.clone()));
            };
            Ok(SortKey { key, order })
        })
        .collect()
}

fn resolve_property(s: &str, bound: &HashMap<String, ()>) -> Result<PropertyRef, AstError> {
    let r = PropertyRef::parse(s).ok_or_else(|| AstError::BadField(s.to_string()))?;
    if !bound.contains_key(r.alias.as_str()) {
        return Err(AstError::UnknownAlias(r.alias.0.clone()));
    }
    Ok(r)
}

fn enforce_aggregation_rules(
    action: Action,
    returns: &[ReturnClause],
    group_by: &[String],
) -> Result<(), AstError> {
    let has_aggregate = returns.iter().any(|r| matches!(r, ReturnClause::Aggregate { .. }));
    let has_plain = returns.iter().any(|r| matches!(r, ReturnClause::Field { .. }));

    match action {
        Action::Find => {
            if has_aggregate {
                return Err(AstError::AggregateInFind);
            }
        }
        Action::Aggregate => {
            // If we project both aggregated and plain columns, the plain ones
            // must appear in `group_by` — that's how SQL/Cypher semantics work.
            if has_aggregate && has_plain && group_by.is_empty() {
                return Err(AstError::MissingGroupBy);
            }
        }
    }
    Ok(())
}
