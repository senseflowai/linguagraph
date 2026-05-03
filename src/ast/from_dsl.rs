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
use crate::metadata::PropertyMetadata;
use crate::types::context::{LowerCtx, RawTypedFilter};
use crate::types::{TypeError, TypeRegistry, TypeId, TypedOp};

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

    #[error("unknown plain filter op '{0}'")]
    UnknownPlainOp(String),

    #[error("typed op '{op}' is not supported by type '{ty}'")]
    UnsupportedTypedOp { ty: String, op: String },

    #[error("type system error: {0}")]
    Type(#[from] TypeError),
}

/// Lower a validated DSL query into the AST.
///
/// `max_depth` is enforced here so the AST itself can advertise that any
/// traversal it carries is within limits; downstream code never has to check.
///
/// Backwards-compatible alias using an empty registry and no metadata.
/// Use this only when the DSL is known not to contain typed filters.
pub fn lower(dsl: d::DslQuery, max_depth: u32) -> Result<ReadQuery, AstError> {
    lower_full(dsl, max_depth, &TypeRegistry::empty(), None)
}

/// Lower a DSL query, dispatching typed filters through `registry`.
/// Equivalent to [`lower_full`] without a metadata snapshot.
pub fn lower_with_registry(
    dsl: d::DslQuery,
    max_depth: u32,
    registry: &TypeRegistry,
) -> Result<ReadQuery, AstError> {
    lower_full(dsl, max_depth, registry, None)
}

/// Lower a DSL query, dispatching typed filters through `registry`,
/// and **auto-resolve** the type for filters that omit `"type"` by
/// looking the field up in `metadata`.
///
/// The lookup key is `<node-label>.<property>` — the same shape
/// [`crate::metadata::collect_from_mapping`] writes. Aliases bound to a
/// node take their label from the MATCH pattern; aliases bound to an
/// edge use the edge label.
///
/// This makes the DSL terser: an LLM can emit
/// `{"field": "c.name", "op": "search", "value": "apple"}` and the
/// SemanticText handler is selected automatically because the mapping
/// declared `c.name` as `SemanticText`.
pub fn lower_full(
    dsl: d::DslQuery,
    max_depth: u32,
    registry: &TypeRegistry,
    metadata: Option<&PropertyMetadata>,
) -> Result<ReadQuery, AstError> {
    let mut bound: HashMap<String, ()> = HashMap::new();
    bound.insert(dsl.start.alias.clone(), ());

    // Track which graph label each alias resolves to. Used to look up
    // typed-property metadata as `<label>.<property>`.
    let mut alias_labels: HashMap<String, String> = HashMap::new();
    alias_labels.insert(dsl.start.alias.clone(), dsl.start.label.clone());

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

        // Resolve `from`. Default = the start node, *not* the previous
        // traversal's target — we don't want a list of traversals to
        // silently collapse into one chained path. Authors who actually
        // want a chain set `from` explicitly.
        let from_name = t.from.clone().unwrap_or_else(|| dsl.start.alias.clone());
        if !bound.contains_key(from_name.as_str()) {
            return Err(AstError::UnknownAlias(from_name));
        }

        bound.insert(t.edge.alias.clone(), ());
        bound.insert(t.target.alias.clone(), ());
        alias_labels.insert(t.edge.alias.clone(), t.edge.label.clone());
        alias_labels.insert(t.target.alias.clone(), t.target.label.clone());
        traversals.push(EdgeTraversal {
            from_alias: Alias::new(from_name),
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

    let filter = lower_filters(&dsl.filters, &bound, &alias_labels, registry, metadata)?;
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
    alias_labels: &HashMap<String, String>,
    registry: &TypeRegistry,
    metadata: Option<&PropertyMetadata>,
) -> Result<Option<FilterExpression>, AstError> {
    if filters.is_empty() {
        return Ok(None);
    }
    let mut preds = Vec::with_capacity(filters.len());
    for f in filters {
        let field = resolve_property(&f.field, bound)?;

        // Effective type tag for this filter.
        //
        // Order of precedence:
        //   1. Explicit `"type"` in the DSL (always wins; lets the LLM
        //      override the mapping when it knows better).
        //   2. The registered field-type from `PropertyMetadata`,
        //      keyed on `<alias-label>.<property>`.
        //   3. None — falls back to plain ops.
        //
        // Scenario (1) keeps the existing surface working unchanged;
        // (2) is what makes typed filters terse for the LLM:
        //     `{"field": "c.name", "op": "search", "value": "apple"}`
        // resolves to `SemanticText` automatically when the mapping
        // declared `Company.name` as such.
        let effective_type = f
            .field_type
            .clone()
            .or_else(|| infer_type(&field, alias_labels, metadata));

        match effective_type {
            // Plain (untyped) predicate: parse the op and convert the
            // value into the typed AST literal.
            None => {
                let op = d::FilterOp::parse(&f.op)
                    .ok_or_else(|| AstError::UnknownPlainOp(f.op.clone()))?;
                preds.push(FilterExpression::Predicate(Predicate {
                    field,
                    op: lower_op(op),
                    value: Literal::from_json(&f.value).ok_or(AstError::UnsupportedLiteral)?,
                }));
            }
            // Typed predicate: route through the registered handler.
            Some(ty_name) => {
                let type_id = TypeId::new(&ty_name);
                let handler = registry.get(&type_id)?;
                let typed_op = parse_typed_op(&f.op).ok_or_else(|| {
                    AstError::UnsupportedTypedOp {
                        ty: ty_name.clone(),
                        op: f.op.clone(),
                    }
                })?;
                if !handler.supported_ops().contains(&typed_op) {
                    return Err(AstError::UnsupportedTypedOp {
                        ty: ty_name.clone(),
                        op: f.op.clone(),
                    });
                }
                let mut ctx = LowerCtx {
                    raw: RawTypedFilter {
                        field: &field,
                        op: typed_op,
                        value: &f.value,
                    },
                    type_id: type_id.clone(),
                };
                let typed = handler.lower(&mut ctx)?;
                preds.push(FilterExpression::Typed(typed));
            }
        }
    }
    Ok(Some(if preds.len() == 1 {
        preds.pop().expect("len checked")
    } else {
        FilterExpression::And(preds)
    }))
}

/// Look up the registered type for `field` in `metadata`. Returns
/// `None` when:
///
/// * the alias has no recorded label (shouldn't happen for valid
///   queries — the lowering step binds every alias before filters
///   run),
/// * the field has no `.<property>` part (entity-level references
///   never carry a type), or
/// * the metadata snapshot has no entry for the resolved key.
fn infer_type(
    field: &PropertyRef,
    alias_labels: &HashMap<String, String>,
    metadata: Option<&PropertyMetadata>,
) -> Option<String> {
    let meta = metadata?;
    let prop = field.property.as_deref()?;
    let label = alias_labels.get(field.alias.as_str())?;
    meta.get_type(&format!("{label}.{prop}")).map(str::to_string)
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

/// Parse a typed-op string. Supports the snake_case form used in the
/// DSL (`search`, `hybrid_search`, `near`, …) as well as the plain ops
/// — handlers can opt into reusing them.
fn parse_typed_op(s: &str) -> Option<TypedOp> {
    Some(match s {
        "eq" => TypedOp::Eq,
        "neq" => TypedOp::Neq,
        "gt" => TypedOp::Gt,
        "gte" => TypedOp::Gte,
        "lt" => TypedOp::Lt,
        "lte" => TypedOp::Lte,
        "in" => TypedOp::In,
        "contains" => TypedOp::Contains,
        "starts_with" => TypedOp::StartsWith,
        "ends_with" => TypedOp::EndsWith,
        "search" => TypedOp::Search,
        "hybrid_search" | "hybrid" => TypedOp::HybridSearch,
        "near" => TypedOp::Near,
        _ => return None,
    })
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
