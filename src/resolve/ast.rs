//! Lower a [`DslQuery`] into the strongly-typed [`Query`] AST.
//!
//! This is the only place where DSL-level concepts (qualified field strings,
//! JSON literals, optional action hints) are translated into the typed domain.
//! Anything downstream may assume the AST is consistent: aliases resolve,
//! aggregations use `Aggregate` queries, and so on.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::ast::query::*;
use crate::dsl::schema as d;
use crate::graph::OntologyCatalog;
use crate::types::context::{LowerCtx, RawTypedFilter};
use crate::types::handlers::{build_canonical_query, SemanticTextHandler};
use crate::types::{TypeError, TypeId, TypeRegistry, TypedOp};

#[derive(Debug, Error)]
pub enum AstError {
    #[error("alias '{0}' is not bound in the MATCH pattern")]
    UnknownAlias(String),

    #[error("field '{0}' does not parse as `<alias>` or `<alias>.<property>`")]
    BadField(String),

    #[error("unsupported literal value (objects/non-finite numbers are not allowed)")]
    UnsupportedLiteral,

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
/// Backwards-compatible alias using an empty registry and no graph specification.
/// Use this only when the DSL is known not to contain typed filters.
pub fn lower(dsl: d::DslQuery, max_depth: u32) -> Result<ReadQuery, AstError> {
    lower_full(dsl, max_depth, &TypeRegistry::empty(), None)
}

/// Lower a DSL query, dispatching typed filters through `registry`.
/// Equivalent to [`lower_full`] without a graph specification snapshot.
pub fn lower_with_registry(
    dsl: d::DslQuery,
    max_depth: u32,
    registry: &TypeRegistry,
) -> Result<ReadQuery, AstError> {
    lower_full(dsl, max_depth, registry, None)
}

/// Lower a DSL query, dispatching typed filters through `registry`,
/// and **auto-resolve** the type for filters that omit `"type"` by
/// looking the field up in `catalog`.
///
/// The lookup key is `<node-label>.<property>` — the same shape
/// [`crate::graph::GraphSpecification`] writes. Aliases bound to a
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
    catalog: Option<&OntologyCatalog>,
) -> Result<ReadQuery, AstError> {
    let mut bound: HashMap<String, ()> = HashMap::new();
    bound.insert(dsl.start.alias.clone(), ());

    // Track which graph label each alias resolves to. Used to look up
    // typed properties in GraphSpecification as `<label>.<property>`.
    let mut alias_labels: HashMap<String, String> = HashMap::new();
    alias_labels.insert(dsl.start.alias.clone(), dsl.start.label.clone());

    // A query-wide prefix label is propagated to every node (start +
    // traversal targets) so the emitter scopes every MATCH pattern
    // with the same extra label. Empty strings are normalised to None.
    let query_prefix_label = dsl
        .prefix_label
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let start = Node {
        label: dsl.start.label.clone(),
        alias: Alias::new(dsl.start.alias.clone()),
        prefix_label: query_prefix_label.clone(),
    };

    let mut traversals = Vec::with_capacity(dsl.traversals.len());
    for t in dsl.traversals {
        if let Some(depth) = t.depth {
            if depth.max > max_depth {
                return Err(AstError::DepthTooLarge {
                    got: depth.max,
                    max: max_depth,
                });
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
                prefix_label: query_prefix_label.clone(),
            },
            depth: t.depth.map(|r| Depth {
                min: r.min,
                max: r.max,
            }),
            optional: t.optional,
        });
    }

    // Normalise the query-wide prefix_index the same way as
    // prefix_label: trim, drop empties, then pass it through to typed
    // filter lowering so handlers fold it into collection names.
    let query_prefix_index = dsl
        .prefix_index
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let filter = lower_filters(
        &dsl.filters,
        &bound,
        &alias_labels,
        registry,
        catalog,
        query_prefix_index.as_deref(),
    )?;
    let mut returns = lower_returns(&dsl.return_, &bound)?;

    let action = infer_action(&returns);

    enforce_aggregation_rules(action, &returns, &dsl.group_by)?;

    let group_by = dsl
        .group_by
        .iter()
        .map(|g| lower_group_by(g, &bound))
        .collect::<Result<Vec<_>, _>>()?;

    let mut sort = lower_sort(&dsl.sort, &returns, &group_by, &bound)?;

    // Cypher has no standalone GROUP BY: the grouping keys of an
    // aggregating projection are exactly its non-aggregate RETURN
    // columns. For an `aggregate` query two things follow, and the
    // engine rejects the query unless both hold:
    //
    //  * Every `group_by` key must be projected, otherwise it has no
    //    effect on the grouping at all.
    //  * After an aggregating RETURN only the projected *aliases* stay
    //    in scope. A bare `ORDER BY sv.work_start` is parsed as a
    //    property access on `sv`, which is already aggregated away —
    //    Memgraph reports "Unbound variable: sv". So each projected key
    //    needs an explicit alias, and a `sort` over that key must
    //    target the alias rather than the property expression.
    if matches!(action, Action::Aggregate) {
        project_group_by_keys(&mut returns, &mut sort, &group_by, &bound);
    }

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

/// One entity alias's accumulated SemanticText filter terms, awaiting
/// consolidation into a single [`TypedOp::EntitySearch`] predicate.
struct SemGroup {
    alias: String,
    label: String,
    terms: Vec<(Option<String>, String)>,
}

/// SemanticText ops that fold into the consolidated per-entity search —
/// exactly the "default rerank" ops that historically each emitted their
/// own `libqlink.search_reranked`. Explicit `search` (pure KNN) and
/// `hybrid_search` keep their per-field semantics and are not folded.
fn is_foldable_entity_op(op: TypedOp) -> bool {
    matches!(
        op,
        TypedOp::Eq | TypedOp::Neq | TypedOp::Contains | TypedOp::SearchReranked
    )
}

fn lower_filters(
    filters: &[d::Filter],
    bound: &HashMap<String, ()>,
    alias_labels: &HashMap<String, String>,
    registry: &TypeRegistry,
    catalog: Option<&OntologyCatalog>,
    prefix_index: Option<&str>,
) -> Result<Option<FilterExpression>, AstError> {
    if filters.is_empty() {
        return Ok(None);
    }
    let mut preds = Vec::with_capacity(filters.len());
    // SemanticText "default rerank" filters that share an alias are
    // consolidated into a single per-entity hybrid search (see the
    // post-loop pass). We accumulate their terms here, grouped by alias
    // in first-appearance order so the generated Cypher is deterministic.
    let mut semantic_groups: Vec<SemGroup> = Vec::new();

    for f in filters {
        let field = resolve_property(&f.field, bound)?;

        // Effective type tag for this filter.
        //
        // Order of precedence:
        //   1. Explicit `"type"` in the DSL (always wins; lets the LLM
        //      override the mapping when it knows better).
        //   2. The registered field-type from `GraphSpecification`,
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
            .or_else(|| infer_type(&field, alias_labels, catalog));

        // ── Fold SemanticText default-rerank filters by alias. ──────────
        // `eq`/`neq`/`contains`/`search_reranked` on a SemanticText field
        // no longer emit their own per-field `search_reranked`; we gather
        // them per entity and emit one consolidated, field-agnostic hybrid
        // search over `_canonical` after the loop. A non-string value or a
        // label we can't resolve falls through to the normal typed path,
        // which validates and reports the error exactly as before.
        if effective_type.as_deref() == Some(SemanticTextHandler::TYPE_ID) {
            if let (Some(op), Some(value), Some(label)) = (
                parse_typed_op(&f.op),
                f.value.as_str(),
                alias_labels.get(field.alias.as_str()),
            ) {
                if is_foldable_entity_op(op) {
                    let alias = field.alias.as_str();
                    match semantic_groups.iter_mut().find(|g| g.alias == alias) {
                        Some(g) => g.terms.push((field.property.clone(), value.to_string())),
                        None => semantic_groups.push(SemGroup {
                            alias: alias.to_string(),
                            label: label.clone(),
                            terms: vec![(field.property.clone(), value.to_string())],
                        }),
                    }
                    continue;
                }
            }
        }

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
                let typed_op =
                    parse_typed_op(&f.op).ok_or_else(|| AstError::UnsupportedTypedOp {
                        ty: ty_name.clone(),
                        op: f.op.clone(),
                    })?;
                if !handler.supported_ops().contains(&typed_op) {
                    return Err(AstError::UnsupportedTypedOp {
                        ty: ty_name.clone(),
                        op: f.op.clone(),
                    });
                }
                let field_label = alias_labels.get(field.alias.as_str()).map(String::as_str);
                let mut ctx = LowerCtx {
                    raw: RawTypedFilter {
                        field: &field,
                        op: typed_op,
                        value: &f.value,
                    },
                    type_id: type_id.clone(),
                    field_label,
                    prefix_index,
                };
                let typed = handler.lower(&mut ctx)?;
                preds.push(FilterExpression::Typed(typed));
            }
        }
    }

    // ── Consolidated per-entity hybrid search. ─────────────────────────
    // One synthetic `EntitySearch` typed predicate per alias: build the
    // combined canonical-style query string (mirroring the `_canonical`
    // document format), then let the SemanticText handler embed it once
    // and target the `_canonical` collection. The handler is the one the
    // registry already holds for `SemanticText`; an inferred type means it
    // is registered, so `get` cannot miss here.
    for group in semantic_groups {
        let query_text = build_canonical_query(&group.label, &group.terms, true);
        let value = serde_json::Value::String(query_text);
        let field = PropertyRef {
            alias: Alias::new(group.alias.as_str()),
            property: None,
        };
        let type_id = TypeId::new(SemanticTextHandler::TYPE_ID);
        let handler = registry.get(&type_id)?;
        let mut ctx = LowerCtx {
            raw: RawTypedFilter {
                field: &field,
                op: TypedOp::EntitySearch,
                value: &value,
            },
            type_id: type_id.clone(),
            field_label: Some(group.label.as_str()),
            prefix_index,
        };
        let typed = handler.lower(&mut ctx)?;
        preds.push(FilterExpression::Typed(typed));
    }

    if preds.is_empty() {
        return Ok(None);
    }
    Ok(Some(if preds.len() == 1 {
        preds.pop().expect("len checked")
    } else {
        FilterExpression::And(preds)
    }))
}

/// Look up the registered type for `field` in `catalog`. Returns
/// `None` when:
///
/// * the alias has no recorded label (shouldn't happen for valid
///   queries — the lowering step binds every alias before filters
///   run),
/// * the field has no `.<property>` part (entity-level references
///   never carry a type), or
/// * the graph specification snapshot has no entry for the resolved key.
fn infer_type(
    field: &PropertyRef,
    alias_labels: &HashMap<String, String>,
    catalog: Option<&OntologyCatalog>,
) -> Option<String> {
    let spec = catalog?;
    let prop = field.property.as_deref()?;
    let label = alias_labels.get(field.alias.as_str())?;
    spec.get_query_type(label, prop).map(str::to_string)
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
        "search_reranked" => TypedOp::SearchReranked,
        "hybrid_search" | "hybrid" => TypedOp::HybridSearch,
        "near" => TypedOp::Near,
        _ => return None,
    })
}

fn lower_group_by(
    item: &d::GroupByItem,
    bound: &HashMap<String, ()>,
) -> Result<GroupByKey, AstError> {
    Ok(match item {
        d::GroupByItem::Field(field) => GroupByKey {
            field: resolve_property(field, bound)?,
            transform: None,
            alias: None,
        },
        d::GroupByItem::DatePart {
            field,
            date_part,
            alias,
        } => GroupByKey {
            field: resolve_property(field, bound)?,
            transform: Some(GroupByTransform::DatePart(lower_date_part(*date_part))),
            alias: alias.clone(),
        },
    })
}

fn lower_date_part(part: d::DatePart) -> DatePart {
    match part {
        d::DatePart::Year => DatePart::Year,
        d::DatePart::Quarter => DatePart::Quarter,
        d::DatePart::Month => DatePart::Month,
        d::DatePart::Day => DatePart::Day,
        d::DatePart::Hour => DatePart::Hour,
    }
}

fn lower_returns(
    items: &[d::ReturnItem],
    bound: &HashMap<String, ()>,
) -> Result<Vec<ReturnClause>, AstError> {
    items
        .iter()
        .map(|item| match item {
            d::ReturnItem::DatePart {
                field,
                date_part,
                alias,
            } => {
                let key = GroupByKey {
                    field: resolve_property(field, bound)?,
                    transform: Some(GroupByTransform::DatePart(lower_date_part(*date_part))),
                    alias: alias.clone(),
                };
                Ok(ReturnClause::GroupKey {
                    alias: alias
                        .clone()
                        .unwrap_or_else(|| default_group_key_alias(&key)),
                    key,
                })
            }
            d::ReturnItem::Field { field, alias } => Ok(ReturnClause::Field {
                field: resolve_property(field, bound)?,
                alias: alias.clone(),
            }),
            d::ReturnItem::Aggregate {
                aggregate,
                field,
                alias,
            } => Ok(ReturnClause::Aggregate {
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
    group_by: &[GroupByKey],
    bound: &HashMap<String, ()>,
) -> Result<Vec<SortKey>, AstError> {
    let mut projection_aliases: Vec<&str> = returns
        .iter()
        .filter_map(|r| match r {
            ReturnClause::Field { alias, .. } => alias.as_deref(),
            ReturnClause::GroupKey { alias, .. } => Some(alias.as_str()),
            ReturnClause::Aggregate { alias, .. } => alias.as_deref(),
        })
        .collect();
    projection_aliases.extend(group_by.iter().filter_map(|g| g.alias.as_deref()));

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
                SortRef::Property(PropertyRef {
                    alias: Alias::new(s.field.clone()),
                    property: None,
                })
            } else {
                return Err(AstError::BadSortKey(s.field.clone()));
            };
            Ok(SortKey { key, order })
        })
        .collect()
}

/// Make an aggregate query's `group_by` keys safe for Cypher.
///
/// Each key is projected as a non-aggregate RETURN column carrying an
/// explicit alias (reusing an existing projection of the same field
/// when there is one), and any `sort` key that targets a group_by
/// property is rewritten to reference that alias. See the call site for
/// why this is required.
fn project_group_by_keys(
    returns: &mut Vec<ReturnClause>,
    sort: &mut [SortKey],
    group_by: &[GroupByKey],
    bound: &HashMap<String, ()>,
) {
    // A generated alias must not collide with an existing projection
    // alias or with a variable bound by the MATCH pattern.
    let mut taken: HashSet<String> = bound.keys().cloned().collect();
    for r in returns.iter() {
        let alias = match r {
            ReturnClause::Field { alias, .. } | ReturnClause::Aggregate { alias, .. } => alias,
            ReturnClause::GroupKey { alias, .. } => {
                taken.insert(alias.clone());
                continue;
            }
        };
        if let Some(a) = alias {
            taken.insert(a.clone());
        }
    }

    let mut key_alias: Vec<(GroupByKey, String)> = Vec::with_capacity(group_by.len());
    for key in group_by {
        enum ExistingProjection<'a> {
            Field(&'a mut Option<String>),
            GroupKey(&'a str),
        }

        let existing = returns.iter_mut().find_map(|r| match r {
            ReturnClause::Field { field, alias }
                if key.transform.is_none() && field == &key.field =>
            {
                Some(ExistingProjection::Field(alias))
            }
            ReturnClause::GroupKey {
                key: projected,
                alias,
            } if projected == key => Some(ExistingProjection::GroupKey(alias.as_str())),
            _ => None,
        });
        let alias = match existing {
            Some(ExistingProjection::Field(slot)) => {
                let alias = slot
                    .clone()
                    .unwrap_or_else(|| unique_alias(key, &mut taken));
                taken.insert(alias.clone());
                *slot = Some(alias.clone());
                alias
            }
            Some(ExistingProjection::GroupKey(alias)) => {
                let alias = alias.to_string();
                taken.insert(alias.clone());
                alias
            }
            None => {
                let a = key
                    .alias
                    .clone()
                    .filter(|candidate| taken.insert(candidate.clone()))
                    .unwrap_or_else(|| unique_alias(key, &mut taken));
                if key.transform.is_none() {
                    returns.push(ReturnClause::Field {
                        field: key.field.clone(),
                        alias: Some(a.clone()),
                    });
                } else {
                    returns.push(ReturnClause::GroupKey {
                        key: key.clone(),
                        alias: a.clone(),
                    });
                }
                a
            }
        };
        key_alias.push((key.clone(), alias));
    }

    for s in sort.iter_mut() {
        if let SortRef::Property(p) = &s.key {
            if let Some((_, a)) = key_alias.iter().find(|(k, _)| &k.field == p) {
                s.key = SortRef::Projected(a.clone());
            }
        }
    }
}

/// Derive an identifier-shaped alias for a group_by key that does not
/// collide with anything in `taken` (which it also updates).
fn unique_alias(key: &GroupByKey, taken: &mut HashSet<String>) -> String {
    let base = default_group_key_alias(key);
    let mut candidate = base.clone();
    let mut n = 2u32;
    while !taken.insert(candidate.clone()) {
        candidate = format!("{base}_{n}");
        n += 1;
    }
    candidate
}

fn default_group_key_alias(key: &GroupByKey) -> String {
    match (&key.field.property, key.transform) {
        (Some(prop), Some(GroupByTransform::DatePart(part))) => {
            format!("{}_{}_{}", key.field.alias, prop, date_part_name(part))
        }
        (Some(prop), None) => format!("{}_{}", key.field.alias, prop),
        (None, Some(GroupByTransform::DatePart(part))) => {
            format!("{}_{}", key.field.alias, date_part_name(part))
        }
        (None, None) => key.field.alias.0.clone(),
    }
}

fn date_part_name(part: DatePart) -> &'static str {
    match part {
        DatePart::Year => "year",
        DatePart::Quarter => "quarter",
        DatePart::Month => "month",
        DatePart::Day => "day",
        DatePart::Hour => "hour",
    }
}

fn resolve_property(s: &str, bound: &HashMap<String, ()>) -> Result<PropertyRef, AstError> {
    let r = PropertyRef::parse(s).ok_or_else(|| AstError::BadField(s.to_string()))?;
    if !bound.contains_key(r.alias.as_str()) {
        return Err(AstError::UnknownAlias(r.alias.0.clone()));
    }
    Ok(r)
}

fn infer_action(returns: &[ReturnClause]) -> Action {
    if returns
        .iter()
        .any(|r| matches!(r, ReturnClause::Aggregate { .. }))
    {
        Action::Aggregate
    } else {
        Action::Find
    }
}

fn enforce_aggregation_rules(
    action: Action,
    returns: &[ReturnClause],
    group_by: &[d::GroupByItem],
) -> Result<(), AstError> {
    let has_plain = returns.iter().any(|r| {
        matches!(
            r,
            ReturnClause::Field { .. } | ReturnClause::GroupKey { .. }
        )
    });

    match action {
        Action::Find => {}
        Action::Aggregate => {
            // If we project both aggregated and plain columns, the plain ones
            // must appear in `group_by` — that's how SQL/Cypher semantics work.
            if has_plain && group_by.is_empty() {
                return Err(AstError::MissingGroupBy);
            }
        }
    }
    Ok(())
}
