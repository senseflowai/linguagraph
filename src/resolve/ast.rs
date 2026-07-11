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
use serde_json::Value;

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

    #[error("filter cardinality '{0}' must be 'one' or 'many'")]
    BadCardinality(String),

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

/// Lower a DSL query, dispatching typed filters through `registry`,
/// and **auto-resolve** the type for filters that omit `"type"` by
/// looking the field up in `catalog`.
///
/// The lookup key is `<node-label>.<property>` — the same shape
/// [`crate::graph::OntologyCatalog`] indexes. Aliases bound to a
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
    // typed properties in the OntologyCatalog as `<label>.<property>`.
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

    // Drop any ORDER BY on a list-typed property. Cypher can't order or
    // compare list values and aborts the whole query at runtime
    // ("Comparison is not defined for values of type list"); a list has no
    // ordering anyway, so silently dropping the key degrades to an
    // unordered result instead of a hard failure. (Scalar comparisons on a
    // list route through the List type handler, which rejects them cleanly
    // at lowering — only `sort` bypasses the handlers.)
    sort.retain(|s| match &s.key {
        SortRef::Property(p) => {
            infer_type(p, &alias_labels, catalog).as_deref()
                != Some(crate::types::BuiltinType::List.id())
        }
        SortRef::Projected(_) => true,
    });

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

/// Whether a folded `SemanticText` group names one specific entity or
/// asks for every matching item. `op` alone can't tell these apart — a
/// free-text `contains` can mean either "the one row whose description
/// says this" or "every row that mentions this" — so this is threaded
/// through as its own axis, sourced from an explicit DSL `cardinality`
/// hint when the model gives one, or inferred from `op` otherwise.
/// Consumed by query-time grounding (skip pinning / rerank for `Many`)
/// and by `SemanticTextHandler::emit` (dense-only recall for `Many`,
/// full hybrid+rerank for `One`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cardinality {
    One,
    Many,
}

impl Cardinality {
    fn as_str(self) -> &'static str {
        match self {
            Cardinality::One => "one",
            Cardinality::Many => "many",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "one" => Some(Cardinality::One),
            "many" => Some(Cardinality::Many),
            _ => None,
        }
    }
}

/// Merge a second `cardinality` hint into the group's running value.
/// Absent hints defer to whatever is already known; two conflicting
/// explicit hints resolve to `Many` — the safe direction, since an
/// extra row is recoverable but one dropped by an over-eager pin isn't.
fn merge_cardinality_hint(
    existing: Option<Cardinality>,
    incoming: Option<Cardinality>,
) -> Option<Cardinality> {
    match (existing, incoming) {
        (None, x) => x,
        (x, None) => x,
        (Some(a), Some(b)) if a == b => Some(a),
        _ => Some(Cardinality::Many),
    }
}

/// One entity alias's accumulated SemanticText filter terms, awaiting
/// consolidation into a single [`TypedOp::EntitySearch`] predicate.
struct SemGroup {
    alias: String,
    label: String,
    terms: Vec<(Option<String>, String)>,
    /// Explicit `cardinality` hint carried by any folded term so far.
    /// Wins over `all_singular_op` below when present.
    cardinality_hint: Option<Cardinality>,
    /// True as long as every folded term came from an `eq` filter — i.e.
    /// the query names one specific entity by value rather than asking
    /// for a fuzzy/broader match. Fallback signal when no term states
    /// `cardinality` explicitly.
    all_singular_op: bool,
}

/// SemanticText ops that fold into one consolidated per-entity hybrid
/// search over `_canonical`.
fn is_foldable_entity_op(op: TypedOp) -> bool {
    matches!(
        op,
        TypedOp::Eq
            | TypedOp::Neq
            | TypedOp::In
            | TypedOp::Contains
            | TypedOp::Search
            | TypedOp::SearchReranked
            | TypedOp::HybridSearch
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
        //   2. The registered field-type from the `OntologyCatalog`,
        //      keyed on `<alias-label>.<property>`.
        //   3. None — falls back to plain ops.
        //
        // Scenario (1) keeps the existing surface working unchanged;
        // (2) is what makes typed filters terse for the LLM:
        //     `{"field": "c.name", "op": "search", "value": "apple"}`
        // resolves to `SemanticText` automatically when the mapping
        // declared `Company.name` as such.
        // Canonicalize contract / legacy spellings (`Text`, `String`,
        // `SemanticText`, …) on an *explicit* DSL `"type"` override to the
        // registry handler id. `infer_type` is intentionally left
        // untouched here: it already returns a query-side id via
        // `OntologyPropertyType::query_type_id`, which for `List` is its
        // own dedicated handler id — re-canonicalizing through
        // `canonical_handler_id` (which maps via the *ingest*-side
        // `handler_id`) would collapse it back onto `Keyword` and lose
        // list-membership `contains` semantics.
        let effective_type = f
            .field_type
            .clone()
            .map(|t| crate::graph::canonical_handler_id(&t))
            .or_else(|| infer_type(&field, alias_labels, catalog));

        // ── Fold SemanticText semantic filters by alias. ───────────────
        // SemanticText filters on the same alias are gathered per entity
        // and emitted as one consolidated, field-agnostic hybrid search
        // over `_canonical` after the loop. Non-string values, or a label
        // we can't resolve, fall through to the normal typed path, which
        // keeps the AST lowering deterministic and reports errors exactly
        // as before.
        if effective_type.as_deref() == Some(SemanticTextHandler::TYPE_ID) {
            if let (Some(op), Some(label)) = (
                parse_typed_op(&f.op),
                alias_labels.get(field.alias.as_str()),
            ) {
                if is_foldable_entity_op(op) {
                    let term = semantic_text_term_value(op, &f.value)?;
                    let alias = field.alias.as_str();
                    let is_eq = op == TypedOp::Eq;
                    let explicit_cardinality = f
                        .cardinality
                        .as_deref()
                        .map(|s| {
                            Cardinality::parse(s)
                                .ok_or_else(|| AstError::BadCardinality(s.to_string()))
                        })
                        .transpose()?;
                    match semantic_groups.iter_mut().find(|g| g.alias == alias) {
                        Some(g) => {
                            g.terms.push((field.property.clone(), term));
                            g.all_singular_op = g.all_singular_op && is_eq;
                            g.cardinality_hint =
                                merge_cardinality_hint(g.cardinality_hint, explicit_cardinality);
                        }
                        None => semantic_groups.push(SemGroup {
                            alias: alias.to_string(),
                            label: label.clone(),
                            terms: vec![(field.property.clone(), term)],
                            all_singular_op: is_eq,
                            cardinality_hint: explicit_cardinality,
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
        let mut typed = handler.lower(&mut ctx)?;
        let cardinality = group.cardinality_hint.unwrap_or(if group.all_singular_op {
            Cardinality::One
        } else {
            Cardinality::Many
        });
        typed.params.insert(
            "cardinality".to_string(),
            Literal::String(cardinality.as_str().to_string()),
        );
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

fn semantic_text_term_value(op: TypedOp, value: &Value) -> Result<String, AstError> {
    match (op, value) {
        (TypedOp::In, Value::Array(items)) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::String(s) => values.push(s.clone()),
                    _ => return Err(AstError::UnsupportedLiteral),
                }
            }
            if values.is_empty() {
                return Err(AstError::UnsupportedLiteral);
            }
            Ok(values.join(" | "))
        }
        (_, Value::String(s)) => Ok(s.clone()),
        _ => Err(AstError::UnsupportedLiteral),
    }
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
        "matches" | "regex" => TypedOp::Matches,
        "search" => TypedOp::Search,
        "search_reranked" => TypedOp::SearchReranked,
        "hybrid_search" | "hybrid" => TypedOp::HybridSearch,
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

#[cfg(test)]
mod cardinality_tests {
    use super::*;

    #[test]
    fn parse_accepts_only_one_and_many() {
        assert_eq!(Cardinality::parse("one"), Some(Cardinality::One));
        assert_eq!(Cardinality::parse("many"), Some(Cardinality::Many));
        assert_eq!(Cardinality::parse("ONE"), None);
        assert_eq!(Cardinality::parse("single"), None);
        assert_eq!(Cardinality::parse(""), None);
    }

    #[test]
    fn merge_hint_defers_to_whichever_side_is_set() {
        assert_eq!(merge_cardinality_hint(None, None), None);
        assert_eq!(
            merge_cardinality_hint(None, Some(Cardinality::One)),
            Some(Cardinality::One)
        );
        assert_eq!(
            merge_cardinality_hint(Some(Cardinality::Many), None),
            Some(Cardinality::Many)
        );
    }

    #[test]
    fn merge_hint_agreeing_explicit_hints_stay_put() {
        assert_eq!(
            merge_cardinality_hint(Some(Cardinality::One), Some(Cardinality::One)),
            Some(Cardinality::One)
        );
        assert_eq!(
            merge_cardinality_hint(Some(Cardinality::Many), Some(Cardinality::Many)),
            Some(Cardinality::Many)
        );
    }

    #[test]
    fn merge_hint_conflict_resolves_to_the_safe_many_direction() {
        // An extra row is recoverable; a row dropped by an over-eager
        // pin isn't — so a genuine disagreement between two terms in
        // the same folded group must not silently commit to `One`.
        assert_eq!(
            merge_cardinality_hint(Some(Cardinality::One), Some(Cardinality::Many)),
            Some(Cardinality::Many)
        );
        assert_eq!(
            merge_cardinality_hint(Some(Cardinality::Many), Some(Cardinality::One)),
            Some(Cardinality::Many)
        );
    }
}
