//! Auto-enrich a DSL query's projection with the fields its own filters
//! reference, so a downstream answer-synthesis LLM can see the values that
//! satisfied a filter instead of just being told about it in prose.
//!
//! Example: `filters: [{"field": "p.price", "op": "lt", "value": 100}]` with
//! `return: [{"field": "p.name"}]` gains a `p.price` column — the rows
//! already prove every product is under $100, but without this the answer
//! synthesizer has no way to *show* the price that made each row match.

use std::collections::HashSet;

use super::schema::{DslQuery, ReturnItem};

/// Properties the ontology prompt never shows the LLM (mirrors
/// `crate::prompt::generator::SCHEMA_HIDDEN_PROPS`). A well-behaved DSL
/// never filters on these, but skipping them here is cheap insurance
/// against a malformed or hand-authored query smuggling one through.
const HIDDEN_PROPS: &[&str] = &["_canonical", "entity_id", "primary_key"];

/// Add each plain (untyped) filter's field to `return_` when its alias is
/// already projected by another return item — i.e. the filtered entity is
/// already part of the response, it's just missing the one column that
/// explains why a row matched.
///
/// No-ops for:
/// * aggregate queries — adding an ungrouped column would violate Cypher's
///   GROUP BY rules (see `resolve::ast::enforce_aggregation_rules`);
/// * `distinct: true` queries — widening the projection changes the dedup
///   key, silently un-collapsing rows that were meant to be merged;
/// * typed filters (`field_type` set) — the field's runtime shape depends
///   on the type handler (e.g. `SemanticText` may cover a whole document
///   body) and isn't safe to echo back unconditionally.
///
/// Idempotent: a field already projected under any alias is never
/// duplicated, so calling this more than once (or after another DSL
/// rewrite pass) is safe.
pub fn with_filter_context_returns(dsl: &DslQuery) -> DslQuery {
    let is_aggregate = !dsl.group_by.is_empty()
        || dsl
            .return_
            .iter()
            .any(|item| matches!(item, ReturnItem::Aggregate { .. }));
    if is_aggregate || dsl.distinct || dsl.return_.is_empty() || dsl.filters.is_empty() {
        return dsl.clone();
    }

    let projected_aliases: HashSet<&str> = dsl
        .return_
        .iter()
        .filter_map(|item| match item {
            ReturnItem::Field { field, .. } | ReturnItem::DatePart { field, .. } => {
                field.split('.').next()
            }
            ReturnItem::Aggregate { .. } => None,
        })
        .collect();

    let mut projected_fields: HashSet<&str> = dsl
        .return_
        .iter()
        .filter_map(|item| match item {
            ReturnItem::Field { field, .. } => Some(field.as_str()),
            _ => None,
        })
        .collect();

    let mut rewritten = dsl.clone();
    for filter in &dsl.filters {
        if filter.field_type.is_some() {
            continue;
        }
        if projected_fields.contains(filter.field.as_str()) {
            continue;
        }
        let Some((alias, prop)) = filter.field.split_once('.') else {
            continue;
        };
        if HIDDEN_PROPS.contains(&prop) {
            continue;
        }
        if !projected_aliases.contains(alias) {
            continue;
        }
        rewritten.return_.push(ReturnItem::Field {
            field: filter.field.clone(),
            alias: None,
        });
        // Guard against two filters on the same field (e.g. a `between`
        // expressed as `gte` + `lt`) adding the column twice.
        projected_fields.insert(filter.field.as_str());
    }
    rewritten
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dsl(json: &str) -> DslQuery {
        crate::dsl::parse_str(json).unwrap()
    }

    fn field_returns(dsl: &DslQuery) -> Vec<(String, Option<String>)> {
        dsl.return_
            .iter()
            .filter_map(|item| match item {
                ReturnItem::Field { field, alias } => Some((field.clone(), alias.clone())),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn adds_filtered_field_when_entity_already_projected() {
        let q = dsl(
            r#"{
                "start": { "label": "Product", "alias": "p" },
                "filters": [ { "field": "p.price", "op": "lt", "value": 100 } ],
                "return": [ { "field": "p.name" } ]
            }"#,
        );
        let rewritten = with_filter_context_returns(&q);
        let returns = field_returns(&rewritten);
        assert!(returns.contains(&("p.name".to_string(), None)));
        assert!(returns.contains(&("p.price".to_string(), None)));
        assert_eq!(rewritten.return_.len(), 2);
    }

    #[test]
    fn skips_filter_whose_alias_is_not_in_the_projection() {
        let q = dsl(
            r#"{
                "start": { "label": "Product", "alias": "p" },
                "traversals": [
                    { "edge": { "label": "IN_CATEGORY", "alias": "r", "direction": "out" },
                      "target": { "label": "Category", "alias": "c" } }
                ],
                "filters": [ { "field": "c.name", "op": "eq", "value": "Electronics" } ],
                "return": [ { "field": "p.name" } ]
            }"#,
        );
        let rewritten = with_filter_context_returns(&q);
        assert_eq!(field_returns(&rewritten), vec![("p.name".to_string(), None)]);
    }

    #[test]
    fn does_not_duplicate_a_field_already_projected_under_another_alias() {
        let q = dsl(
            r#"{
                "start": { "label": "Person", "alias": "p" },
                "filters": [ { "field": "p.name", "op": "eq", "value": "Keanu Reeves" } ],
                "return": [ { "field": "p.name", "alias": "name" } ]
            }"#,
        );
        let rewritten = with_filter_context_returns(&q);
        assert_eq!(rewritten.return_.len(), 1, "no duplicate p.name column added");
    }

    #[test]
    fn skips_aggregate_and_group_by_queries() {
        let aggregate = dsl(
            r#"{
                "start": { "label": "Product", "alias": "p" },
                "filters": [ { "field": "p.price", "op": "lt", "value": 100 } ],
                "return": [ { "aggregate": "count", "field": "p", "alias": "total" } ]
            }"#,
        );
        assert_eq!(
            with_filter_context_returns(&aggregate).return_.len(),
            aggregate.return_.len()
        );

        let grouped = dsl(
            r#"{
                "start": { "label": "Product", "alias": "p" },
                "filters": [ { "field": "p.price", "op": "lt", "value": 100 } ],
                "return": [ { "field": "p.category" } ],
                "group_by": [ "p.category" ]
            }"#,
        );
        assert_eq!(
            with_filter_context_returns(&grouped).return_.len(),
            grouped.return_.len()
        );
    }

    #[test]
    fn skips_distinct_queries() {
        let q = dsl(
            r#"{
                "start": { "label": "Product", "alias": "p" },
                "filters": [ { "field": "p.price", "op": "lt", "value": 100 } ],
                "return": [ { "field": "p.name" } ],
                "distinct": true
            }"#,
        );
        assert_eq!(with_filter_context_returns(&q).return_.len(), 1);
    }

    #[test]
    fn skips_typed_filters() {
        let q = dsl(
            r#"{
                "start": { "label": "Document", "alias": "d" },
                "filters": [ { "field": "d.body", "op": "search", "value": "invoice", "type": "SemanticText" } ],
                "return": [ { "field": "d.title" } ]
            }"#,
        );
        assert_eq!(with_filter_context_returns(&q).return_.len(), 1);
    }

    #[test]
    fn is_idempotent() {
        let q = dsl(
            r#"{
                "start": { "label": "Product", "alias": "p" },
                "filters": [ { "field": "p.price", "op": "lt", "value": 100 } ],
                "return": [ { "field": "p.name" } ]
            }"#,
        );
        let once = with_filter_context_returns(&q);
        let twice = with_filter_context_returns(&once);
        assert_eq!(once.return_.len(), twice.return_.len());
    }

    #[test]
    fn dedupes_two_filters_on_the_same_field() {
        let q = dsl(
            r#"{
                "start": { "label": "Product", "alias": "p" },
                "filters": [
                    { "field": "p.price", "op": "gte", "value": 50 },
                    { "field": "p.price", "op": "lt", "value": 100 }
                ],
                "return": [ { "field": "p.name" } ]
            }"#,
        );
        let rewritten = with_filter_context_returns(&q);
        assert_eq!(rewritten.return_.len(), 2, "price added exactly once");
    }
}
