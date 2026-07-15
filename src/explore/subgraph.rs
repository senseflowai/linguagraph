//! ask()/run_dsl: execute a DSL query and materialize its result as an
//! entity subgraph.
//!
//! Query results are tabular; to recover the *graph* behind the rows the
//! explorer rewrites Find-shaped queries before execution, appending a
//! hidden `__id_<alias>` projection for every bound node alias. After
//! execution the distinct ids hydrate into full [`NodeView`]s (T4) plus
//! the edges among them (T5). Aggregate queries have no per-entity
//! bindings — their subgraph is empty by design.
//!
//! Known trade-off: with `distinct: true` the injected id columns widen
//! the projection, so the database may return more raw rows than the
//! original query would. The visible table is re-deduplicated after the
//! hidden columns are stripped; combined with `limit` the visible rows
//! can undercount what the un-rewritten query would return.
//!
//! Before the id injection, [`with_filter_context_returns`] runs a second,
//! visible (non-hidden) rewrite: it adds each filter's field to the
//! projection when the filtered entity is already returned, so an
//! answer-synthesis LLM (and the table itself) can see the value that made
//! a row match — e.g. a `price < 100` filter alongside `return: [name]`
//! gains a `price` column. It no-ops for aggregate and `distinct: true`
//! queries; see its doc comment for the full rule set.
//!
//! The same hidden `__id_<alias>` columns that drive subgraph hydration
//! also answer a second question a UI needs: "which entity does this
//! table cell belong to?". [`visible_table`] repurposes them into
//! [`TableSlice::row_entities`] (per row, `alias -> handle`) instead of
//! only using them to seed the subgraph; [`column_entity_aliases`] maps
//! each visible column to the alias that owns it. A UI joins the two —
//! `row_entities[i][entity_columns[column]]` — to turn any cell into a
//! link, without a second round trip. This is gated by
//! `include_entity_refs` rather than folded into `include_subgraph`,
//! since a plain data-grid UI may want clickable rows without paying for
//! full subgraph hydration.

use std::collections::BTreeMap;
use std::time::Instant;

use serde_json::Value as JsonValue;

use crate::db::{QueryResult, Value as DbValue};
use crate::dsl::schema::ReturnItem;
use crate::dsl::{with_filter_context_returns, DslQuery};
use crate::error::Result;
use crate::nl;

use super::dto::{
    AskOptions, AskResult, EdgeView, NodeView, QueryTrace, SourceRef, Subgraph, TableSlice,
};
use super::{queries, ExploreError, Explorer};

/// Hidden projection prefix for injected node-id columns.
const ID_COLUMN_PREFIX: &str = "__id_";

/// Columns stripped from the visible table: the Cypher builder's
/// auto-injected provenance/score columns plus our hidden id columns.
fn is_hidden_column(name: &str) -> bool {
    name == "score" || name == "sources" || name.starts_with(ID_COLUMN_PREFIX)
}

/// Rewrite a Find query to also project `<alias>.id AS __id_<alias>` for
/// every bound node alias. Returns `None` for aggregate-shaped queries
/// (any aggregate projection or a `group_by`) — id injection would break
/// their grouping semantics.
pub(crate) fn inject_node_id_returns(dsl: &DslQuery) -> Option<DslQuery> {
    let is_aggregate = !dsl.group_by.is_empty()
        || dsl
            .return_
            .iter()
            .any(|item| matches!(item, ReturnItem::Aggregate { .. }));
    if is_aggregate || dsl.return_.is_empty() {
        return None;
    }

    let mut aliases: Vec<&str> = vec![dsl.start.alias.as_str()];
    for traversal in &dsl.traversals {
        if !aliases.contains(&traversal.target.alias.as_str()) {
            aliases.push(traversal.target.alias.as_str());
        }
    }

    let existing_aliases: Vec<&str> = dsl
        .return_
        .iter()
        .filter_map(|item| match item {
            ReturnItem::Field { alias, .. }
            | ReturnItem::Aggregate { alias, .. }
            | ReturnItem::DatePart { alias, .. } => alias.as_deref(),
        })
        .collect();

    let mut rewritten = dsl.clone();
    for alias in aliases {
        let id_alias = format!("{ID_COLUMN_PREFIX}{alias}");
        if existing_aliases.contains(&id_alias.as_str()) {
            continue; // already injected — idempotent
        }
        rewritten.return_.push(ReturnItem::Field {
            field: format!("{alias}.id"),
            alias: Some(id_alias),
        });
    }
    Some(rewritten)
}

/// Extract a stable id string from a `__id_<alias>` cell. Test doubles
/// produce typed cells; the Memgraph client flattens everything to
/// `Json`. Integer-stored ids are stringified to match the public handle
/// type. Empty strings and any other shape mean "no usable id".
fn cell_as_id_string(value: &DbValue) -> Option<String> {
    match value {
        DbValue::String(s) if !s.is_empty() => Some(s.clone()),
        DbValue::Int(v) => Some(v.to_string()),
        DbValue::Json(JsonValue::String(s)) if !s.is_empty() => Some(s.clone()),
        DbValue::Json(JsonValue::Number(n)) => Some(n.to_string()),
        _ => None,
    }
}

/// Map each visible column a Find-shaped query renders to the node alias
/// whose property it projects, e.g. `"price" -> "l"` for
/// `{"field": "l.price", "alias": "price"}`, or `"l.price" -> "l"` when
/// unaliased (the builder names an unaliased field column after the raw
/// `<alias>.<property>` expression). Aggregates, date-part group keys, and
/// bare entity references (no `.<property>`) have no single owning
/// property and are left out.
pub(crate) fn column_entity_aliases(dsl: &DslQuery) -> BTreeMap<String, String> {
    dsl.return_
        .iter()
        .filter_map(|item| match item {
            ReturnItem::Field { field, alias } => {
                let (entity_alias, _prop) = field.split_once('.')?;
                let column = alias.clone().unwrap_or_else(|| field.clone());
                Some((column, entity_alias.to_string()))
            }
            _ => None,
        })
        .collect()
}

impl Explorer {
    /// Shared tail of [`Explorer::ask`] and [`Explorer::run_dsl`]:
    /// compile, execute, and assemble the [`AskResult`].
    pub(crate) async fn run_query_flow(
        &self,
        question: Option<String>,
        dsl: DslQuery,
        llm_attempts: usize,
        started: Instant,
        opts: &AskOptions,
    ) -> Result<AskResult> {
        let with_filters = if opts.include_filter_context {
            with_filter_context_returns(&dsl)
        } else {
            dsl.clone()
        };
        let want_entity_refs = opts.include_subgraph || opts.include_entity_refs;
        let rewritten = if want_entity_refs {
            inject_node_id_returns(&with_filters)
        } else {
            None
        };
        let executed_dsl = rewritten.unwrap_or_else(|| with_filters.clone());

        let compiled = self.pipeline().compile_for_run(executed_dsl.clone()).await?;
        let result = self.pipeline().execute(&compiled).await?;

        let subgraph = if opts.include_subgraph {
            self.materialize_subgraph(&executed_dsl, &result).await?
        } else {
            Subgraph::default()
        };

        let mut table = visible_table(&result, dsl.distinct, opts.max_rows);
        // `row_entities` is already empty here when `want_entity_refs` is
        // false: without it, `inject_node_id_returns` never ran, so the
        // executed query never carried `__id_<alias>` columns to extract
        // from. Only the static column -> alias map needs gating.
        if want_entity_refs {
            table.entity_columns = column_entity_aliases(&with_filters);
        }
        let sources = collect_row_sources(&result);

        let answer = if opts.synthesize_answer {
            let translator = self
                .translator()
                .ok_or(ExploreError::TranslatorMissing)?;
            let query_summary = dsl.describe();
            let question_text = question.clone().unwrap_or_else(|| query_summary.clone());
            Some(
                translator
                    .synthesize_answer(&question_text, &query_summary, &table.rows)
                    .await
                    .map_err(crate::error::Error::from)?,
            )
        } else {
            None
        };

        let trace = QueryTrace {
            question,
            dsl: serde_json::to_value(&executed_dsl)?,
            dsl_summary: dsl.describe(),
            cypher: compiled.text.clone(),
            cypher_params: nl::mask_cypher_params(&compiled.params, false),
            llm_attempts,
            elapsed_ms: started.elapsed().as_millis() as u64,
        };

        Ok(AskResult {
            trace,
            table,
            subgraph,
            sources,
            answer,
        })
    }

    /// Hydrate the `__id_*` bindings of an executed query into a
    /// displayable subgraph: nodes via T4, edges among them via T5.
    async fn materialize_subgraph(
        &self,
        executed_dsl: &DslQuery,
        result: &QueryResult,
    ) -> Result<Subgraph> {
        let mut ids: Vec<String> = Vec::new();
        let mut truncated = false;
        'rows: for row in &result.rows {
            for (column, value) in &row.fields {
                if !column.starts_with(ID_COLUMN_PREFIX) {
                    continue;
                }
                let Some(id) = cell_as_id_string(value) else {
                    continue;
                };
                if ids.contains(&id) {
                    continue;
                }
                if ids.len() >= self.limits().max_subgraph_nodes {
                    truncated = true;
                    break 'rows;
                }
                ids.push(id);
            }
        }
        if ids.is_empty() {
            return Ok(Subgraph::default());
        }

        let prefix = executed_dsl
            .prefix_label
            .as_deref()
            .or_else(|| self.pipeline().prefix_label());
        let catalog = self.pipeline().ontology_catalog();

        let node_result = self
            .pipeline()
            .execute(&queries::nodes_by_ids(&ids, prefix)?)
            .await?;
        let nodes: Vec<NodeView> = node_result
            .rows
            .iter()
            .filter_map(|row| self.node_view_from_row(row, catalog.as_deref()))
            .collect();

        let edge_cap = self.limits().max_subgraph_edges;
        let edge_result = self
            .pipeline()
            .execute(&queries::edges_among(&ids, edge_cap, prefix)?)
            .await?;
        let mut edges: Vec<EdgeView> = Vec::new();
        for row in &edge_result.rows {
            let (Some(from), Some(to), Some(edge_type)) = (
                super::id_string_field(row, "from_id"),
                super::id_string_field(row, "to_id"),
                super::string_field(row, "rel"),
            ) else {
                continue;
            };
            let properties = super::json_object_field(row, "props");
            let confidence = properties.get("confidence").and_then(JsonValue::as_f64);
            let edge = EdgeView {
                id: format!("{from}:{edge_type}:{to}"),
                edge_type,
                from,
                to,
                properties,
                confidence,
            };
            if !edges.iter().any(|e| e.id == edge.id) {
                edges.push(edge);
            }
        }
        truncated = truncated || edge_result.rows.len() >= edge_cap;

        Ok(Subgraph {
            nodes,
            edges,
            truncated,
        })
    }
}

/// Project the raw result into the user-visible table: hidden columns
/// stripped, rows re-deduplicated when the query was `distinct`, capped
/// at `max_rows`.
fn visible_table(result: &QueryResult, distinct: bool, max_rows: Option<u32>) -> TableSlice {
    let mut columns: Vec<String> = result
        .columns
        .iter()
        .map(|c| c.name.clone())
        .filter(|name| !is_hidden_column(name))
        .collect();
    if columns.is_empty() {
        if let Some(row) = result.rows.first() {
            columns = row
                .fields
                .keys()
                .filter(|name| !is_hidden_column(name))
                .cloned()
                .collect();
        }
    }

    // Whether the executed query carried any `__id_<alias>` projection at
    // all — checked once so a query run with entity refs disabled leaves
    // `row_entities` genuinely empty (`Vec::new()`) rather than a run of
    // empty per-row maps, keeping `skip_serializing_if` effective.
    let has_id_columns = result
        .columns
        .iter()
        .any(|c| c.name.starts_with(ID_COLUMN_PREFIX))
        || result
            .rows
            .first()
            .is_some_and(|r| r.fields.keys().any(|k| k.starts_with(ID_COLUMN_PREFIX)));

    let mut rows: Vec<BTreeMap<String, JsonValue>> = Vec::new();
    let mut row_entities: Vec<BTreeMap<String, String>> = Vec::new();
    for row in &result.rows {
        let visible: BTreeMap<String, JsonValue> = row
            .fields
            .iter()
            .filter(|(name, _)| !is_hidden_column(name))
            .map(|(name, value)| (name.clone(), value.to_json()))
            .collect();
        // The injected id columns widen DISTINCT projections; restore the
        // user's dedup semantics on the visible slice.
        if distinct && rows.contains(&visible) {
            continue;
        }
        if has_id_columns {
            // Fold this row's `__id_<alias>` cells into `alias -> handle`,
            // index-aligned with the visible row pushed below.
            let entities: BTreeMap<String, String> = row
                .fields
                .iter()
                .filter_map(|(name, value)| {
                    let alias = name.strip_prefix(ID_COLUMN_PREFIX)?;
                    Some((alias.to_string(), cell_as_id_string(value)?))
                })
                .collect();
            row_entities.push(entities);
        }
        rows.push(visible);
        if let Some(cap) = max_rows {
            if rows.len() >= cap as usize {
                break;
            }
        }
    }
    TableSlice {
        columns,
        rows,
        entity_columns: BTreeMap::new(),
        row_entities,
    }
}

/// Dedup the provenance refs from the builder's auto-injected `sources`
/// column (a JSON list of `{id, name, …}` maps per row).
fn collect_row_sources(result: &QueryResult) -> Vec<SourceRef> {
    let mut refs: Vec<SourceRef> = Vec::new();
    for row in &result.rows {
        let Some(DbValue::Json(JsonValue::Array(items))) = row.fields.get("sources") else {
            continue;
        };
        for item in items {
            let Some(map) = item.as_object() else {
                continue;
            };
            let source = SourceRef {
                id: map.get("id").and_then(|v| v.as_str()).map(str::to_string),
                name: map.get("name").and_then(|v| v.as_str()).map(str::to_string),
            };
            if (source.id.is_some() || source.name.is_some()) && !refs.contains(&source) {
                refs.push(source);
            }
        }
    }
    refs.sort();
    refs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::{Column, Row};
    use crate::dsl;
    use serde_json::json;

    fn find_dsl() -> DslQuery {
        dsl::parse_str(
            r#"{
                "start": { "label": "Person", "alias": "p" },
                "traversals": [
                    { "edge": { "label": "ACTED_IN", "alias": "r", "direction": "out" },
                      "target": { "label": "Movie", "alias": "m" } }
                ],
                "return": [ { "field": "p.name", "alias": "name" } ]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn inject_adds_id_projection_per_node_alias() {
        let rewritten = inject_node_id_returns(&find_dsl()).expect("find query rewrites");
        let aliases: Vec<_> = rewritten
            .return_
            .iter()
            .filter_map(|item| match item {
                ReturnItem::Field { field, alias } => Some((field.clone(), alias.clone())),
                _ => None,
            })
            .collect();
        assert!(aliases.contains(&("p.id".to_string(), Some("__id_p".to_string()))));
        assert!(aliases.contains(&("m.id".to_string(), Some("__id_m".to_string()))));
        assert_eq!(rewritten.return_.len(), 3);
    }

    #[test]
    fn inject_is_idempotent() {
        let once = inject_node_id_returns(&find_dsl()).unwrap();
        let twice = inject_node_id_returns(&once).unwrap();
        assert_eq!(once.return_.len(), twice.return_.len());
    }

    #[test]
    fn inject_skips_aggregates_and_group_by() {
        let aggregate = dsl::parse_str(
            r#"{
                "start": { "label": "Person", "alias": "p" },
                "return": [ { "aggregate": "count", "field": "p", "alias": "total" } ]
            }"#,
        )
        .unwrap();
        assert!(inject_node_id_returns(&aggregate).is_none());
    }

    #[test]
    fn visible_table_strips_hidden_columns_and_restores_distinct() {
        let mut row_a = Row::default();
        row_a.fields.insert("name".into(), DbValue::String("Keanu".into()));
        row_a.fields.insert("__id_p".into(), DbValue::String("p1".into()));
        row_a.fields.insert("score".into(), DbValue::Float(0.9));
        row_a
            .fields
            .insert("sources".into(), DbValue::Json(json!([{"id": "s1"}])));
        let mut row_b = Row::default();
        row_b.fields.insert("name".into(), DbValue::String("Keanu".into()));
        row_b.fields.insert("__id_p".into(), DbValue::String("p2".into()));

        let result = QueryResult {
            columns: vec![
                Column::new("name"),
                Column::new("__id_p"),
                Column::new("score"),
                Column::new("sources"),
            ],
            rows: vec![row_a, row_b],
        };

        let table = visible_table(&result, true, None);
        assert_eq!(table.columns, vec!["name".to_string()]);
        assert_eq!(table.rows.len(), 1, "distinct restored after stripping");
        // Distinct collapses onto the first matching row — its entity
        // handle (`p1`, not the discarded duplicate's `p2`) is what survives.
        assert_eq!(
            table.row_entities,
            vec![BTreeMap::from([("p".to_string(), "p1".to_string())])]
        );

        let plain = visible_table(&result, false, None);
        assert_eq!(plain.rows.len(), 2);
        assert_eq!(
            plain.row_entities,
            vec![
                BTreeMap::from([("p".to_string(), "p1".to_string())]),
                BTreeMap::from([("p".to_string(), "p2".to_string())]),
            ],
            "row_entities stays index-aligned with rows"
        );

        let capped = visible_table(&result, false, Some(1));
        assert_eq!(capped.rows.len(), 1);
        assert_eq!(capped.row_entities.len(), 1);
    }

    #[test]
    fn column_entity_aliases_maps_aliased_and_raw_fields_and_skips_non_fields() {
        let dsl = dsl::parse_str(
            r#"{
                "start": { "label": "Listing", "alias": "l" },
                "return": [
                    { "field": "l.price", "alias": "price" },
                    { "field": "l.title" },
                    { "field": "l.created_at", "date_part": "year", "alias": "created_year" },
                    { "aggregate": "count", "field": "l", "alias": "total" }
                ]
            }"#,
        )
        .unwrap();

        let mapping = column_entity_aliases(&dsl);
        assert_eq!(mapping.get("price").map(String::as_str), Some("l"));
        assert_eq!(mapping.get("l.title").map(String::as_str), Some("l"));
        assert!(
            !mapping.contains_key("created_year"),
            "date_part group keys have no single owning property"
        );
        assert!(
            !mapping.contains_key("total"),
            "aggregates have no single owning entity"
        );
        assert_eq!(mapping.len(), 2);
    }

    #[test]
    fn collect_row_sources_dedups_across_rows() {
        let mut row_a = Row::default();
        row_a.fields.insert(
            "sources".into(),
            DbValue::Json(json!([{"id": "s1", "name": "doc"}])),
        );
        let mut row_b = Row::default();
        row_b.fields.insert(
            "sources".into(),
            DbValue::Json(json!([{"id": "s1", "name": "doc"}, {"id": null, "name": null}])),
        );
        let result = QueryResult {
            columns: Vec::new(),
            rows: vec![row_a, row_b],
        };
        let sources = collect_row_sources(&result);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].id.as_deref(), Some("s1"));
    }
}
