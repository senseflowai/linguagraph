//! Normalization passes over [`super::plan::LogicalPlan`].
//!
//! Each pass is a `fn(&mut LogicalPlan, &Opts) -> Result<(), Error>`
//! that transforms a plan in place. Passes are composable and
//! order-sensitive — the pipeline picks an order, runs them, then
//! emits Cypher.
//!
//! Today this module contains one pass: source-projection injection.
//! That logic lives hardcoded inside [`super::cypher::build_read_with`]
//! (the `write_sources_stage` helper); pulling it out as a plan
//! transform decouples the *generic* read builder from the
//! *specific* ingest model (`:Source` / `:mention` / `:part_of`) and
//! makes the pass testable in isolation.
//!
//! The emitter still runs the hardcoded path by default — a follow-up
//! step (rewiring `build_read_with` through `LogicalPlan`) swaps the
//! producer side.

use super::plan::{Clause, LogicalPlan, NodePattern, ProjectClause, ReturnItem};

/// Configuration for [`inject_sources_projection`]. Default values
/// match the built-in document-ingestion labels (Source / mention /
/// part_of); custom graphs override.
#[derive(Debug, Clone)]
pub struct SourceProjectionOpts {
    pub source_label: String,
    pub mention_rel: String,
    pub part_of_rel: String,
    /// Column name to expose the projection under. Defaults to
    /// [`super::cypher::SOURCES_COLUMN`].
    pub column: String,
}

impl Default for SourceProjectionOpts {
    fn default() -> Self {
        Self {
            source_label: crate::graph::SOURCE_LABEL.to_string(),
            mention_rel: crate::graph::MENTION_REL.to_string(),
            part_of_rel: crate::graph::PART_OF_REL.to_string(),
            column: super::SOURCES_COLUMN.to_string(),
        }
    }
}

/// Inject a `WITH ... OPTIONAL MATCH ... WITH ..., collect(DISTINCT
/// __src__) AS __sources__` chain plus a `__sources__ AS sources`
/// item in the projection.
///
/// Idempotent: if the plan already projects a column with
/// `opts.column` as its alias, the pass is a no-op.
///
/// Aggregate-style plans (those containing a `ReturnItem::Aggregate`)
/// are skipped, mirroring today's `build_read_with` behavior: a
/// per-row Sources list has no well-defined aggregation, so the
/// emitter omits the projection.
pub fn inject_sources_projection(plan: &mut LogicalPlan, opts: &SourceProjectionOpts) {
    if has_sources_alias(plan, &opts.column) {
        return;
    }
    if plan_is_aggregate(plan) {
        return;
    }
    let node_aliases = collect_node_aliases(plan);
    if node_aliases.is_empty() {
        return;
    }

    // Insert a `WITH ... OPTIONAL MATCH ... WITH ...` chain immediately
    // before the Project clause. We model the whole chain as one
    // `With` clause carrying the rendered text — the chain is locally
    // textual; future passes can refine it into structured sub-clauses.
    let carry = node_aliases.join(", ");
    let list = format!("[{carry}]");
    let chain = format!(
        "WITH {carry}\n\
         OPTIONAL MATCH (__src__:{src})<-[:{mention}|{part_of}]-(__sn__)\n\
         WHERE __sn__ IN {list}\n\
         WITH {carry}, collect(DISTINCT __src__) AS __sources__",
        src = opts.source_label,
        mention = opts.mention_rel,
        part_of = opts.part_of_rel,
    );

    let project_idx = plan
        .clauses
        .iter()
        .position(|c| matches!(c, Clause::Project(_)));
    let insert_at = match project_idx {
        Some(i) => i,
        None => plan.clauses.len(),
    };
    plan.clauses.insert(
        insert_at,
        Clause::With(super::plan::WithClause { text: chain }),
    );

    // Now extend the project clause with the sources column.
    if let Some(Clause::Project(p)) = plan
        .clauses
        .iter_mut()
        .find(|c| matches!(c, Clause::Project(_)))
    {
        p.items.push(ReturnItem::Expr {
            expr: "__sources__".into(),
            alias: Some(opts.column.clone()),
        });
    } else {
        // No project clause yet — synthesize one.
        plan.clauses.push(Clause::Project(ProjectClause {
            items: vec![ReturnItem::Expr {
                expr: "__sources__".into(),
                alias: Some(opts.column.clone()),
            }],
        }));
    }
}

fn has_sources_alias(plan: &LogicalPlan, alias: &str) -> bool {
    plan.clauses.iter().any(|c| match c {
        Clause::Project(p) => p.items.iter().any(|i| match i {
            ReturnItem::Field { alias: a, .. }
            | ReturnItem::Aggregate { alias: a, .. }
            | ReturnItem::Expr { alias: a, .. } => a.as_deref() == Some(alias),
        }),
        _ => false,
    })
}

fn plan_is_aggregate(plan: &LogicalPlan) -> bool {
    plan.clauses.iter().any(|c| match c {
        Clause::Project(p) => p
            .items
            .iter()
            .any(|i| matches!(i, ReturnItem::Aggregate { .. })),
        _ => false,
    })
}

fn collect_node_aliases(plan: &LogicalPlan) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mut push = |n: &NodePattern, out: &mut Vec<String>| {
        if seen.insert(n.alias.clone()) {
            out.push(n.alias.clone());
        }
    };
    for c in &plan.clauses {
        match c {
            Clause::Match(m) | Clause::OptionalMatch(m) => {
                push(&m.start, &mut out);
                for t in &m.tail {
                    push(&t.target, &mut out);
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::query::*;
    use crate::builder::plan::LogicalPlan;

    fn alias(s: &str) -> Alias {
        Alias::new(s)
    }

    fn pref(a: &str, p: Option<&str>) -> PropertyRef {
        PropertyRef {
            alias: alias(a),
            property: p.map(str::to_string),
        }
    }

    fn find_query() -> ReadQuery {
        ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
            },
            traversals: vec![],
            filter: None,
            returns: vec![ReturnClause::Field {
                field: pref("p", Some("name")),
                alias: Some("name".into()),
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        }
    }

    fn aggregate_query() -> ReadQuery {
        ReadQuery {
            action: Action::Aggregate,
            start: Node {
                label: "Order".into(),
                alias: alias("o"),
            },
            traversals: vec![],
            filter: None,
            returns: vec![ReturnClause::Aggregate {
                func: AggregateFn::Count,
                field: pref("o", None),
                alias: Some("n".into()),
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        }
    }

    #[test]
    fn injects_with_chain_and_sources_column_for_find_plans() {
        let mut plan = LogicalPlan::from_read_query(&find_query());
        inject_sources_projection(&mut plan, &SourceProjectionOpts::default());

        // Find a With clause that mentions our source label.
        let with_text = plan
            .clauses
            .iter()
            .find_map(|c| match c {
                Clause::With(w) if w.text.contains("__src__:") => Some(&w.text),
                _ => None,
            })
            .expect("WITH chain present");
        assert!(with_text.contains("OPTIONAL MATCH"));
        assert!(with_text.contains("collect(DISTINCT __src__) AS __sources__"));

        // Sources column added to the projection.
        let project = plan
            .clauses
            .iter()
            .find_map(|c| match c {
                Clause::Project(p) => Some(p),
                _ => None,
            })
            .unwrap();
        let aliases: Vec<&str> = project
            .items
            .iter()
            .filter_map(|i| match i {
                ReturnItem::Field { alias, .. }
                | ReturnItem::Aggregate { alias, .. }
                | ReturnItem::Expr { alias, .. } => alias.as_deref(),
            })
            .collect();
        assert!(aliases.contains(&"sources"));
    }

    #[test]
    fn aggregate_plans_are_skipped() {
        let mut plan = LogicalPlan::from_read_query(&aggregate_query());
        let before = plan.clauses.len();
        inject_sources_projection(&mut plan, &SourceProjectionOpts::default());
        let after = plan.clauses.len();
        assert_eq!(before, after, "aggregate plans get no sources injection");
    }

    #[test]
    fn idempotent_when_alias_already_present() {
        let mut plan = LogicalPlan::from_read_query(&find_query());
        // Pre-populate a `sources` projection alias.
        if let Some(Clause::Project(p)) = plan
            .clauses
            .iter_mut()
            .find(|c| matches!(c, Clause::Project(_)))
        {
            p.items.push(ReturnItem::Expr {
                expr: "[]".into(),
                alias: Some("sources".into()),
            });
        }
        let before = plan.clauses.len();
        inject_sources_projection(&mut plan, &SourceProjectionOpts::default());
        let after = plan.clauses.len();
        assert_eq!(
            before, after,
            "pass is a no-op when sources is already projected"
        );
    }
}
