//! Staged decision pipeline that scores each incoming candidate against
//! its top-K hits and routes the result to `AutoMerge` / `NeedsReview`
//! / `NoMerge`.
//!
//! Design: false splits are recoverable; false merges destroy data. We
//! evaluate every gate (no short-circuit) so a `NeedsReview` record can
//! tell a reviewer *every* reason an AutoMerge was rejected — gate
//! tuning over real data is the primary downstream use case.

use std::collections::HashMap;

use serde_json::Value as JsonValue;

use crate::config::SoftMergeConfig;
use crate::graph::Property;

use super::lexical::{lexical_score, primary_name_of};
use super::query::Hit;

/// What we'll do with one incoming candidate after looking at its hits.
#[derive(Debug, Clone)]
pub(super) enum Decision {
    /// Safe to rewrite the candidate's soft-merge field to `canonical`
    /// — the standard MERGE will collapse the rows.
    AutoMerge {
        canonical: String,
        hit_id: i64,
        top_score: f64,
    },
    /// Plausible match but at least one gate failed. The caller
    /// surfaces these in `SoftMergeReport.review_candidates`.
    NeedsReview {
        top: ReviewHit,
        runners_up: Vec<ReviewHit>,
        rejected_by: Vec<GateReason>,
    },
    /// No usable hit — leave the candidate alone and create a new node.
    NoMerge,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReviewHit {
    pub hit_id: i64,
    pub score: f64,
    pub canonical: String,
    pub lexical: f64,
}

/// Reason an AutoMerge decision was rejected. Stable JSON shape for
/// downstream auditing — gate name plus the numbers that drove the
/// rejection so reviewers can tune thresholds without re-running the
/// resolver.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GateReason {
    BelowAutoMergeThreshold { top: f64, required: f64 },
    InsufficientMargin { margin: f64, required: f64 },
    InsufficientLexical { lexical: f64, required: f64 },
    TooManyCloseCandidates { count: usize, allowed: usize },
    HardConflict {
        property: String,
        incoming: String,
        candidate: String,
    },
    TypeOnly,
}

pub(super) struct CandidateInfo<'a> {
    pub canonical_text: &'a str,
    pub props: &'a HashMap<String, Property>,
}

/// Classify a single candidate. `hits` must already be sorted by score
/// descending (the Cypher does this).
pub(super) fn classify(
    cand: &CandidateInfo<'_>,
    hits: &[Hit],
    cfg: &SoftMergeConfig,
) -> Decision {
    // Gate 0: no candidate at all.
    if hits.is_empty() {
        return Decision::NoMerge;
    }
    let top = &hits[0];
    // Always compute lexical against the top hit — used for both the
    // AutoMerge gate and the review payload.
    let top_lex = lexical_score(cand.canonical_text, &top.canonical);

    // Below `review_threshold` the candidate is clearly distinct;
    // don't even surface it as a review record.
    if top.score < cfg.review_threshold {
        return Decision::NoMerge;
    }

    let mut rejected: Vec<GateReason> = Vec::new();

    // Gate 1: AutoMerge floor.
    if top.score < cfg.auto_merge_threshold {
        rejected.push(GateReason::BelowAutoMergeThreshold {
            top: top.score,
            required: cfg.auto_merge_threshold,
        });
    }

    // Gate 2: margin (top1 - top2). Single-hit cases have effectively
    // infinite margin and pass.
    let margin = if hits.len() >= 2 {
        top.score - hits[1].score
    } else {
        f64::INFINITY
    };
    if margin < cfg.min_margin {
        rejected.push(GateReason::InsufficientMargin {
            margin,
            required: cfg.min_margin,
        });
    }

    // Gate 3: lexical.
    if top_lex < cfg.min_lexical_similarity {
        rejected.push(GateReason::InsufficientLexical {
            lexical: top_lex,
            required: cfg.min_lexical_similarity,
        });
    }

    // Gate 4: ambiguity. How many runner-up hits sit within
    // `close_candidate_delta` of the top score. Top itself excluded.
    let close_count = hits
        .iter()
        .skip(1)
        .take_while(|h| top.score - h.score <= cfg.close_candidate_delta)
        .count();
    if close_count > cfg.max_close_candidates {
        rejected.push(GateReason::TooManyCloseCandidates {
            count: close_count,
            allowed: cfg.max_close_candidates,
        });
    }

    // Gate 5: hard conflict on a disambiguating property.
    if let Some(conflict) = detect_hard_conflict(cand.props, &top.props, cfg) {
        rejected.push(conflict);
    }

    // Gate 6: type-only canonical. Only triggers for `_canonical`-style
    // soft keys (no `type:` prefix → never triggered).
    if !cfg.allow_type_only_auto_merge && is_type_only(cand.canonical_text) {
        rejected.push(GateReason::TypeOnly);
    }

    if rejected.is_empty() {
        return Decision::AutoMerge {
            canonical: top.canonical.clone(),
            hit_id: top.id,
            top_score: top.score,
        };
    }

    let runners_up_cap = cfg.review_max_candidates.saturating_sub(1);
    let runners_up: Vec<ReviewHit> = hits
        .iter()
        .skip(1)
        .take(runners_up_cap)
        .map(|h| ReviewHit {
            hit_id: h.id,
            score: h.score,
            canonical: h.canonical.clone(),
            lexical: lexical_score(cand.canonical_text, &h.canonical),
        })
        .collect();
    Decision::NeedsReview {
        top: ReviewHit {
            hit_id: top.id,
            score: top.score,
            canonical: top.canonical.clone(),
            lexical: top_lex,
        },
        runners_up,
        rejected_by: rejected,
    }
}

/// Detect a hard conflict on any of `cfg.conflict_properties`. Both
/// sides must have a non-null, non-empty value that differs as strings
/// (case-sensitive) for it to count. First match wins so the review
/// reason names a specific property.
fn detect_hard_conflict(
    incoming: &HashMap<String, Property>,
    candidate: &serde_json::Map<String, JsonValue>,
    cfg: &SoftMergeConfig,
) -> Option<GateReason> {
    for key in &cfg.conflict_properties {
        let inc = incoming.get(key).and_then(|p| json_to_compare_string(&p.value));
        let hit = candidate.get(key).and_then(json_to_compare_string);
        if let (Some(a), Some(b)) = (inc, hit) {
            if !a.is_empty() && !b.is_empty() && a != b {
                return Some(GateReason::HardConflict {
                    property: key.clone(),
                    incoming: a,
                    candidate: b,
                });
            }
        }
    }
    None
}

fn json_to_compare_string(v: &JsonValue) -> Option<String> {
    match v {
        JsonValue::Null => None,
        JsonValue::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

/// True iff the incoming canonical text consists of nothing but a
/// `type: X` line. Such candidates have no distinguishing properties
/// to anchor a merge to — auto-merging them would collapse every
/// type-only mention onto whichever node of that type embeds nearest.
pub(super) fn is_type_only(canonical: &str) -> bool {
    let mut lines = canonical.lines().filter(|l| !l.trim().is_empty());
    let first = lines.next();
    let second = lines.next();
    matches!(first, Some(l) if l.starts_with("type: ")) && second.is_none()
}

/// Look up the primary name of the incoming canonical — exported so
/// the review-record builder in `mod.rs` can populate
/// `ReviewCandidate.incoming_value` with the same string the lexical
/// gate compared against.
pub(super) fn incoming_name(canonical: &str) -> String {
    primary_name_of(canonical).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PropertyType;
    use serde_json::json;

    fn strict_cfg() -> SoftMergeConfig {
        SoftMergeConfig::default()
    }

    fn hit(score: f64, canonical: &str) -> Hit {
        Hit {
            id: 1,
            score,
            canonical: canonical.into(),
            props: serde_json::Map::new(),
        }
    }

    fn hit_with_props(score: f64, canonical: &str, props: serde_json::Value) -> Hit {
        let props = match props {
            serde_json::Value::Object(m) => m,
            _ => panic!("hit_with_props requires a JSON object"),
        };
        Hit {
            id: 42,
            score,
            canonical: canonical.into(),
            props,
        }
    }

    fn empty_props() -> HashMap<String, Property> {
        HashMap::new()
    }

    fn props_with(pairs: Vec<(&str, serde_json::Value)>) -> HashMap<String, Property> {
        pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), Property::new(k, PropertyType::Keyword, v)))
            .collect()
    }

    #[test]
    fn automerge_when_all_gates_pass() {
        let cand = CandidateInfo {
            canonical_text: "Microsoft",
            props: &empty_props(),
        };
        let hits = vec![hit(0.97, "Microsoft")];
        let d = classify(&cand, &hits, &strict_cfg());
        match d {
            Decision::AutoMerge {
                canonical,
                hit_id: _,
                top_score,
            } => {
                assert_eq!(canonical, "Microsoft");
                assert!((top_score - 0.97).abs() < 1e-9);
            }
            other => panic!("expected AutoMerge, got {other:?}"),
        }
    }

    #[test]
    fn below_review_threshold_is_no_merge() {
        let cand = CandidateInfo {
            canonical_text: "Acme",
            props: &empty_props(),
        };
        let hits = vec![hit(0.70, "Other")];
        assert!(matches!(
            classify(&cand, &hits, &strict_cfg()),
            Decision::NoMerge
        ));
    }

    #[test]
    fn between_review_and_auto_is_review() {
        let cand = CandidateInfo {
            canonical_text: "Microsoft",
            props: &empty_props(),
        };
        let hits = vec![hit(0.90, "Microsoft")];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::NeedsReview { rejected_by, .. } => {
                assert!(rejected_by.iter().any(|r| matches!(
                    r,
                    GateReason::BelowAutoMergeThreshold { .. }
                )));
            }
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn margin_gate_blocks_automerge() {
        let cand = CandidateInfo {
            canonical_text: "Microsoft",
            props: &empty_props(),
        };
        // top=0.97, second=0.94 → margin 0.03 < default 0.08.
        let hits = vec![hit(0.97, "Microsoft"), hit(0.94, "Apple Microsoft")];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::NeedsReview { rejected_by, .. } => assert!(rejected_by
                .iter()
                .any(|r| matches!(r, GateReason::InsufficientMargin { .. }))),
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn lexical_gate_blocks_automerge() {
        // Embedder says 0.99 but names are visually unrelated.
        let cand = CandidateInfo {
            canonical_text: "Alice Smith",
            props: &empty_props(),
        };
        let hits = vec![hit(0.99, "Бенедикт Иванович")];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::NeedsReview { rejected_by, .. } => assert!(rejected_by
                .iter()
                .any(|r| matches!(r, GateReason::InsufficientLexical { .. }))),
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn close_candidates_gate_blocks_automerge() {
        // top=0.97 and TWO runners-up within 0.03.
        let cand = CandidateInfo {
            canonical_text: "Microsoft",
            props: &empty_props(),
        };
        let hits = vec![
            hit(0.97, "Microsoft"),
            hit(0.96, "Microsoft Corp"),
            hit(0.95, "Microsoft Inc"),
        ];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::NeedsReview { rejected_by, .. } => assert!(rejected_by
                .iter()
                .any(|r| matches!(r, GateReason::TooManyCloseCandidates { .. }))),
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn hard_conflict_email_blocks_automerge() {
        let incoming = props_with(vec![("email", json!("a@example.com"))]);
        let cand = CandidateInfo {
            canonical_text: "Alice",
            props: &incoming,
        };
        let hits = vec![hit_with_props(
            0.99,
            "Alice",
            json!({"email": "b@example.com"}),
        )];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::NeedsReview { rejected_by, .. } => assert!(rejected_by
                .iter()
                .any(|r| matches!(r, GateReason::HardConflict { property, .. } if property == "email"))),
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn hard_conflict_one_side_null_does_not_block() {
        let incoming = props_with(vec![("email", json!("a@example.com"))]);
        let cand = CandidateInfo {
            canonical_text: "Alice",
            props: &incoming,
        };
        // Candidate has no `email` at all → not a hard conflict.
        let hits = vec![hit_with_props(0.99, "Alice", json!({}))];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::AutoMerge { .. } => {}
            other => panic!("expected AutoMerge, got {other:?}"),
        }
    }

    #[test]
    fn type_only_blocks_automerge_by_default() {
        let cand = CandidateInfo {
            canonical_text: "type: Person",
            props: &empty_props(),
        };
        let hits = vec![hit(0.99, "type: Person\nname: Alice")];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::NeedsReview { rejected_by, .. } => assert!(rejected_by
                .iter()
                .any(|r| matches!(r, GateReason::TypeOnly))),
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn type_only_allowed_when_configured() {
        let cfg = SoftMergeConfig {
            allow_type_only_auto_merge: true,
            ..SoftMergeConfig::default()
        };
        let cand = CandidateInfo {
            canonical_text: "type: Person",
            props: &empty_props(),
        };
        // Use a hit that also passes lexical (both are "Person").
        let hits = vec![hit(0.99, "type: Person")];
        match classify(&cand, &hits, &cfg) {
            Decision::AutoMerge { .. } => {}
            other => panic!("expected AutoMerge, got {other:?}"),
        }
    }

    #[test]
    fn all_gates_failing_records_every_reason() {
        let incoming = props_with(vec![("email", json!("a@example.com"))]);
        let cand = CandidateInfo {
            canonical_text: "type: Person",
            props: &incoming,
        };
        // top=0.90 (below auto), runner-up at 0.89 (margin 0.01 <
        // 0.08, AND counts as a close candidate), unrelated names →
        // lexical fails, email differs → hard conflict, type-only.
        let hits = vec![
            hit_with_props(0.90, "Xyz", json!({"email": "b@example.com"})),
            hit_with_props(0.89, "Pqr", json!({})),
        ];
        let cfg = SoftMergeConfig {
            max_close_candidates: 0,
            ..strict_cfg()
        };
        let d = classify(&cand, &hits, &cfg);
        match d {
            Decision::NeedsReview { rejected_by, .. } => {
                let kinds: Vec<&str> = rejected_by
                    .iter()
                    .map(|r| match r {
                        GateReason::BelowAutoMergeThreshold { .. } => "below_auto",
                        GateReason::InsufficientMargin { .. } => "margin",
                        GateReason::InsufficientLexical { .. } => "lexical",
                        GateReason::TooManyCloseCandidates { .. } => "close",
                        GateReason::HardConflict { .. } => "conflict",
                        GateReason::TypeOnly => "type_only",
                    })
                    .collect();
                for expected in [
                    "below_auto",
                    "margin",
                    "lexical",
                    "close",
                    "conflict",
                    "type_only",
                ] {
                    assert!(
                        kinds.contains(&expected),
                        "expected {expected} in {kinds:?}"
                    );
                }
            }
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn runners_up_capped_at_review_max_candidates() {
        let cfg = SoftMergeConfig {
            review_max_candidates: 3,
            // Force a review (set auto threshold > top).
            auto_merge_threshold: 0.99,
            ..SoftMergeConfig::default()
        };
        let cand = CandidateInfo {
            canonical_text: "Microsoft",
            props: &empty_props(),
        };
        // 5 hits, all within review band.
        let hits = vec![
            hit(0.95, "Microsoft"),
            hit(0.94, "Microsoft Corp"),
            hit(0.93, "Microsoft Inc"),
            hit(0.92, "MS"),
            hit(0.91, "MSFT"),
        ];
        match classify(&cand, &hits, &cfg) {
            Decision::NeedsReview { runners_up, .. } => {
                assert_eq!(runners_up.len(), 2, "top + 2 runners-up = 3 max");
            }
            other => panic!("expected NeedsReview, got {other:?}"),
        }
    }

    #[test]
    fn single_hit_passes_margin_gate() {
        // Only one hit — margin is +∞ by definition.
        let cand = CandidateInfo {
            canonical_text: "Microsoft",
            props: &empty_props(),
        };
        let hits = vec![hit(0.97, "Microsoft")];
        match classify(&cand, &hits, &strict_cfg()) {
            Decision::AutoMerge { .. } => {}
            other => panic!("expected AutoMerge, got {other:?}"),
        }
    }

    #[test]
    fn is_type_only_recognises_canonical_pattern() {
        assert!(is_type_only("type: Person"));
        assert!(is_type_only("\ntype: Person\n"));
        assert!(!is_type_only("type: Person\nname: Alice"));
        assert!(!is_type_only("Alice")); // no type: prefix → not type-only
        assert!(!is_type_only(""));
    }
}
