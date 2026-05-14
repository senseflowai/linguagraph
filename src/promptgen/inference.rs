//! Heuristic type inference for JSON leaf values.
//!
//! Each rule is a small pure function that takes the field name and
//! statistics about the values seen at that field, and returns an
//! [`InferredType`] guess. The analyzer composes these — it does **not**
//! re-implement them — so we can exercise each rule in isolation.

use std::fmt;

use serde_json::Value;

/// A type guess emitted by [`infer`].
///
/// The string identifiers match what the prompt asks the LLM to use
/// in the resulting mapping (`"type": "SemanticText"` etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum InferredType {
    /// Used for the entity's primary key — surfaced as a hint, not as
    /// a `type` tag in the output mapping.
    Identifier,
    /// Long natural-language string. Maps to `SemanticText` in the
    /// rendered prompt.
    SemanticText,
    /// Short categorical / enum-like string. Maps to `Keyword`.
    Keyword,
    /// ISO-8601-shaped string. Maps to `DateTime`.
    DateTime,
    /// Free-form short string with high cardinality but no obvious
    /// "categorical" pattern — kept distinct from Keyword so the
    /// prompt can suggest the LLM verify.
    Text,
    Number,
    Boolean,
    /// Catch-all when the analyzer couldn't decide.
    Unknown,
}

impl InferredType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Identifier => "Identifier",
            Self::SemanticText => "SemanticText",
            Self::Keyword => "Keyword",
            Self::DateTime => "DateTime",
            Self::Text => "Text",
            Self::Number => "Number",
            Self::Boolean => "Boolean",
            Self::Unknown => "Unknown",
        }
    }
}

impl fmt::Display for InferredType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Aggregate statistics used by the heuristics.
///
/// All counts include nulls; `non_null` excludes them. Length stats
/// only consider string samples.
#[derive(Debug, Clone, Default)]
pub struct LeafStats {
    pub count: usize,
    pub non_null: usize,
    pub distinct: usize,
    pub mean_str_len: f64,
    pub max_str_len: usize,
    /// JSON kind seen at this leaf. `Mixed` means the same path
    /// produced different kinds across samples (rare; surfaced as
    /// `Unknown` so the prompt asks the LLM to look closer).
    pub kind: LeafKind,
    /// Whether **every** non-null sample looked like ISO-8601.
    pub all_iso8601: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeafKind {
    String,
    Integer,
    Float,
    Bool,
    Null,
    Mixed,
    #[default]
    Empty,
}

impl LeafStats {
    /// Update stats with a single sample.
    pub fn observe(&mut self, v: &Value) {
        self.count += 1;
        let kind = leaf_kind_of(v);
        // Walk the kind state machine: first observation sets it,
        // subsequent ones either match or collapse to Mixed.
        match (self.kind, kind) {
            (LeafKind::Empty, k) => self.kind = k,
            (a, b) if a == b => {}
            (LeafKind::Null, b) | (b, LeafKind::Null) => self.kind = b,
            _ => self.kind = LeafKind::Mixed,
        }
        if !v.is_null() {
            self.non_null += 1;
            if let Some(s) = v.as_str() {
                let len = s.chars().count();
                self.max_str_len = self.max_str_len.max(len);
                let n = self.non_null as f64;
                self.mean_str_len += (len as f64 - self.mean_str_len) / n;
                if self.non_null == 1 {
                    self.all_iso8601 = looks_like_iso8601(s);
                } else {
                    self.all_iso8601 = self.all_iso8601 && looks_like_iso8601(s);
                }
            }
        }
    }
}

fn leaf_kind_of(v: &Value) -> LeafKind {
    match v {
        Value::Null => LeafKind::Null,
        Value::Bool(_) => LeafKind::Bool,
        Value::Number(n) if n.is_i64() || n.is_u64() => LeafKind::Integer,
        Value::Number(_) => LeafKind::Float,
        Value::String(_) => LeafKind::String,
        // Arrays/objects shouldn't reach the leaf-stats stage; fall
        // through to Mixed so they're flagged for review.
        _ => LeafKind::Mixed,
    }
}

/// Names that strongly suggest a free-text/prose field.
const PROSE_NAMES: &[&str] = &[
    "description",
    "desc",
    "bio",
    "biography",
    "summary",
    "abstract",
    "body",
    "content",
    "comment",
    "comments",
    "text",
    "notes",
    "note",
    "details",
    "detail",
    "message",
    "title",
    "headline",
    "review",
    "synopsis",
];

/// Names that strongly suggest a categorical / enum field.
const KEYWORD_NAMES: &[&str] = &[
    "status",
    "state",
    "category",
    "categories",
    "industry",
    "kind",
    "type",
    "role",
    "tier",
    "level",
    "priority",
    "severity",
    "phase",
    "stage",
    "department",
    "channel",
];

/// Names that mark a primary-key candidate.
const ID_NAMES: &[&str] = &["id", "_id", "uuid", "guid", "key", "pk"];

/// Length above which a string is presumed to be prose.
const SEMANTIC_TEXT_LEN_THRESHOLD: f64 = 30.0;

/// Distinct-value cap for a Keyword (when explicit name hints don't fire).
const KEYWORD_DISTINCT_CAP: usize = 10;

/// Cardinality ratio cap for Keyword.
const KEYWORD_RATIO_CAP: f64 = 0.20;

/// Is `name` a likely primary-key field?
pub fn is_id_field(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if ID_NAMES.iter().any(|n| *n == lower) {
        return true;
    }
    // Suffix forms: <thing>_id, <thing>Id.
    lower.ends_with("_id") || lower.ends_with("id") && name.chars().any(|c| c.is_uppercase())
}

/// Bag of properties matched on field-name lowercased.
fn name_matches_any(name: &str, names: &[&str]) -> bool {
    let lower = name.to_ascii_lowercase();
    names.iter().any(|n| *n == lower)
}

/// Conservative ISO-8601 detector. Accepts dates and date-times in
/// the YYYY-MM-DD[Thh:mm[:ss[.fff]][Z|+hh:mm]] family. We don't try
/// to parse the calendar — the goal is "this *looks* like a
/// timestamp" so the prompt can suggest a DateTime type.
fn looks_like_iso8601(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() < 10 {
        return false;
    }
    let is_digit = |i: usize| bytes.get(i).is_some_and(|b| b.is_ascii_digit());
    let dash = |i: usize| bytes.get(i) == Some(&b'-');
    if !(is_digit(0)
        && is_digit(1)
        && is_digit(2)
        && is_digit(3)
        && dash(4)
        && is_digit(5)
        && is_digit(6)
        && dash(7)
        && is_digit(8)
        && is_digit(9))
    {
        return false;
    }
    if bytes.len() == 10 {
        return true;
    }
    matches!(bytes.get(10), Some(b'T') | Some(b' '))
}

/// Run the rules against `(name, stats)` and return the best guess.
pub fn infer(name: &str, stats: &LeafStats) -> InferredType {
    if stats.non_null == 0 {
        return InferredType::Unknown;
    }

    // The primary-key field is identified by name; the analyser
    // surfaces it as `Identifier` so the prompt can put it in
    // `primary_key` rather than `properties`.
    if is_id_field(name) {
        return InferredType::Identifier;
    }

    match stats.kind {
        LeafKind::Bool => return InferredType::Boolean,
        LeafKind::Integer | LeafKind::Float => return InferredType::Number,
        LeafKind::String => {} // fall through
        LeafKind::Mixed | LeafKind::Empty | LeafKind::Null => return InferredType::Unknown,
    }

    // String paths.
    if stats.all_iso8601 {
        return InferredType::DateTime;
    }
    if name_matches_any(name, PROSE_NAMES) {
        return InferredType::SemanticText;
    }
    if name_matches_any(name, KEYWORD_NAMES) {
        return InferredType::Keyword;
    }
    if stats.mean_str_len >= SEMANTIC_TEXT_LEN_THRESHOLD {
        return InferredType::SemanticText;
    }
    // Categorical heuristic: a small distinct set that repeats. We
    // don't gate on ratio strictly (a very small dataset can have a
    // 1:1 ratio yet still be enum-like — `industry` with two rows is
    // still categorical) but we do require evidence of repetition
    // when there are enough samples to expect it.
    let ratio = stats.distinct as f64 / stats.non_null.max(1) as f64;
    let distinct_small = stats.distinct <= KEYWORD_DISTINCT_CAP;
    let repeats = stats.distinct < stats.non_null;
    if distinct_small && stats.non_null >= 5 && (repeats || ratio <= KEYWORD_RATIO_CAP) {
        return InferredType::Keyword;
    }
    InferredType::Text
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn stats_with(samples: &[Value]) -> LeafStats {
        let mut s = LeafStats::default();
        let mut seen = std::collections::HashSet::new();
        for v in samples {
            s.observe(v);
            seen.insert(v.to_string());
        }
        s.distinct = seen.len();
        s
    }

    #[test]
    fn id_field_resolves_to_identifier_regardless_of_kind() {
        let s = stats_with(&[json!(1), json!(2), json!(3)]);
        assert_eq!(infer("id", &s), InferredType::Identifier);
        let s = stats_with(&[json!("a"), json!("b")]);
        assert_eq!(infer("uuid", &s), InferredType::Identifier);
    }

    #[test]
    fn long_strings_are_semantic_text() {
        let s = stats_with(&[
            json!("Lorem ipsum dolor sit amet consectetur adipiscing elit."),
            json!("Quite a verbose description of something."),
        ]);
        assert_eq!(infer("blurb", &s), InferredType::SemanticText);
    }

    #[test]
    fn name_hint_overrides_length() {
        let s = stats_with(&[json!("ok"), json!("ok"), json!("err")]);
        assert_eq!(infer("description", &s), InferredType::SemanticText);
    }

    #[test]
    fn enum_like_short_string_is_keyword() {
        let s = stats_with(&[
            json!("active"),
            json!("active"),
            json!("inactive"),
            json!("active"),
            json!("error"),
            json!("inactive"),
        ]);
        assert_eq!(infer("status_v", &s), InferredType::Keyword);
    }

    #[test]
    fn keyword_name_hint_fires_on_low_sample_count() {
        let s = stats_with(&[json!("Tech"), json!("Health")]);
        assert_eq!(infer("industry", &s), InferredType::Keyword);
    }

    #[test]
    fn iso8601_string_is_datetime() {
        let s = stats_with(&[json!("2024-01-02T15:04:05Z"), json!("2025-06-07T01:02:03Z")]);
        assert_eq!(infer("created_at", &s), InferredType::DateTime);
    }

    #[test]
    fn integer_is_number() {
        let s = stats_with(&[json!(1), json!(2), json!(3)]);
        assert_eq!(infer("count", &s), InferredType::Number);
    }

    #[test]
    fn boolean_is_boolean() {
        let s = stats_with(&[json!(true), json!(false)]);
        assert_eq!(infer("active", &s), InferredType::Boolean);
    }

    #[test]
    fn empty_or_null_only_is_unknown() {
        let s = stats_with(&[]);
        assert_eq!(infer("anything", &s), InferredType::Unknown);
        let s = stats_with(&[json!(null), json!(null)]);
        assert_eq!(infer("anything", &s), InferredType::Unknown);
    }

    #[test]
    fn id_field_detection_handles_variants() {
        assert!(is_id_field("id"));
        assert!(is_id_field("ID"));
        assert!(is_id_field("_id"));
        assert!(is_id_field("uuid"));
        assert!(is_id_field("user_id"));
        assert!(is_id_field("userId"));
        assert!(!is_id_field("identity"));
        assert!(!is_id_field("invalid"));
    }
}
