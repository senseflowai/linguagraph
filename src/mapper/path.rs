//! Tiny JSONPath subset.
//!
//! The mapping language only ever uses three kinds of segments:
//!
//! * `$`            — root anchor
//! * `.<key>`       — descend into an object field
//! * `[*]`          — fan out over every element of an array
//!
//! Anything else (filters, recursive descent, slice expressions) is
//! intentionally rejected: the mapping language is not a query language.
//!
//! Evaluation is *position-aware* — every match carries the array indices
//! it traversed, in order. Two matches that came from `[*]` at the same
//! depth are siblings iff their context prefixes agree, which is what the
//! extractor uses to align properties with their parent rows and what the
//! planner uses to resolve implicit relationships.

use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PathError {
    #[error("path must start with '$'")]
    MissingRoot,
    #[error("unexpected character '{0}' at position {1}")]
    UnexpectedChar(char, usize),
    #[error("unterminated bracket expression")]
    UnterminatedBracket,
    #[error("unsupported bracket expression '{0}' (only [*] is allowed)")]
    UnsupportedBracket(String),
    #[error("empty field name")]
    EmptyField,
}

/// Single segment in a parsed path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// `.field` — descend into an object key.
    Field(String),
    /// `[*]` — fan out over every array element.
    Wildcard,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonPath {
    pub segments: Vec<Segment>,
}

impl JsonPath {
    pub fn parse(s: &str) -> Result<Self, PathError> {
        let bytes = s.as_bytes();
        if bytes.first() != Some(&b'$') {
            return Err(PathError::MissingRoot);
        }
        let mut i = 1;
        let mut segments = Vec::new();
        while i < bytes.len() {
            match bytes[i] {
                b'.' => {
                    i += 1;
                    let start = i;
                    while i < bytes.len() && bytes[i] != b'.' && bytes[i] != b'[' {
                        i += 1;
                    }
                    if start == i {
                        return Err(PathError::EmptyField);
                    }
                    segments.push(Segment::Field(s[start..i].to_string()));
                }
                b'[' => {
                    let close = bytes[i + 1..]
                        .iter()
                        .position(|&b| b == b']')
                        .ok_or(PathError::UnterminatedBracket)?;
                    let inner = &s[i + 1..i + 1 + close];
                    if inner == "*" {
                        segments.push(Segment::Wildcard);
                    } else {
                        return Err(PathError::UnsupportedBracket(inner.to_string()));
                    }
                    i += close + 2;
                }
                c => return Err(PathError::UnexpectedChar(c as char, i)),
            }
        }
        Ok(JsonPath { segments })
    }

    /// Convenience: number of `Wildcard` segments — equals the depth of the
    /// `context` vector returned by [`evaluate`].
    pub fn wildcard_arity(&self) -> usize {
        self.segments
            .iter()
            .filter(|s| matches!(s, Segment::Wildcard))
            .count()
    }

    /// Returns true iff `self.segments` starts with `prefix.segments` —
    /// i.e. `self` selects values *under* the rows selected by `prefix`.
    pub fn starts_with(&self, prefix: &JsonPath) -> bool {
        self.segments.len() >= prefix.segments.len()
            && self.segments[..prefix.segments.len()] == prefix.segments
    }

    /// The segments that come after `prefix`. Caller must have checked
    /// [`starts_with`] first.
    pub fn relative_to(&self, prefix: &JsonPath) -> Vec<Segment> {
        self.segments[prefix.segments.len()..].to_vec()
    }

    /// Evaluate against `root`, returning every match together with the
    /// array indices traversed at each `Wildcard`.
    pub fn evaluate<'a>(&self, root: &'a Value) -> Vec<Match<'a>> {
        let mut out = Vec::new();
        walk(&self.segments, root, Vec::new(), &mut out);
        out
    }
}

/// One match produced by [`JsonPath::evaluate`].
#[derive(Debug, Clone)]
pub struct Match<'a> {
    pub value: &'a Value,
    /// Indices captured at each `Wildcard` segment, in order.
    pub context: Vec<usize>,
}

/// Walk a path against a *sub*-tree (no root anchor), useful when we have
/// already resolved the parent and only want to evaluate a relative
/// expression. Mirrors [`JsonPath::evaluate`] but takes raw segments.
pub fn walk_segments<'a>(
    segments: &[Segment],
    value: &'a Value,
    base_context: Vec<usize>,
) -> Vec<Match<'a>> {
    let mut out = Vec::new();
    walk(segments, value, base_context, &mut out);
    out
}

fn walk<'a>(segments: &[Segment], value: &'a Value, context: Vec<usize>, out: &mut Vec<Match<'a>>) {
    let Some((seg, rest)) = segments.split_first() else {
        out.push(Match { value, context });
        return;
    };
    match seg {
        Segment::Field(k) => {
            if let Value::Object(map) = value {
                if let Some(v) = map.get(k) {
                    walk(rest, v, context, out);
                }
            }
        }
        Segment::Wildcard => {
            if let Value::Array(items) = value {
                for (idx, v) in items.iter().enumerate() {
                    let mut ctx = context.clone();
                    ctx.push(idx);
                    walk(rest, v, ctx, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_simple_path() {
        let p = JsonPath::parse("$.cameras[*].id").unwrap();
        assert_eq!(p.segments.len(), 3);
        assert_eq!(p.wildcard_arity(), 1);
    }

    #[test]
    fn parses_nested_wildcards() {
        let p = JsonPath::parse("$.cameras[*].video_analytics.modules[*].module").unwrap();
        assert_eq!(p.wildcard_arity(), 2);
    }

    #[test]
    fn rejects_missing_root() {
        assert_eq!(JsonPath::parse("cameras[*]"), Err(PathError::MissingRoot));
    }

    #[test]
    fn rejects_unsupported_bracket() {
        assert!(matches!(
            JsonPath::parse("$.a[0]"),
            Err(PathError::UnsupportedBracket(_))
        ));
        assert!(matches!(
            JsonPath::parse("$.a[?(@.x>1)]"),
            Err(PathError::UnsupportedBracket(_))
        ));
    }

    #[test]
    fn rejects_unterminated_bracket() {
        assert_eq!(
            JsonPath::parse("$.a[*"),
            Err(PathError::UnterminatedBracket)
        );
    }

    #[test]
    fn rejects_empty_field() {
        assert_eq!(JsonPath::parse("$..a"), Err(PathError::EmptyField));
    }

    #[test]
    fn evaluates_root_only() {
        let p = JsonPath::parse("$").unwrap();
        let data = json!({"x": 1});
        let m = p.evaluate(&data);
        assert_eq!(m.len(), 1);
        assert!(m[0].context.is_empty());
    }

    #[test]
    fn fans_out_over_array() {
        let data = json!({"items": [{"id": 1}, {"id": 2}, {"id": 3}]});
        let p = JsonPath::parse("$.items[*].id").unwrap();
        let m = p.evaluate(&data);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].context, vec![0]);
        assert_eq!(m[2].context, vec![2]);
        assert_eq!(m[2].value, &json!(3));
    }

    #[test]
    fn nested_wildcards_record_both_indices() {
        let data = json!({
            "cameras": [
                {"modules": [{"name": "a"}, {"name": "b"}]},
                {"modules": [{"name": "c"}]}
            ]
        });
        let p = JsonPath::parse("$.cameras[*].modules[*].name").unwrap();
        let m = p.evaluate(&data);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].context, vec![0, 0]);
        assert_eq!(m[1].context, vec![0, 1]);
        assert_eq!(m[2].context, vec![1, 0]);
    }

    #[test]
    fn missing_field_yields_no_match() {
        let p = JsonPath::parse("$.a.b").unwrap();
        let data = json!({"a": {}});
        let m = p.evaluate(&data);
        assert!(m.is_empty());
    }

    #[test]
    fn starts_with_and_relative_to() {
        let parent = JsonPath::parse("$.cameras[*]").unwrap();
        let child = JsonPath::parse("$.cameras[*].origin.place_id").unwrap();
        let unrelated = JsonPath::parse("$.places[*]").unwrap();
        assert!(child.starts_with(&parent));
        assert!(!unrelated.starts_with(&parent));
        let rel = child.relative_to(&parent);
        assert_eq!(rel.len(), 2);
    }
}
