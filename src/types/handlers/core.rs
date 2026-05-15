//! Built-in scalar type handlers.
//!
//! Each handler validates a raw JSON value and rewrites it into a
//! storage-ready [`Literal`] during ingestion. They share no state with
//! each other so a registry can compose any subset without coupling.
//!
//! The handlers are deliberately local: they don't queue side effects and
//! they don't need an embedder. `Text` participates in query lowering so it
//! can normalize comparison values the same way it normalizes stored values;
//! the remaining scalar handlers are ingestion-only.
//!
//! Adding a new scalar type follows a recipe:
//!
//! 1. Implement [`ScalarParser::parse`] for raw JSON → [`Literal`].
//! 2. Wrap the parser in [`ScalarTypeHandler`] with the public type id
//!    (`"MyType"`).
//! 3. Register the handler via [`super::register_default`] (or an
//!    explicit `RegistryBuilder::register(...)` for tests).
//!
//! The handler's [`TypeHandler::on_ingest`] delegates to its parser. `Text`
//! implements query stages directly; other scalar query stages keep the
//! generic loud-failure implementation.

use crate::ast::query::Literal;
use crate::types::context::{EmitCtx, IngestCtx, LowerCtx};
use crate::types::{Capabilities, TypeError, TypeHandler, TypeId, TypedOp, TypedPredicate};
use serde_json::Value;
use tracing::warn;

/// Pure parser from raw JSON to a storable [`Literal`].
///
/// Implementations are stateless and `Send + Sync` so the same parser
/// can be cloned cheaply into a [`ScalarTypeHandler`] and shared across
/// async boundaries.
///
/// `parse` returns:
///
/// * `Ok(Some(lit))` — store `lit` on the node.
/// * `Ok(None)` — the value is null; the property is dropped (matches
///   the existing "missing values are tolerated" contract).
/// * `Err(...)` — the value is malformed for this type. The planner
///   surfaces this as an `IngestError::Type` so authors get a precise
///   error per row.
pub trait ScalarParser: Send + Sync + std::fmt::Debug {
    fn parse(&self, raw: &Value) -> Result<Option<Literal>, TypeError>;
}

/// Generic [`TypeHandler`] that delegates ingestion to a [`ScalarParser`].
///
/// All five built-in scalar types are instances of this struct — adding
/// a new one is a one-liner: write a parser, wrap it here, register it.
pub struct ScalarTypeHandler {
    type_id: TypeId,
    parser: Box<dyn ScalarParser>,
}

impl std::fmt::Debug for ScalarTypeHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarTypeHandler")
            .field("type_id", &self.type_id)
            .field("parser", &self.parser)
            .finish()
    }
}

impl ScalarTypeHandler {
    pub fn new(type_id: impl Into<TypeId>, parser: Box<dyn ScalarParser>) -> Self {
        Self {
            type_id: type_id.into(),
            parser,
        }
    }

    /// Parse a raw JSON value through this handler's parser.
    /// Public so callers (the extractor, the planner) can apply the
    /// type to the property without going through the full
    /// [`TypeHandler::on_ingest`] flow.
    pub fn parse(&self, raw: &Value) -> Result<Option<Literal>, TypeError> {
        self.parser.parse(raw)
    }
}

impl TypeHandler for ScalarTypeHandler {
    fn type_id(&self) -> TypeId {
        self.type_id.clone()
    }

    fn capabilities(&self) -> Capabilities {
        // Scalar types support exact-match comparisons; the builder's
        // generic plain-op path renders them as standard Cypher.
        Capabilities::INGEST | Capabilities::EXACT_MATCH
    }

    fn supported_ops(&self) -> Vec<TypedOp> {
        vec![TypedOp::Eq, TypedOp::Neq]
    }

    fn on_ingest(&self, ctx: &mut IngestCtx<'_>) -> Result<(), TypeError> {
        match self.parser.parse(ctx.value())? {
            Some(lit) => ctx.set_value(lit),
            None => ctx.skip(),
        }
        Ok(())
    }

    fn lower(&self, _ctx: &mut LowerCtx<'_>) -> Result<TypedPredicate, TypeError> {
        // Scalar types don't contribute to query lowering — the DSL
        // path bypasses this and renders plain Cypher comparisons. The
        // builder's `from_dsl` only calls `lower` when an op needs
        // type-aware compilation (Search, HybridSearch, …); this branch
        // would only ever be reached if a future op were routed here
        // by mistake, in which case loud failure is correct.
        Err(TypeError::Handler(format!(
            "type '{}' has no query-time lowering — use a plain DSL filter",
            self.type_id
        )))
    }

    fn emit(&self, _ctx: &mut EmitCtx<'_>, _pred: &TypedPredicate) -> Result<(), TypeError> {
        Err(TypeError::Handler(format!(
            "type '{}' has no Cypher emitter",
            self.type_id
        )))
    }
}

// ─── Built-in parsers ───────────────────────────────────────────────────

/// `Text` — store the value as a string.
///
/// Accepts strings as-is and stringifies bools and finite numbers so
/// authors can lift mixed-typed JSON fields without a custom converter.
#[derive(Debug, Default)]
pub struct TextParser;

impl TextParser {
    /// Normalize text for storage and query comparisons.
    ///
    /// Keeps letters and numbers, removes whitespace/punctuation/symbols,
    /// and lowercases with Unicode-aware case conversion.
    pub fn normalize(value: &str) -> String {
        value
            .chars()
            .filter(|ch| ch.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    }
}

impl ScalarParser for TextParser {
    fn parse(&self, raw: &Value) -> Result<Option<Literal>, TypeError> {
        match raw {
            Value::Null => Ok(None),
            Value::String(s) => Ok(Some(Literal::String(Self::normalize(s)))),
            Value::Bool(b) => Ok(Some(Literal::String(Self::normalize(&b.to_string())))),
            Value::Number(n) => Ok(Some(Literal::String(Self::normalize(&n.to_string())))),
            Value::Array(items) => {
                let lits = items
                    .iter()
                    .map(|v| self.parse(v))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                Ok(Some(Literal::List(lits)))
            }
            Value::Object(_) => Err(TypeError::InvalidValue {
                ty: "Text".into(),
                reason: "objects cannot be stored as Text".into(),
            }),
        }
    }
}

/// `Number` — store the value as an integer or float.
///
/// Strings are accepted to support common authoring conventions:
///
/// * trailing `%` is interpreted as a percentage (`"50%"` → `0.5`).
/// * thousands separators (`,` or `_`) are stripped.
/// * leading/trailing whitespace is ignored.
///
/// Booleans are rejected — coercing them silently masks bugs.
#[derive(Debug, Default)]
pub struct NumberParser;

impl ScalarParser for NumberParser {
    fn parse(&self, raw: &Value) -> Result<Option<Literal>, TypeError> {
        match raw {
            Value::Null => Ok(None),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(Some(Literal::Int(i)))
                } else if let Some(f) = n.as_f64() {
                    if !f.is_finite() {
                        return Err(TypeError::InvalidValue {
                            ty: "Number".into(),
                            reason: format!("non-finite numeric value: {n}"),
                        });
                    }
                    Ok(Some(Literal::Float(f)))
                } else {
                    Err(TypeError::InvalidValue {
                        ty: "Number".into(),
                        reason: format!("unsupported numeric encoding: {n}"),
                    })
                }
            }
            Value::String(s) => Ok(Some(parse_number_string(s)?)),
            Value::Array(items) => {
                let lits = items
                    .iter()
                    .map(|v| self.parse(v))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                Ok(Some(Literal::List(lits)))
            }
            other => Err(TypeError::InvalidValue {
                ty: "Number".into(),
                reason: format!("expected number, got {}", json_kind(other)),
            }),
        }
    }
}

fn parse_number_string(s: &str) -> Result<Literal, TypeError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        // return Err(TypeError::InvalidValue {
        //     ty: "Number".into(),
        //     reason: "empty string is not a number".into(),
        // });
        warn!("empty string is not a number");
        return Ok(Literal::Int(0));
    }

    // Percentage: divide by 100, always a float.
    if let Some(body) = trimmed.strip_suffix('%') {
        let cleaned = clean_number(body);
        let f: f64 = cleaned.parse().map_err(|_| TypeError::InvalidValue {
            ty: "Number".into(),
            reason: format!("invalid percentage: {s:?}"),
        })?;
        if !f.is_finite() {
            return Err(TypeError::InvalidValue {
                ty: "Number".into(),
                reason: format!("non-finite percentage: {s:?}"),
            });
        }
        return Ok(Literal::Float(f / 100.0));
    }

    let cleaned = clean_number(trimmed);
    // Try integer first so whole numbers stay typed as Int.
    if let Ok(i) = cleaned.parse::<i64>() {
        return Ok(Literal::Int(i));
    }
    if let Ok(f) = cleaned.parse::<f64>() {
        if !f.is_finite() {
            return Err(TypeError::InvalidValue {
                ty: "Number".into(),
                reason: format!("non-finite number: {s:?}"),
            });
        }
        return Ok(Literal::Float(f));
    }
    Err(TypeError::InvalidValue {
        ty: "Number".into(),
        reason: format!("invalid number: {s:?}"),
    })
}

fn clean_number(s: &str) -> String {
    s.chars().filter(|c| *c != ',' && *c != '_').collect()
}

/// `Boolean` — true/false.
///
/// Accepts the JSON literals as well as common string spellings
/// (`"true"`, `"yes"`, `"1"`, `"on"` and the inverses). Numbers are
/// rejected to keep DSL semantics unambiguous — author the value as a
/// real JSON boolean if possible.
#[derive(Debug, Default)]
pub struct BooleanParser;

impl ScalarParser for BooleanParser {
    fn parse(&self, raw: &Value) -> Result<Option<Literal>, TypeError> {
        match raw {
            Value::Null => Ok(None),
            Value::Bool(b) => Ok(Some(Literal::Bool(*b))),
            Value::String(s) => parse_bool_string(s).map(|b| Some(Literal::Bool(b))),
            Value::Array(items) => {
                let lits = items
                    .iter()
                    .map(|v| self.parse(v))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                Ok(Some(Literal::List(lits)))
            }
            other => Err(TypeError::InvalidValue {
                ty: "Boolean".into(),
                reason: format!("expected boolean, got {}", json_kind(other)),
            }),
        }
    }
}

fn parse_bool_string(s: &str) -> Result<bool, TypeError> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "yes" | "y" | "1" | "on" => Ok(true),
        "false" | "no" | "n" | "0" | "off" => Ok(false),
        _ => Err(TypeError::InvalidValue {
            ty: "Boolean".into(),
            reason: format!("not a boolean: {s:?}"),
        }),
    }
}

/// `Date` — calendar date stored as an ISO-8601 string.
///
/// Accepts:
///
/// * date-only strings (`"2024-05-06"`).
/// * full ISO-8601 timestamps; the time component is dropped.
/// * Unix epoch seconds as an integer (rendered as the `YYYY-MM-DD`
///   the integer falls in, UTC).
#[derive(Debug, Default)]
pub struct DateParser;

impl ScalarParser for DateParser {
    fn parse(&self, raw: &Value) -> Result<Option<Literal>, TypeError> {
        match raw {
            Value::Null => Ok(None),
            Value::String(s) => parse_date_string(s).map(|d| Some(Literal::String(d))),
            Value::Number(n) => match n.as_i64() {
                Some(secs) => Ok(Some(Literal::String(epoch_to_date(secs)))),
                None => Err(TypeError::InvalidValue {
                    ty: "Date".into(),
                    reason: format!("expected integer epoch seconds, got {n}"),
                }),
            },
            Value::Array(items) => {
                let lits = items
                    .iter()
                    .map(|v| self.parse(v))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                Ok(Some(Literal::List(lits)))
            }
            other => Err(TypeError::InvalidValue {
                ty: "Date".into(),
                reason: format!("expected date string, got {}", json_kind(other)),
            }),
        }
    }
}

/// `Timestamp` — instant stored as an ISO-8601 timestamp string in UTC.
///
/// Accepts:
///
/// * full ISO-8601 timestamps (passed through, normalised to `Z`).
/// * date-only strings (the time component is set to `T00:00:00Z`).
/// * Unix epoch seconds as an integer.
#[derive(Debug, Default)]
pub struct TimestampParser;

impl ScalarParser for TimestampParser {
    fn parse(&self, raw: &Value) -> Result<Option<Literal>, TypeError> {
        match raw {
            Value::Null => Ok(None),
            Value::String(s) => parse_timestamp_string(s).map(|t| Some(Literal::String(t))),
            Value::Number(n) => match n.as_i64() {
                Some(secs) => Ok(Some(Literal::String(epoch_to_timestamp(secs)))),
                None => Err(TypeError::InvalidValue {
                    ty: "Timestamp".into(),
                    reason: format!("expected integer epoch seconds, got {n}"),
                }),
            },
            Value::Array(items) => {
                let lits = items
                    .iter()
                    .map(|v| self.parse(v))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                Ok(Some(Literal::List(lits)))
            }
            other => Err(TypeError::InvalidValue {
                ty: "Timestamp".into(),
                reason: format!("expected timestamp string, got {}", json_kind(other)),
            }),
        }
    }
}

// ─── Date/Timestamp string parsers ─────────────────────────────────────
//
// We deliberately keep this dependency-free. The supported shapes are:
//
//   YYYY-MM-DD
//   YYYY-MM-DDTHH:MM:SS
//   YYYY-MM-DDTHH:MM:SS.fff
//   any of the above with a trailing 'Z' or '+HH:MM' / '-HH:MM' offset.
//
// Anything else fails fast with the offending input embedded in the
// error so authors know exactly what to fix.

fn parse_date_string(s: &str) -> Result<String, TypeError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(TypeError::InvalidValue {
            ty: "Date".into(),
            reason: "empty string is not a date".into(),
        });
    }
    let (date, _rest) = split_date_component(trimmed)?;
    validate_date_components(&date)?;
    Ok(date)
}

fn parse_timestamp_string(s: &str) -> Result<String, TypeError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(TypeError::InvalidValue {
            ty: "Timestamp".into(),
            reason: "empty string is not a timestamp".into(),
        });
    }
    let (date, rest) = split_date_component(trimmed).map_err(|e| relabel(e, "Timestamp"))?;
    validate_date_components(&date).map_err(|e| relabel(e, "Timestamp"))?;
    let time_part = match rest {
        Some(r) => normalise_time_component(r)?,
        None => "T00:00:00Z".to_string(),
    };
    Ok(format!("{date}{time_part}"))
}

/// Split a string into the leading `YYYY-MM-DD` component and whatever
/// follows (possibly `T...`). Returns the date as the canonical
/// `YYYY-MM-DD` plus the remainder.
fn split_date_component(s: &str) -> Result<(String, Option<&str>), TypeError> {
    if s.len() < 10 {
        return Err(TypeError::InvalidValue {
            ty: "Date".into(),
            reason: format!("not an ISO-8601 date: {s:?}"),
        });
    }
    let (head, tail) = s.split_at(10);
    let bytes = head.as_bytes();
    let valid_shape = bytes.len() == 10
        && bytes[..4].iter().all(u8::is_ascii_digit)
        && bytes[4] == b'-'
        && bytes[5..7].iter().all(u8::is_ascii_digit)
        && bytes[7] == b'-'
        && bytes[8..10].iter().all(u8::is_ascii_digit);
    if !valid_shape {
        return Err(TypeError::InvalidValue {
            ty: "Date".into(),
            reason: format!("not an ISO-8601 date: {s:?}"),
        });
    }
    let rest = if tail.is_empty() { None } else { Some(tail) };
    Ok((head.to_string(), rest))
}

fn validate_date_components(date: &str) -> Result<(), TypeError> {
    let year: u32 = date[0..4].parse().expect("digits checked");
    let month: u32 = date[5..7].parse().expect("digits checked");
    let day: u32 = date[8..10].parse().expect("digits checked");
    if !(1..=12).contains(&month) {
        return Err(TypeError::InvalidValue {
            ty: "Date".into(),
            reason: format!("month out of range in {date:?}"),
        });
    }
    let max_day = days_in_month(year, month);
    if !(1..=max_day).contains(&day) {
        return Err(TypeError::InvalidValue {
            ty: "Date".into(),
            reason: format!("day out of range in {date:?}"),
        });
    }
    Ok(())
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Validate / normalise the `Thh:mm:ss[.fff][Z|±hh:mm]` tail of a
/// timestamp. Returns the normalised tail (always terminated by `Z` or
/// an explicit offset).
fn normalise_time_component(rest: &str) -> Result<String, TypeError> {
    let body = rest
        .strip_prefix('T')
        .or_else(|| rest.strip_prefix(' '))
        .ok_or_else(|| TypeError::InvalidValue {
            ty: "Timestamp".into(),
            reason: format!("expected 'T' separator, got {rest:?}"),
        })?;

    // Split off zone suffix.
    let (time, zone) = if let Some(idx) = body.rfind(['Z', '+', '-']) {
        // Hours like "10:00" contain '-'? No — the time uses ':'.
        // Anything matching '+' or '-' in the tail is an offset.
        let candidate = &body[idx..];
        if candidate.starts_with('Z') || is_valid_offset(candidate) {
            (&body[..idx], candidate.to_string())
        } else {
            (body, "Z".to_string())
        }
    } else {
        (body, "Z".to_string())
    };

    validate_time_body(time)?;
    Ok(format!("T{time}{zone}"))
}

fn is_valid_offset(s: &str) -> bool {
    // ±HHMM or ±HH:MM
    let bytes = s.as_bytes();
    if bytes.len() == 5 {
        return (bytes[0] == b'+' || bytes[0] == b'-') && bytes[1..].iter().all(u8::is_ascii_digit);
    }
    if bytes.len() == 6 {
        return (bytes[0] == b'+' || bytes[0] == b'-')
            && bytes[1..3].iter().all(u8::is_ascii_digit)
            && bytes[3] == b':'
            && bytes[4..].iter().all(u8::is_ascii_digit);
    }
    false
}

fn validate_time_body(time: &str) -> Result<(), TypeError> {
    // `HH:MM:SS` or `HH:MM:SS.fff`.
    let (hms, _frac) = match time.split_once('.') {
        Some((h, f)) if !f.is_empty() && f.bytes().all(|b| b.is_ascii_digit()) => (h, Some(f)),
        Some(_) => {
            return Err(TypeError::InvalidValue {
                ty: "Timestamp".into(),
                reason: format!("invalid fractional seconds in {time:?}"),
            })
        }
        None => (time, None),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3
        || parts
            .iter()
            .any(|p| p.len() != 2 || !p.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err(TypeError::InvalidValue {
            ty: "Timestamp".into(),
            reason: format!("invalid HH:MM:SS in {time:?}"),
        });
    }
    let h: u32 = parts[0].parse().expect("digits checked");
    let m: u32 = parts[1].parse().expect("digits checked");
    let sec: u32 = parts[2].parse().expect("digits checked");
    if h > 23 || m > 59 || sec > 60 {
        return Err(TypeError::InvalidValue {
            ty: "Timestamp".into(),
            reason: format!("HH:MM:SS out of range in {time:?}"),
        });
    }
    Ok(())
}

fn relabel(err: TypeError, ty: &str) -> TypeError {
    match err {
        TypeError::InvalidValue { reason, .. } => TypeError::InvalidValue {
            ty: ty.into(),
            reason,
        },
        other => other,
    }
}

/// Render `secs` (seconds since the Unix epoch, UTC) as `YYYY-MM-DD`.
fn epoch_to_date(secs: i64) -> String {
    let (y, m, d, _h, _mi, _s) = epoch_to_ymdhms(secs);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Render `secs` as a full ISO-8601 timestamp in UTC.
fn epoch_to_timestamp(secs: i64) -> String {
    let (y, m, d, h, mi, s) = epoch_to_ymdhms(secs);
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// Convert seconds since the Unix epoch into civil `(year, month, day,
/// hour, minute, second)` components in UTC.
///
/// Implements Howard Hinnant's `civil_from_days` algorithm — fast,
/// correct across the full proleptic Gregorian range, and dependency-
/// free. We need this here because we deliberately don't pull in
/// `chrono` for what is otherwise a pure-data conversion.
fn epoch_to_ymdhms(secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    // Seconds in a day.
    const SPD: i64 = 86_400;
    let days = secs.div_euclid(SPD);
    let day_secs = secs.rem_euclid(SPD);
    let h = (day_secs / 3600) as u32;
    let mi = ((day_secs % 3600) / 60) as u32;
    let s = (day_secs % 60) as u32;

    // Hinnant's `civil_from_days` (z is days since 1970-01-01).
    let z = days + 719468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = y + if m <= 2 { 1 } else { 0 };
    (year, m, d, h, mi, s)
}

pub(super) fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ─── Convenience constructors ───────────────────────────────────────────

/// Build a [`ScalarTypeHandler`] for the built-in `Number` type.
pub fn number_handler() -> ScalarTypeHandler {
    ScalarTypeHandler::new("Number", Box::new(NumberParser))
}

/// Build a [`ScalarTypeHandler`] for the built-in `Boolean` type.
pub fn boolean_handler() -> ScalarTypeHandler {
    ScalarTypeHandler::new("Boolean", Box::new(BooleanParser))
}

/// Build a [`ScalarTypeHandler`] for the built-in `Date` type.
pub fn date_handler() -> ScalarTypeHandler {
    ScalarTypeHandler::new("Date", Box::new(DateParser))
}

/// Build a [`ScalarTypeHandler`] for the built-in `Timestamp` type.
pub fn timestamp_handler() -> ScalarTypeHandler {
    ScalarTypeHandler::new("Timestamp", Box::new(TimestampParser))
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    #[test]
    fn text_parses_strings_numbers_bools_and_arrays() {
        let p = TextParser;
        assert_eq!(
            p.parse(&json!("Hello, World!")).unwrap(),
            Some(Literal::String("helloworld".into()))
        );
        assert_eq!(
            p.parse(&json!(42)).unwrap(),
            Some(Literal::String("42".into()))
        );
        assert_eq!(
            p.parse(&json!(true)).unwrap(),
            Some(Literal::String("true".into()))
        );
        assert_eq!(p.parse(&json!(null)).unwrap(), None);
        let arr = p.parse(&json!(["a", 1])).unwrap().unwrap();
        match arr {
            Literal::List(items) => {
                assert_eq!(
                    items,
                    vec![Literal::String("a".into()), Literal::String("1".into())]
                );
            }
            _ => panic!("expected list"),
        }
    }

    #[test]
    fn text_rejects_objects() {
        let err = TextParser.parse(&json!({"k": 1})).unwrap_err();
        assert!(matches!(err, TypeError::InvalidValue { .. }));
    }

    #[test]
    fn number_parses_int_float_and_string_forms() {
        let p = NumberParser;
        assert_eq!(p.parse(&json!(7)).unwrap(), Some(Literal::Int(7)));
        assert_eq!(p.parse(&json!(7.5)).unwrap(), Some(Literal::Float(7.5)));
        assert_eq!(p.parse(&json!("42")).unwrap(), Some(Literal::Int(42)));
        assert_eq!(p.parse(&json!("3.5")).unwrap(), Some(Literal::Float(3.5)));
        assert_eq!(p.parse(&json!("1,234")).unwrap(), Some(Literal::Int(1234)));
        assert_eq!(p.parse(&json!(null)).unwrap(), None);
    }

    #[test]
    fn number_handles_percentages() {
        let p = NumberParser;
        assert_eq!(p.parse(&json!("50%")).unwrap(), Some(Literal::Float(0.5)));
        assert_eq!(
            p.parse(&json!("12.5%")).unwrap(),
            Some(Literal::Float(0.125))
        );
        assert_eq!(p.parse(&json!("100%")).unwrap(), Some(Literal::Float(1.0)));
    }

    #[test]
    fn number_rejects_garbage_and_booleans() {
        let p = NumberParser;
        assert!(p.parse(&json!("not a number")).is_err());
        assert!(p.parse(&json!(true)).is_err());
        assert!(p.parse(&json!("")).is_err());
    }

    #[test]
    fn boolean_accepts_native_and_common_strings() {
        let p = BooleanParser;
        assert_eq!(p.parse(&json!(true)).unwrap(), Some(Literal::Bool(true)));
        assert_eq!(p.parse(&json!("yes")).unwrap(), Some(Literal::Bool(true)));
        assert_eq!(
            p.parse(&json!("FALSE")).unwrap(),
            Some(Literal::Bool(false))
        );
        assert_eq!(p.parse(&json!("1")).unwrap(), Some(Literal::Bool(true)));
        assert_eq!(p.parse(&json!("off")).unwrap(), Some(Literal::Bool(false)));
        assert_eq!(p.parse(&json!(null)).unwrap(), None);
    }

    #[test]
    fn boolean_rejects_numbers_and_unknown_strings() {
        let p = BooleanParser;
        assert!(p.parse(&json!(1)).is_err());
        assert!(p.parse(&json!("maybe")).is_err());
    }

    #[test]
    fn date_parses_iso_and_epoch_seconds() {
        let p = DateParser;
        assert_eq!(
            p.parse(&json!("2024-05-06")).unwrap(),
            Some(Literal::String("2024-05-06".into()))
        );
        assert_eq!(
            p.parse(&json!("2024-05-06T12:30:00Z")).unwrap(),
            Some(Literal::String("2024-05-06".into()))
        );
        // 2024-01-01T00:00:00Z = 1704067200
        assert_eq!(
            p.parse(&json!(1_704_067_200)).unwrap(),
            Some(Literal::String("2024-01-01".into()))
        );
        assert_eq!(p.parse(&json!(null)).unwrap(), None);
    }

    #[test]
    fn date_rejects_garbage() {
        let p = DateParser;
        assert!(p.parse(&json!("not a date")).is_err());
        assert!(p.parse(&json!("2024-13-01")).is_err());
        assert!(p.parse(&json!("2023-02-29")).is_err());
        assert!(p.parse(&json!(true)).is_err());
    }

    #[test]
    fn timestamp_normalises_to_z() {
        let p = TimestampParser;
        assert_eq!(
            p.parse(&json!("2024-05-06T12:30:00")).unwrap(),
            Some(Literal::String("2024-05-06T12:30:00Z".into()))
        );
        assert_eq!(
            p.parse(&json!("2024-05-06")).unwrap(),
            Some(Literal::String("2024-05-06T00:00:00Z".into()))
        );
        assert_eq!(
            p.parse(&json!("2024-05-06T12:30:00+02:00")).unwrap(),
            Some(Literal::String("2024-05-06T12:30:00+02:00".into()))
        );
        assert_eq!(
            p.parse(&json!(1_704_067_200)).unwrap(),
            Some(Literal::String("2024-01-01T00:00:00Z".into()))
        );
    }

    #[test]
    fn timestamp_rejects_garbage() {
        let p = TimestampParser;
        assert!(p.parse(&json!("not a timestamp")).is_err());
        assert!(p.parse(&json!("2024-05-06T25:00:00Z")).is_err());
    }

    #[test]
    fn epoch_round_trip() {
        // 1970-01-01T00:00:00Z
        assert_eq!(epoch_to_timestamp(0), "1970-01-01T00:00:00Z");
        // 2000-02-29T12:34:56Z (leap day)
        // 2000-02-29 is 30 years, 7 leap years between 1970 and 2000 (72,76,80,84,88,92,96)
        // days = 30*365 + 7 = 10957 from 1970-01-01 to 2000-01-01
        // + 31 days (Jan) + 28 days (Feb) = 11016 -> 2000-02-29.
        let secs: i64 = 11_016 * 86_400 + 12 * 3600 + 34 * 60 + 56;
        assert_eq!(epoch_to_timestamp(secs), "2000-02-29T12:34:56Z");
    }
}
