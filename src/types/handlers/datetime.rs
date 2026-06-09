//! Query-capable handler for the built-in `Date` / `Timestamp` types.
//!
//! A datetime filter value names an *interval*, not an instant. When an
//! LLM emits
//!
//! ```json
//! { "field": "sv.work_start", "op": "eq", "value": "2026-05-15T00:00:00" }
//! ```
//!
//! it means "every row whose `work_start` falls on 2026-05-15" — but the
//! stored values carry a real time-of-day (`2026-05-15T10:12:32`), so a
//! literal string `=` never matches. This handler fixes that by reading
//! the *granularity* of the filter value: a bare `YYYY-MM-DD`, or a
//! timestamp at exactly midnight, is treated as the whole calendar day
//! and lowered to a half-open range `[day, day+1)`. A value with a real
//! time-of-day keeps exact-instant semantics.
//!
//! Day-range bounds are emitted as date-only strings (`YYYY-MM-DD`).
//! ISO-8601 is fixed-width, so a lexicographic `>=` / `<` over the date
//! prefix is a correct range test whether the stored value is date-only
//! (`2026-05-15`) or a full timestamp (`2026-05-15T10:12:32`, with or
//! without a `Z`/offset suffix).

use std::collections::BTreeMap;

use serde_json::Value;

use crate::ast::query::{Literal, PropertyRef};
use crate::types::context::{EmitCtx, IngestCtx, LowerCtx, PromptHint};
use crate::types::{
    BuiltinType, Capabilities, TypeError, TypeHandler, TypeId, TypedOp, TypedPredicate,
};

use super::core::{
    days_in_month, epoch_to_ymdhms, json_kind, DateParser, ScalarParser, TimestampParser,
};

/// Handler shared by the `Date` and `Timestamp` types. Ingestion is
/// delegated to a [`ScalarParser`]; query lowering applies the
/// interval-aware semantics described in the module docs.
pub struct DateTimeHandler {
    type_id: TypeId,
    parser: Box<dyn ScalarParser>,
}

impl std::fmt::Debug for DateTimeHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DateTimeHandler")
            .field("type_id", &self.type_id)
            .finish_non_exhaustive()
    }
}

impl DateTimeHandler {
    pub fn new(type_id: impl Into<TypeId>, parser: Box<dyn ScalarParser>) -> Self {
        Self {
            type_id: type_id.into(),
            parser,
        }
    }

    fn ops() -> Vec<TypedOp> {
        vec![
            TypedOp::Eq,
            TypedOp::Neq,
            TypedOp::Gt,
            TypedOp::Gte,
            TypedOp::Lt,
            TypedOp::Lte,
        ]
    }
}

impl TypeHandler for DateTimeHandler {
    fn type_id(&self) -> TypeId {
        self.type_id.clone()
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::INGEST | Capabilities::EXACT_MATCH | Capabilities::RANGE
    }

    fn supported_ops(&self) -> Vec<TypedOp> {
        Self::ops()
    }

    fn on_ingest(&self, ctx: &mut IngestCtx<'_>) -> Result<(), TypeError> {
        match self.parser.parse(ctx.value())? {
            Some(lit) => ctx.set_value(lit),
            None => ctx.skip(),
        }
        Ok(())
    }

    fn lower(&self, ctx: &mut LowerCtx<'_>) -> Result<TypedPredicate, TypeError> {
        let op = ctx.raw.op;
        if !Self::ops().contains(&op) {
            return Err(TypeError::UnsupportedOp {
                ty: self.type_id.to_string(),
                op: op.to_string(),
            });
        }

        let resolved = resolve(self.type_id.as_str(), ctx.raw.value)?;
        let mut params: BTreeMap<String, Literal> = BTreeMap::new();
        let value = if resolved.whole_day {
            let start = format!("{:04}-{:02}-{:02}", resolved.y, resolved.m, resolved.d);
            let (ny, nm, nd) = next_day(resolved.y, resolved.m, resolved.d);
            let end = format!("{ny:04}-{nm:02}-{nd:02}");
            params.insert("start".to_string(), Literal::String(start.clone()));
            params.insert("end".to_string(), Literal::String(end));
            Literal::String(start)
        } else {
            let exact = format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                resolved.y, resolved.m, resolved.d, resolved.h, resolved.mi, resolved.s
            );
            params.insert("exact".to_string(), Literal::String(exact.clone()));
            Literal::String(exact)
        };

        Ok(TypedPredicate {
            type_id: ctx.type_id.clone(),
            field: ctx.raw.field.clone(),
            op,
            value,
            params,
        })
    }

    fn emit(&self, ctx: &mut EmitCtx<'_>, pred: &TypedPredicate) -> Result<(), TypeError> {
        let lhs = render_property(&pred.field);

        // Whole-day predicate: lowered to a half-open day range.
        if let Some(Literal::String(start)) = pred.params.get("start") {
            let Some(Literal::String(end)) = pred.params.get("end") else {
                return Err(TypeError::Handler(
                    "datetime day predicate is missing its upper bound".into(),
                ));
            };
            let expr = match pred.op {
                TypedOp::Eq => {
                    let s = ctx.bind(Literal::String(start.clone()));
                    let e = ctx.bind(Literal::String(end.clone()));
                    format!("({lhs} >= {s} AND {lhs} < {e})")
                }
                TypedOp::Neq => {
                    let s = ctx.bind(Literal::String(start.clone()));
                    let e = ctx.bind(Literal::String(end.clone()));
                    format!("({lhs} < {s} OR {lhs} >= {e})")
                }
                TypedOp::Gt => {
                    let e = ctx.bind(Literal::String(end.clone()));
                    format!("{lhs} >= {e}")
                }
                TypedOp::Gte => {
                    let s = ctx.bind(Literal::String(start.clone()));
                    format!("{lhs} >= {s}")
                }
                TypedOp::Lt => {
                    let s = ctx.bind(Literal::String(start.clone()));
                    format!("{lhs} < {s}")
                }
                TypedOp::Lte => {
                    let e = ctx.bind(Literal::String(end.clone()));
                    format!("{lhs} < {e}")
                }
                other => return Err(unsupported(&self.type_id, other)),
            };
            ctx.set_where(expr);
            return Ok(());
        }

        // Instant predicate: exact comparison at the value's precision.
        if let Some(Literal::String(exact)) = pred.params.get("exact") {
            let op = match pred.op {
                TypedOp::Eq => "=",
                TypedOp::Neq => "<>",
                TypedOp::Gt => ">",
                TypedOp::Gte => ">=",
                TypedOp::Lt => "<",
                TypedOp::Lte => "<=",
                other => return Err(unsupported(&self.type_id, other)),
            };
            let placeholder = ctx.bind(Literal::String(exact.clone()));
            ctx.set_where(format!("{lhs} {op} {placeholder}"));
            return Ok(());
        }

        Err(TypeError::Handler(
            "datetime predicate is missing its resolved bounds".into(),
        ))
    }

    fn prompt_hint(&self) -> PromptHint {
        PromptHint {
            type_id: self.type_id(),
            capabilities: self.capabilities(),
            ops: Self::ops(),
            doc: Some(
                "calendar date / timestamp. A date-only value (or a timestamp at \
                 midnight) is treated as the whole calendar day: `eq` on \
                 \"2026-05-15\" matches every row recorded on that day, and \
                 `gt`/`lt`/etc. compare against the day boundary. Pass an \
                 explicit time of day for instant-precision comparisons."
                    .into(),
            ),
            example: Some(
                r#"{"field": "sv.work_start", "op": "eq", "value": "2026-05-15"}"#.into(),
            ),
        }
    }
}

/// Build a [`DateTimeHandler`] for the built-in `Date` type.
pub fn date_handler() -> DateTimeHandler {
    DateTimeHandler::new(BuiltinType::Date.type_id(), Box::new(DateParser))
}

/// Build a [`DateTimeHandler`] for the built-in `Timestamp` type.
pub fn timestamp_handler() -> DateTimeHandler {
    DateTimeHandler::new(BuiltinType::Timestamp.type_id(), Box::new(TimestampParser))
}

// ─── Value resolution ───────────────────────────────────────────────────

/// A datetime filter value resolved to civil components plus the
/// precision the author expressed it at.
struct Resolved {
    y: i64,
    m: u32,
    d: u32,
    h: u32,
    mi: u32,
    s: u32,
    /// `true` when the value carries no time-of-day (a bare
    /// `YYYY-MM-DD`) or names exactly midnight — both read as "the whole
    /// calendar day".
    whole_day: bool,
}

fn resolve(type_id: &str, raw: &Value) -> Result<Resolved, TypeError> {
    match raw {
        Value::String(s) => parse_string(type_id, s),
        Value::Number(n) => {
            let secs = n.as_i64().ok_or_else(|| {
                invalid(type_id, format!("expected integer epoch seconds, got {n}"))
            })?;
            let (y, m, d, h, mi, s) = epoch_to_ymdhms(secs);
            Ok(Resolved {
                y,
                m,
                d,
                h,
                mi,
                s,
                whole_day: h == 0 && mi == 0 && s == 0,
            })
        }
        other => Err(invalid(
            type_id,
            format!(
                "expected a date/timestamp string or epoch seconds, got {}",
                json_kind(other)
            ),
        )),
    }
}

fn parse_string(type_id: &str, raw: &str) -> Result<Resolved, TypeError> {
    let s = raw.trim();
    if s.len() < 10 {
        return Err(invalid(type_id, format!("not an ISO-8601 date: {raw:?}")));
    }
    let date = &s[..10];
    let b = date.as_bytes();
    let shape_ok = b[0..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit);
    if !shape_ok {
        return Err(invalid(type_id, format!("not an ISO-8601 date: {raw:?}")));
    }
    let y: i64 = date[0..4].parse().expect("digits checked");
    let m: u32 = date[5..7].parse().expect("digits checked");
    let d: u32 = date[8..10].parse().expect("digits checked");
    if !(1..=12).contains(&m) {
        return Err(invalid(type_id, format!("month out of range in {raw:?}")));
    }
    if !(1..=days_in_month(y as u32, m)).contains(&d) {
        return Err(invalid(type_id, format!("day out of range in {raw:?}")));
    }

    let rest = &s[10..];
    if rest.is_empty() {
        return Ok(Resolved {
            y,
            m,
            d,
            h: 0,
            mi: 0,
            s: 0,
            whole_day: true,
        });
    }

    let body = rest
        .strip_prefix('T')
        .or_else(|| rest.strip_prefix(' '))
        .ok_or_else(|| invalid(type_id, format!("expected 'T' separator in {raw:?}")))?;
    let (h, mi, sec, frac_zero) = parse_time(type_id, body, raw)?;
    Ok(Resolved {
        y,
        m,
        d,
        h,
        mi,
        s: sec,
        whole_day: h == 0 && mi == 0 && sec == 0 && frac_zero,
    })
}

/// Parse the `HH:MM[:SS][.fff][Z|±hh:mm]` tail of a timestamp. The zone
/// suffix is accepted but ignored — comparisons run on the civil
/// components. Returns `(hour, minute, second, fractional_is_zero)`.
fn parse_time(type_id: &str, body: &str, raw: &str) -> Result<(u32, u32, u32, bool), TypeError> {
    let core = match body.find(['Z', '+', '-']) {
        Some(i) => &body[..i],
        None => body,
    };
    let (hms, frac) = match core.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (core, None),
    };
    let parts: Vec<&str> = hms.split(':').collect();
    if !(2..=3).contains(&parts.len())
        || parts.iter().any(|p| !p.bytes().all(|c| c.is_ascii_digit()))
    {
        return Err(invalid(
            type_id,
            format!("invalid time component in {raw:?}"),
        ));
    }
    let h: u32 = parts[0]
        .parse()
        .map_err(|_| invalid(type_id, format!("invalid hour in {raw:?}")))?;
    let mi: u32 = parts[1]
        .parse()
        .map_err(|_| invalid(type_id, format!("invalid minute in {raw:?}")))?;
    let sec: u32 = match parts.get(2) {
        Some(p) => p
            .parse()
            .map_err(|_| invalid(type_id, format!("invalid second in {raw:?}")))?,
        None => 0,
    };
    if h > 23 || mi > 59 || sec > 60 {
        return Err(invalid(type_id, format!("time out of range in {raw:?}")));
    }
    let frac_zero = match frac {
        None => true,
        Some(f) => !f.is_empty() && f.bytes().all(|c| c == b'0'),
    };
    Ok((h, mi, sec, frac_zero))
}

/// The calendar day after `(y, m, d)`, rolling month and year over.
fn next_day(y: i64, m: u32, d: u32) -> (i64, u32, u32) {
    if d < days_in_month(y as u32, m) {
        (y, m, d + 1)
    } else if m < 12 {
        (y, m + 1, 1)
    } else {
        (y + 1, 1, 1)
    }
}

fn invalid(type_id: &str, reason: String) -> TypeError {
    TypeError::InvalidValue {
        ty: type_id.to_string(),
        reason,
    }
}

fn unsupported(type_id: &TypeId, op: TypedOp) -> TypeError {
    TypeError::UnsupportedOp {
        ty: type_id.to_string(),
        op: op.to_string(),
    }
}

fn render_property(p: &PropertyRef) -> String {
    match &p.property {
        Some(prop) => format!("{}.{}", p.alias, prop),
        None => p.alias.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use serde_json::json;

    use crate::ast::query::{Alias, PropertyRef};
    use crate::types::context::{CypherContribution, ParamBinder, RawTypedFilter};

    #[derive(Default)]
    struct TestBinder {
        params: BTreeMap<String, Literal>,
        next_id: usize,
    }

    impl ParamBinder for TestBinder {
        fn bind(&mut self, value: Literal) -> String {
            let name = format!("p{}", self.next_id);
            self.next_id += 1;
            self.params.insert(name.clone(), value);
            format!("${name}")
        }

        fn fresh_id(&mut self) -> usize {
            let id = self.next_id;
            self.next_id += 1;
            id
        }
    }

    fn field() -> PropertyRef {
        PropertyRef {
            alias: Alias::new("sv"),
            property: Some("work_start".into()),
        }
    }

    fn lower(op: TypedOp, value: serde_json::Value) -> TypedPredicate {
        let handler = timestamp_handler();
        let field = field();
        let raw = RawTypedFilter {
            field: &field,
            op,
            value: &value,
        };
        let mut ctx = LowerCtx {
            raw,
            type_id: TypeId::new("Timestamp"),
            field_label: Some("ServiceVisit"),
            prefix_index: None,
        };
        handler.lower(&mut ctx).unwrap()
    }

    fn emit(pred: &TypedPredicate) -> (Option<String>, BTreeMap<String, Literal>) {
        let handler = timestamp_handler();
        let mut contribution = CypherContribution::default();
        let mut binder = TestBinder::default();
        let mut ctx = EmitCtx::new(&mut contribution, &mut binder);
        handler.emit(&mut ctx, pred).unwrap();
        (contribution.where_inline, binder.params)
    }

    #[test]
    fn eq_on_midnight_timestamp_expands_to_a_day_range() {
        let pred = lower(TypedOp::Eq, json!("2026-05-15T00:00:00"));
        let (where_inline, params) = emit(&pred);
        assert_eq!(
            where_inline.as_deref(),
            Some("(sv.work_start >= $p0 AND sv.work_start < $p1)")
        );
        assert_eq!(
            params.get("p0"),
            Some(&Literal::String("2026-05-15".into()))
        );
        assert_eq!(
            params.get("p1"),
            Some(&Literal::String("2026-05-16".into()))
        );
    }

    #[test]
    fn eq_on_bare_date_expands_to_a_day_range() {
        let pred = lower(TypedOp::Eq, json!("2026-05-15"));
        let (where_inline, params) = emit(&pred);
        assert_eq!(
            where_inline.as_deref(),
            Some("(sv.work_start >= $p0 AND sv.work_start < $p1)")
        );
        assert_eq!(
            params.get("p1"),
            Some(&Literal::String("2026-05-16".into()))
        );
    }

    #[test]
    fn eq_on_value_with_real_time_keeps_instant_semantics() {
        let pred = lower(TypedOp::Eq, json!("2026-05-15T10:12:32"));
        let (where_inline, params) = emit(&pred);
        assert_eq!(where_inline.as_deref(), Some("sv.work_start = $p0"));
        assert_eq!(
            params.get("p0"),
            Some(&Literal::String("2026-05-15T10:12:32".into()))
        );
    }

    #[test]
    fn neq_on_day_negates_the_range() {
        let pred = lower(TypedOp::Neq, json!("2026-05-15"));
        let (where_inline, _) = emit(&pred);
        assert_eq!(
            where_inline.as_deref(),
            Some("(sv.work_start < $p0 OR sv.work_start >= $p1)")
        );
    }

    #[test]
    fn day_range_comparisons_use_the_right_boundary() {
        // `gt` a whole day → strictly after it → next day onwards.
        let (gt, _) = emit(&lower(TypedOp::Gt, json!("2026-05-15")));
        assert_eq!(gt.as_deref(), Some("sv.work_start >= $p0"));

        // `gte` → from the start of the day.
        let (gte, gte_params) = emit(&lower(TypedOp::Gte, json!("2026-05-15")));
        assert_eq!(gte.as_deref(), Some("sv.work_start >= $p0"));
        assert_eq!(
            gte_params.get("p0"),
            Some(&Literal::String("2026-05-15".into()))
        );

        // `lt` → before the start of the day.
        let (lt, _) = emit(&lower(TypedOp::Lt, json!("2026-05-15")));
        assert_eq!(lt.as_deref(), Some("sv.work_start < $p0"));

        // `lte` → through the end of the day.
        let (lte, lte_params) = emit(&lower(TypedOp::Lte, json!("2026-05-15")));
        assert_eq!(lte.as_deref(), Some("sv.work_start < $p0"));
        assert_eq!(
            lte_params.get("p0"),
            Some(&Literal::String("2026-05-16".into()))
        );
    }

    #[test]
    fn day_range_rolls_over_month_and_year_boundaries() {
        let pred = lower(TypedOp::Eq, json!("2026-12-31"));
        let (_, params) = emit(&pred);
        assert_eq!(
            params.get("p1"),
            Some(&Literal::String("2027-01-01".into()))
        );

        let feb = lower(TypedOp::Eq, json!("2024-02-29"));
        let (_, feb_params) = emit(&feb);
        assert_eq!(
            feb_params.get("p1"),
            Some(&Literal::String("2024-03-01".into()))
        );
    }

    #[test]
    fn epoch_seconds_are_accepted() {
        // 2024-01-01T00:00:00Z = 1704067200 → midnight → whole day.
        let pred = lower(TypedOp::Eq, json!(1_704_067_200i64));
        let (where_inline, params) = emit(&pred);
        assert_eq!(
            where_inline.as_deref(),
            Some("(sv.work_start >= $p0 AND sv.work_start < $p1)")
        );
        assert_eq!(
            params.get("p0"),
            Some(&Literal::String("2024-01-01".into()))
        );
    }

    #[test]
    fn fractional_midnight_is_still_a_whole_day() {
        let pred = lower(TypedOp::Eq, json!("2026-05-15T00:00:00.000"));
        let (where_inline, _) = emit(&pred);
        assert_eq!(
            where_inline.as_deref(),
            Some("(sv.work_start >= $p0 AND sv.work_start < $p1)")
        );
    }

    #[test]
    fn zone_suffix_is_tolerated() {
        let pred = lower(TypedOp::Eq, json!("2026-05-15T00:00:00Z"));
        let (where_inline, _) = emit(&pred);
        assert_eq!(
            where_inline.as_deref(),
            Some("(sv.work_start >= $p0 AND sv.work_start < $p1)")
        );
        let offset = lower(TypedOp::Gte, json!("2026-05-15T08:30:00+02:00"));
        let (gte, params) = emit(&offset);
        assert_eq!(gte.as_deref(), Some("sv.work_start >= $p0"));
        assert_eq!(
            params.get("p0"),
            Some(&Literal::String("2026-05-15T08:30:00".into()))
        );
    }

    #[test]
    fn garbage_values_are_rejected() {
        let handler = timestamp_handler();
        let f = field();
        for bad in [
            json!("not a date"),
            json!("2026-13-01"),
            json!(true),
            json!(null),
        ] {
            let raw = RawTypedFilter {
                field: &f,
                op: TypedOp::Eq,
                value: &bad,
            };
            let mut ctx = LowerCtx {
                raw,
                type_id: TypeId::new("Timestamp"),
                field_label: None,
                prefix_index: None,
            };
            assert!(handler.lower(&mut ctx).is_err(), "expected error for {bad}");
        }
    }
}
