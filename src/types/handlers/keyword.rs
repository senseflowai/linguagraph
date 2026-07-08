//! Query-capable handler for the built-in `Keyword` type.
//!
//! A `Keyword` is a plain string stored **verbatim** (no normalization),
//! so Cypher matches it with the standard string operators: `=`, `<>`,
//! `<`, `>`, `<=`, `>=`, `=~` (regex), `CONTAINS`, `STARTS WITH`,
//! `ENDS WITH`, and `IN`. Use it for identifiers, codes, statuses, and
//! other short categorical values. Free-form / natural-language text that
//! should be searchable belongs to the `SemanticText` type instead.

use crate::ast::query::Literal;
use crate::types::context::{EmitCtx, IngestCtx, LowerCtx};
use crate::types::{
    BuiltinType, Capabilities, TypeError, TypeHandler, TypeId, TypedOp, TypedPredicate,
};

use super::core::{json_kind, KeywordParser, ScalarParser};

#[derive(Debug, Default)]
pub struct KeywordHandler {
    parser: KeywordParser,
}

impl KeywordHandler {
    pub fn new() -> Self {
        Self::default()
    }

    /// The set of operators a `Keyword` filter may use.
    fn ops() -> Vec<TypedOp> {
        vec![
            TypedOp::Eq,
            TypedOp::Neq,
            TypedOp::Gt,
            TypedOp::Gte,
            TypedOp::Lt,
            TypedOp::Lte,
            TypedOp::In,
            TypedOp::Contains,
            TypedOp::StartsWith,
            TypedOp::EndsWith,
            TypedOp::Matches,
        ]
    }

    fn literal_from_json(raw: &serde_json::Value) -> Result<Literal, TypeError> {
        Literal::from_json(raw).ok_or_else(|| TypeError::InvalidValue {
            ty: "Keyword".into(),
            reason: format!("expected scalar or array, got {}", json_kind(raw)),
        })
    }

    /// Ops that require a single string operand.
    fn requires_string(op: TypedOp) -> bool {
        matches!(
            op,
            TypedOp::Contains | TypedOp::StartsWith | TypedOp::EndsWith | TypedOp::Matches
        )
    }

    /// Validate the operand shape for an op against `value`.
    fn check_value_shape(op: TypedOp, value: &Literal) -> Result<(), TypeError> {
        if Self::requires_string(op) && !matches!(value, Literal::String(_)) {
            return Err(TypeError::InvalidValue {
                ty: "Keyword".into(),
                reason: format!("{op} expects a string value"),
            });
        }
        if matches!(op, TypedOp::In) && !matches!(value, Literal::List(_)) {
            return Err(TypeError::InvalidValue {
                ty: "Keyword".into(),
                reason: "in expects an array value".into(),
            });
        }
        Ok(())
    }
}

impl TypeHandler for KeywordHandler {
    fn type_id(&self) -> TypeId {
        BuiltinType::Keyword.type_id()
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::INGEST
            | Capabilities::EXACT_MATCH
            | Capabilities::RANGE
            | Capabilities::CONTAINS
    }

    fn supported_ops(&self) -> Vec<TypedOp> {
        Self::ops()
    }

    fn on_ingest(&self, ctx: &mut IngestCtx<'_>) -> Result<(), TypeError> {
        if ctx.node_key_field == ctx.field_name {
            return Ok(());
        }

        match self.parser.parse(ctx.value())? {
            Some(lit) => ctx.set_value(lit),
            None => ctx.skip(),
        }
        Ok(())
    }

    fn lower(&self, ctx: &mut LowerCtx<'_>) -> Result<TypedPredicate, TypeError> {
        if !Self::ops().contains(&ctx.raw.op) {
            return Err(TypeError::UnsupportedOp {
                ty: self.type_id().to_string(),
                op: ctx.raw.op.to_string(),
            });
        }

        // Store the value verbatim — no normalization.
        let value = Self::literal_from_json(ctx.raw.value)?;
        Self::check_value_shape(ctx.raw.op, &value)?;

        Ok(TypedPredicate {
            type_id: ctx.type_id.clone(),
            field: ctx.raw.field.clone(),
            op: ctx.raw.op,
            value,
            params: Default::default(),
        })
    }

    fn emit(&self, ctx: &mut EmitCtx<'_>, pred: &TypedPredicate) -> Result<(), TypeError> {
        // The stored predicate already carries the raw value.
        Self::check_value_shape(pred.op, &pred.value)?;

        let lhs = super::render_property(&pred.field);
        // Keyword values are short categorical strings (statuses, codes,
        // enum labels) whose casing is not semantically meaningful. We
        // match them case-insensitively by wrapping both sides in
        // `toLower(...)` for the operators where that preserves intent.
        //
        // Left untouched:
        //   * `Matches` (=~) — case is governed by regex flags.
        //   * ordering (`Gt`/`Gte`/`Lt`/`Lte`) — lexicographic order under
        //     `toLower` would change the semantics, and enums don't use it.
        //
        // Caveat: `toLower(n.field)` can prevent Memgraph from using an
        // index on `field`. Acceptable for low-cardinality enums; if a
        // hot high-cardinality path ever needs this, normalise on ingest
        // and match the normalised value instead.
        let op = match pred.op {
            TypedOp::Eq => "=",
            TypedOp::Neq => "<>",
            TypedOp::Gt => ">",
            TypedOp::Gte => ">=",
            TypedOp::Lt => "<",
            TypedOp::Lte => "<=",
            TypedOp::In => {
                let Literal::List(items) = &pred.value else {
                    return Err(TypeError::InvalidValue {
                        ty: "Keyword".into(),
                        reason: "in expects an array value".into(),
                    });
                };
                if items.is_empty() {
                    ctx.set_where("false".to_string());
                    return Ok(());
                }
                let clauses = items
                    .iter()
                    .map(|item| format!("toLower({lhs}) = toLower({})", ctx.bind(item.clone())))
                    .collect::<Vec<_>>()
                    .join(" OR ");
                ctx.set_where(format!("({clauses})"));
                return Ok(());
            }
            TypedOp::Contains => "CONTAINS",
            TypedOp::StartsWith => "STARTS WITH",
            TypedOp::EndsWith => "ENDS WITH",
            TypedOp::Matches => "=~",
            other => {
                return Err(TypeError::UnsupportedOp {
                    ty: self.type_id().to_string(),
                    op: other.to_string(),
                })
            }
        };
        let placeholder = ctx.bind(pred.value.clone());
        // Ordering and regex keep verbatim casing; equality, inequality
        // and substring ops fold case on both operands.
        let case_insensitive = matches!(
            pred.op,
            TypedOp::Eq
                | TypedOp::Neq
                | TypedOp::Contains
                | TypedOp::StartsWith
                | TypedOp::EndsWith
        );
        if case_insensitive {
            ctx.set_where(format!("toLower({lhs}) {op} toLower({placeholder})"));
        } else {
            ctx.set_where(format!("{lhs} {op} {placeholder}"));
        }
        Ok(())
    }
}

/// Build a [`KeywordHandler`] for the built-in `Keyword` type.
pub fn keyword_handler() -> KeywordHandler {
    KeywordHandler::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::query::PropertyRef;
    use std::collections::BTreeMap;

    use serde_json::json;

    #[derive(Default)]
    struct TestBinder {
        params: BTreeMap<String, Literal>,
        next_id: usize,
    }

    impl crate::types::context::ParamBinder for TestBinder {
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

    fn prop(name: &str) -> PropertyRef {
        PropertyRef {
            alias: crate::ast::query::Alias::new("c"),
            property: Some(name.into()),
        }
    }

    fn emit_pred(op: TypedOp, value: Literal) -> (Option<String>, BTreeMap<String, Literal>) {
        let handler = KeywordHandler::new();
        let pred = TypedPredicate {
            type_id: TypeId::new("Keyword"),
            field: prop("name"),
            op,
            value,
            params: Default::default(),
        };
        let mut contribution = crate::types::context::CypherContribution::default();
        let mut binder = TestBinder::default();
        let mut emit_ctx = EmitCtx::new(&mut contribution, &mut binder);
        handler.emit(&mut emit_ctx, &pred).unwrap();
        (contribution.where_inline, binder.params)
    }

    #[test]
    fn keyword_lower_and_emit_keep_value_verbatim() {
        let handler = KeywordHandler::new();
        let field = prop("vin");
        let raw = crate::types::context::RawTypedFilter {
            field: &field,
            op: TypedOp::Eq,
            value: &json!("WBA 8G-510/60K587560"),
        };
        let mut lower_ctx = LowerCtx {
            raw,
            type_id: TypeId::new("Keyword"),
            field_label: Some("Car"),
            prefix_index: None,
        };
        let pred = handler.lower(&mut lower_ctx).unwrap();

        let mut contribution = crate::types::context::CypherContribution::default();
        let mut binder = TestBinder::default();
        let mut emit_ctx = EmitCtx::new(&mut contribution, &mut binder);
        handler.emit(&mut emit_ctx, &pred).unwrap();

        assert_eq!(
            contribution.where_inline.as_deref(),
            Some("toLower(c.vin) = toLower($p0)")
        );
        // Stored verbatim — no normalization; casing is folded at match
        // time via toLower, not by mutating the bound value.
        assert_eq!(
            binder.params.get("p0"),
            Some(&Literal::String("WBA 8G-510/60K587560".into()))
        );
    }

    #[test]
    fn keyword_emits_contains_verbatim() {
        let (where_inline, params) = emit_pred(TypedOp::Contains, Literal::String("BMW X".into()));
        assert_eq!(
            where_inline.as_deref(),
            Some("toLower(c.name) CONTAINS toLower($p0)")
        );
        assert_eq!(params.get("p0"), Some(&Literal::String("BMW X".into())));
    }

    #[test]
    fn keyword_emits_starts_with_and_ends_with() {
        for (op, expected) in [
            (
                TypedOp::StartsWith,
                "toLower(c.name) STARTS WITH toLower($p0)",
            ),
            (TypedOp::EndsWith, "toLower(c.name) ENDS WITH toLower($p0)"),
        ] {
            let (where_inline, params) = emit_pred(op, Literal::String("BMW X".into()));
            assert_eq!(where_inline.as_deref(), Some(expected));
            assert_eq!(params.get("p0"), Some(&Literal::String("BMW X".into())));
        }
    }

    #[test]
    fn keyword_emits_ordering_operators() {
        for (op, expected) in [
            (TypedOp::Gt, "c.name > $p0"),
            (TypedOp::Gte, "c.name >= $p0"),
            (TypedOp::Lt, "c.name < $p0"),
            (TypedOp::Lte, "c.name <= $p0"),
        ] {
            let (where_inline, _) = emit_pred(op, Literal::String("M".into()));
            assert_eq!(where_inline.as_deref(), Some(expected));
        }
    }

    #[test]
    fn keyword_emits_case_insensitive_eq_and_neq() {
        let (eq, _) = emit_pred(TypedOp::Eq, Literal::String("Completed".into()));
        assert_eq!(eq.as_deref(), Some("toLower(c.name) = toLower($p0)"));

        let (neq, _) = emit_pred(TypedOp::Neq, Literal::String("Completed".into()));
        assert_eq!(neq.as_deref(), Some("toLower(c.name) <> toLower($p0)"));
    }

    #[test]
    fn keyword_emits_regex_match() {
        let (where_inline, params) =
            emit_pred(TypedOp::Matches, Literal::String("(?i)bmw.*".into()));
        assert_eq!(where_inline.as_deref(), Some("c.name =~ $p0"));
        assert_eq!(params.get("p0"), Some(&Literal::String("(?i)bmw.*".into())));
    }

    #[test]
    fn keyword_emits_in_with_verbatim_list() {
        let (where_inline, params) = emit_pred(
            TypedOp::In,
            Literal::List(vec![
                Literal::String("BMW X".into()),
                Literal::String("Audi-Q7".into()),
            ]),
        );
        assert_eq!(
            where_inline.as_deref(),
            Some("(toLower(c.name) = toLower($p0) OR toLower(c.name) = toLower($p1))")
        );
        assert_eq!(params.get("p0"), Some(&Literal::String("BMW X".into())));
        assert_eq!(params.get("p1"), Some(&Literal::String("Audi-Q7".into())));
    }

    #[test]
    fn keyword_does_not_parse_primary_key_field_on_ingest() {
        let handler = KeywordHandler::new();
        let node_key = Literal::String("raw key".into());
        let raw = json!({"objects": "would normally be rejected"});
        let mut effects = crate::types::SideEffectQueue::new();
        let mut ctx = IngestCtx::new("Thing", "id", &node_key, "id", &raw, &mut effects);

        handler.on_ingest(&mut ctx).unwrap();

        assert_eq!(ctx.finish(), None);
    }

    #[test]
    fn keyword_rejects_wrong_value_shapes_for_new_ops() {
        let handler = KeywordHandler::new();
        let field = prop("name");

        let mut starts_ctx = LowerCtx {
            raw: crate::types::context::RawTypedFilter {
                field: &field,
                op: TypedOp::StartsWith,
                value: &json!(["BMW"]),
            },
            type_id: TypeId::new("Keyword"),
            field_label: Some("Car"),
            prefix_index: None,
        };
        assert!(matches!(
            handler.lower(&mut starts_ctx),
            Err(TypeError::InvalidValue { .. })
        ));

        let mut matches_ctx = LowerCtx {
            raw: crate::types::context::RawTypedFilter {
                field: &field,
                op: TypedOp::Matches,
                value: &json!(42),
            },
            type_id: TypeId::new("Keyword"),
            field_label: Some("Car"),
            prefix_index: None,
        };
        assert!(matches!(
            handler.lower(&mut matches_ctx),
            Err(TypeError::InvalidValue { .. })
        ));

        let mut in_ctx = LowerCtx {
            raw: crate::types::context::RawTypedFilter {
                field: &field,
                op: TypedOp::In,
                value: &json!("BMW"),
            },
            type_id: TypeId::new("Keyword"),
            field_label: Some("Car"),
            prefix_index: None,
        };
        assert!(matches!(
            handler.lower(&mut in_ctx),
            Err(TypeError::InvalidValue { .. })
        ));
    }
}
