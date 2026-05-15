//! Query-capable handler for the built-in `Text` type.
//!
//! Text is stored in normalized form so exact/contains matching is less
//! sensitive to punctuation, spacing, and case differences.

use crate::ast::query::{Literal, PropertyRef};
use crate::types::context::{EmitCtx, IngestCtx, LowerCtx};
use crate::types::{Capabilities, TypeError, TypeHandler, TypeId, TypedOp, TypedPredicate};

use super::core::{json_kind, ScalarParser, TextParser};

#[derive(Debug, Default)]
pub struct TextHandler {
    parser: TextParser,
}

impl TextHandler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Normalize text for storage and query comparisons.
    pub fn normalize(value: &str) -> String {
        TextParser::normalize(value)
    }

    fn normalize_literal(value: &Literal) -> Result<Literal, TypeError> {
        match value {
            Literal::String(s) => Ok(Literal::String(Self::normalize(s))),
            Literal::Bool(b) => Ok(Literal::String(Self::normalize(&b.to_string()))),
            Literal::Int(i) => Ok(Literal::String(Self::normalize(&i.to_string()))),
            Literal::Float(f) => Ok(Literal::String(Self::normalize(&f.to_string()))),
            Literal::List(items) => {
                let items = items
                    .iter()
                    .map(Self::normalize_literal)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Literal::List(items))
            }
            Literal::Null => Ok(Literal::Null),
            Literal::Object(_) => Err(TypeError::InvalidValue {
                ty: "Text".into(),
                reason: "objects cannot be compared as Text".into(),
            }),
        }
    }

    fn literal_from_json(raw: &serde_json::Value) -> Result<Literal, TypeError> {
        Literal::from_json(raw).ok_or_else(|| TypeError::InvalidValue {
            ty: "Text".into(),
            reason: format!("expected scalar or array, got {}", json_kind(raw)),
        })
    }
}

impl TypeHandler for TextHandler {
    fn type_id(&self) -> TypeId {
        TypeId::new("Text")
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::INGEST | Capabilities::EXACT_MATCH | Capabilities::CONTAINS
    }

    fn supported_ops(&self) -> Vec<TypedOp> {
        vec![
            TypedOp::Eq,
            TypedOp::Neq,
            TypedOp::In,
            TypedOp::Contains,
            TypedOp::StartsWith,
            TypedOp::EndsWith,
        ]
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
        if !matches!(
            ctx.raw.op,
            TypedOp::Eq
                | TypedOp::Neq
                | TypedOp::In
                | TypedOp::Contains
                | TypedOp::StartsWith
                | TypedOp::EndsWith
        ) {
            return Err(TypeError::UnsupportedOp {
                ty: self.type_id().to_string(),
                op: ctx.raw.op.to_string(),
            });
        }

        let value = Self::literal_from_json(ctx.raw.value)?;
        if matches!(
            ctx.raw.op,
            TypedOp::Contains | TypedOp::StartsWith | TypedOp::EndsWith
        ) && !matches!(value, Literal::String(_))
        {
            return Err(TypeError::InvalidValue {
                ty: "Text".into(),
                reason: format!("{} expects a string value", ctx.raw.op),
            });
        }
        if matches!(ctx.raw.op, TypedOp::In) && !matches!(value, Literal::List(_)) {
            return Err(TypeError::InvalidValue {
                ty: "Text".into(),
                reason: "in expects an array value".into(),
            });
        }

        Ok(TypedPredicate {
            type_id: ctx.type_id.clone(),
            field: ctx.raw.field.clone(),
            op: ctx.raw.op,
            value,
            params: Default::default(),
        })
    }

    fn emit(&self, ctx: &mut EmitCtx<'_>, pred: &TypedPredicate) -> Result<(), TypeError> {
        let value = Self::normalize_literal(&pred.value)?;
        if matches!(
            pred.op,
            TypedOp::Contains | TypedOp::StartsWith | TypedOp::EndsWith
        ) && !matches!(value, Literal::String(_))
        {
            return Err(TypeError::InvalidValue {
                ty: "Text".into(),
                reason: format!("{} expects a string value", pred.op),
            });
        }
        if matches!(pred.op, TypedOp::In) && !matches!(value, Literal::List(_)) {
            return Err(TypeError::InvalidValue {
                ty: "Text".into(),
                reason: "in expects an array value".into(),
            });
        }

        let lhs = render_property(&pred.field);
        let placeholder = ctx.bind(value);
        let op = match pred.op {
            TypedOp::Eq => "=",
            TypedOp::Neq => "<>",
            TypedOp::In => "IN",
            TypedOp::Contains => "CONTAINS",
            TypedOp::StartsWith => "STARTS WITH",
            TypedOp::EndsWith => "ENDS WITH",
            other => {
                return Err(TypeError::UnsupportedOp {
                    ty: self.type_id().to_string(),
                    op: other.to_string(),
                })
            }
        };
        ctx.set_where(format!("{lhs} {op} {placeholder}"));
        Ok(())
    }
}

/// Build a [`TextHandler`] for the built-in `Text` type.
pub fn text_handler() -> TextHandler {
    TextHandler::new()
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

    #[test]
    fn text_normalize_removes_special_chars_and_lowercases() {
        assert_eq!(TextHandler::normalize(" AB-12  / cd_ЭЮ! "), "ab12cdэю");
    }

    #[test]
    fn text_handler_lower_and_emit_normalizes_eq() {
        let handler = TextHandler::new();
        let field = crate::ast::query::PropertyRef {
            alias: crate::ast::query::Alias::new("c"),
            property: Some("vin".into()),
        };
        let raw = crate::types::context::RawTypedFilter {
            field: &field,
            op: TypedOp::Eq,
            value: &json!("WBA 8G-510/60K587560"),
        };
        let mut lower_ctx = LowerCtx {
            raw,
            type_id: TypeId::new("Text"),
            field_label: Some("Car"),
            prefix_index: None,
        };
        let pred = handler.lower(&mut lower_ctx).unwrap();

        let mut contribution = crate::types::context::CypherContribution::default();
        let mut binder = TestBinder::default();
        let mut emit_ctx = EmitCtx::new(&mut contribution, &mut binder);
        handler.emit(&mut emit_ctx, &pred).unwrap();

        assert_eq!(contribution.where_inline.as_deref(), Some("c.vin = $p0"));
        assert_eq!(
            binder.params.get("p0"),
            Some(&Literal::String("wba8g51060k587560".into()))
        );
    }

    #[test]
    fn text_handler_emits_contains() {
        let handler = TextHandler::new();
        let pred = TypedPredicate {
            type_id: TypeId::new("Text"),
            field: crate::ast::query::PropertyRef {
                alias: crate::ast::query::Alias::new("c"),
                property: Some("name".into()),
            },
            op: TypedOp::Contains,
            value: Literal::String("BMW X".into()),
            params: Default::default(),
        };

        let mut contribution = crate::types::context::CypherContribution::default();
        let mut binder = TestBinder::default();
        let mut emit_ctx = EmitCtx::new(&mut contribution, &mut binder);
        handler.emit(&mut emit_ctx, &pred).unwrap();

        assert_eq!(
            contribution.where_inline.as_deref(),
            Some("c.name CONTAINS $p0")
        );
        assert_eq!(
            binder.params.get("p0"),
            Some(&Literal::String("bmwx".into()))
        );
    }

    #[test]
    fn text_handler_emits_starts_with_and_ends_with() {
        for (op, expected) in [
            (TypedOp::StartsWith, "c.name STARTS WITH $p0"),
            (TypedOp::EndsWith, "c.name ENDS WITH $p0"),
        ] {
            let handler = TextHandler::new();
            let pred = TypedPredicate {
                type_id: TypeId::new("Text"),
                field: crate::ast::query::PropertyRef {
                    alias: crate::ast::query::Alias::new("c"),
                    property: Some("name".into()),
                },
                op,
                value: Literal::String(" BMW X ".into()),
                params: Default::default(),
            };

            let mut contribution = crate::types::context::CypherContribution::default();
            let mut binder = TestBinder::default();
            let mut emit_ctx = EmitCtx::new(&mut contribution, &mut binder);
            handler.emit(&mut emit_ctx, &pred).unwrap();

            assert_eq!(contribution.where_inline.as_deref(), Some(expected));
            assert_eq!(
                binder.params.get("p0"),
                Some(&Literal::String("bmwx".into()))
            );
        }
    }

    #[test]
    fn text_handler_emits_in_with_normalized_list() {
        let handler = TextHandler::new();
        let pred = TypedPredicate {
            type_id: TypeId::new("Text"),
            field: crate::ast::query::PropertyRef {
                alias: crate::ast::query::Alias::new("c"),
                property: Some("name".into()),
            },
            op: TypedOp::In,
            value: Literal::List(vec![
                Literal::String("BMW X".into()),
                Literal::String("Audi-Q7".into()),
            ]),
            params: Default::default(),
        };

        let mut contribution = crate::types::context::CypherContribution::default();
        let mut binder = TestBinder::default();
        let mut emit_ctx = EmitCtx::new(&mut contribution, &mut binder);
        handler.emit(&mut emit_ctx, &pred).unwrap();

        assert_eq!(contribution.where_inline.as_deref(), Some("c.name IN $p0"));
        assert_eq!(
            binder.params.get("p0"),
            Some(&Literal::List(vec![
                Literal::String("bmwx".into()),
                Literal::String("audiq7".into())
            ]))
        );
    }

    #[test]
    fn text_handler_does_not_parse_primary_key_field_on_ingest() {
        let handler = TextHandler::new();
        let node_key = Literal::String("raw key".into());
        let raw = json!({"objects": "would normally be rejected"});
        let mut effects = crate::types::SideEffectQueue::new();
        let mut ctx = IngestCtx::new("Thing", "id", &node_key, "id", &raw, &mut effects);

        handler.on_ingest(&mut ctx).unwrap();

        assert_eq!(ctx.finish(), None);
    }

    #[test]
    fn text_handler_rejects_wrong_value_shapes_for_new_ops() {
        let handler = TextHandler::new();
        let field = crate::ast::query::PropertyRef {
            alias: crate::ast::query::Alias::new("c"),
            property: Some("name".into()),
        };

        let mut prefix_ctx = LowerCtx {
            raw: crate::types::context::RawTypedFilter {
                field: &field,
                op: TypedOp::StartsWith,
                value: &json!(["BMW"]),
            },
            type_id: TypeId::new("Text"),
            field_label: Some("Car"),
            prefix_index: None,
        };
        assert!(matches!(
            handler.lower(&mut prefix_ctx),
            Err(TypeError::InvalidValue { .. })
        ));

        let mut in_ctx = LowerCtx {
            raw: crate::types::context::RawTypedFilter {
                field: &field,
                op: TypedOp::In,
                value: &json!("BMW"),
            },
            type_id: TypeId::new("Text"),
            field_label: Some("Car"),
            prefix_index: None,
        };
        assert!(matches!(
            handler.lower(&mut in_ctx),
            Err(TypeError::InvalidValue { .. })
        ));
    }
}
