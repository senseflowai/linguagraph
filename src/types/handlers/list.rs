//! Query-capable handler for the built-in `List` type.
//!
//! A `List` property (e.g. `ACTED_IN.roles: ["Neo"]`) is stored verbatim,
//! same as `Keyword` (see [`OntologyPropertyType::handler_id`]). The only
//! reason this handler exists separately from [`super::KeywordHandler`] is
//! query emission: Cypher's `CONTAINS` is a *substring* operator that only
//! accepts strings, so `field CONTAINS $value` on a list property fails at
//! runtime (`Memgraph.ClientError: 'contains' argument at position 1 must
//! be either 'null' or 'string'`). `contains` on a list means *element
//! membership*, which Cypher spells `$value IN field`.

use crate::ast::query::Literal;
use crate::types::context::{EmitCtx, IngestCtx, LowerCtx};
use crate::types::{
    BuiltinType, Capabilities, TypeError, TypeHandler, TypeId, TypedOp, TypedPredicate,
};

use super::core::{json_kind, KeywordParser, ScalarParser};

#[derive(Debug, Default)]
pub struct ListHandler {
    parser: KeywordParser,
}

impl ListHandler {
    pub fn new() -> Self {
        Self::default()
    }

    fn literal_from_json(raw: &serde_json::Value) -> Result<Literal, TypeError> {
        Literal::from_json(raw).ok_or_else(|| TypeError::InvalidValue {
            ty: "List".into(),
            reason: format!("expected scalar or array, got {}", json_kind(raw)),
        })
    }
}

impl TypeHandler for ListHandler {
    fn type_id(&self) -> TypeId {
        BuiltinType::List.type_id()
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::INGEST | Capabilities::CONTAINS
    }

    // Only element membership makes sense for a list; `starts_with` /
    // `ends_with` (implied by the `CONTAINS` capability's default ops)
    // are substring operators that don't apply here.
    fn supported_ops(&self) -> Vec<TypedOp> {
        vec![TypedOp::Contains]
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
        if ctx.raw.op != TypedOp::Contains {
            return Err(TypeError::UnsupportedOp {
                ty: self.type_id().to_string(),
                op: ctx.raw.op.to_string(),
            });
        }

        let value = Self::literal_from_json(ctx.raw.value)?;
        if !matches!(value, Literal::String(_)) {
            return Err(TypeError::InvalidValue {
                ty: "List".into(),
                reason: "contains expects a string value (list element)".into(),
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
        if pred.op != TypedOp::Contains {
            return Err(TypeError::UnsupportedOp {
                ty: self.type_id().to_string(),
                op: pred.op.to_string(),
            });
        }
        let lhs = super::render_property(&pred.field);
        let placeholder = ctx.bind(pred.value.clone());
        // `contains` on a list means "some element contains the search
        // term as a substring" (e.g. a `roles: ["Captain Molyneux"]`
        // entry should match a search for "Captain"), not exact element
        // equality — mirrors Cypher's own `CONTAINS` substring semantics,
        // just applied per-element via `ANY(...)`. Case-folded to match
        // KeywordHandler's case-insensitive `contains` on scalar fields.
        ctx.set_where(format!(
            "ANY(__el__ IN {lhs} WHERE toLower(__el__) CONTAINS toLower({placeholder}))"
        ));
        Ok(())
    }
}

/// Build a [`ListHandler`] for the built-in `List` type.
pub fn list_handler() -> ListHandler {
    ListHandler::new()
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
            alias: crate::ast::query::Alias::new("e_m"),
            property: Some(name.into()),
        }
    }

    #[test]
    fn list_lower_and_emit_use_any_in_membership() {
        let handler = ListHandler::new();
        let field = prop("roles");
        let raw = crate::types::context::RawTypedFilter {
            field: &field,
            op: TypedOp::Contains,
            value: &json!("Captain"),
        };
        let mut lower_ctx = LowerCtx {
            raw,
            type_id: TypeId::new("List"),
            field_label: Some("ACTED_IN"),
            prefix_index: None,
        };
        let pred = handler.lower(&mut lower_ctx).unwrap();

        let mut contribution = crate::types::context::CypherContribution::default();
        let mut binder = TestBinder::default();
        let mut emit_ctx = EmitCtx::new(&mut contribution, &mut binder);
        handler.emit(&mut emit_ctx, &pred).unwrap();

        assert_eq!(
            contribution.where_inline.as_deref(),
            Some("ANY(__el__ IN e_m.roles WHERE toLower(__el__) CONTAINS toLower($p0))")
        );
        assert_eq!(
            binder.params.get("p0"),
            Some(&Literal::String("Captain".into()))
        );
    }

    #[test]
    fn list_rejects_non_contains_ops() {
        let handler = ListHandler::new();
        let field = prop("roles");
        let raw = crate::types::context::RawTypedFilter {
            field: &field,
            op: TypedOp::Eq,
            value: &json!("Captain"),
        };
        let mut lower_ctx = LowerCtx {
            raw,
            type_id: TypeId::new("List"),
            field_label: Some("ACTED_IN"),
            prefix_index: None,
        };
        assert!(matches!(
            handler.lower(&mut lower_ctx),
            Err(TypeError::UnsupportedOp { .. })
        ));
    }
}
