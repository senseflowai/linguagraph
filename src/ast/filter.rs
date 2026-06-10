//! Unified predicate representation.
//!
//! Today [`super::query::FilterExpression`] has two parallel
//! predicate shapes:
//!
//! * [`super::query::Predicate`] for built-in scalar ops (`ComparisonOp`).
//! * [`crate::types::TypedPredicate`] for handler-dispatched ops
//!   (`TypedOp`).
//!
//! Plain ops (`Eq`, `Neq`, `Contains`, …) live in *both* enums, with
//! the resolver deciding which path a filter takes. The builder then
//! branches on the AST variant rather than on the predicate's actual
//! semantics.
//!
//! [`ResolvedPredicate`] collapses both shapes into one:
//!
//! ```ignore
//! struct ResolvedPredicate {
//!     field: PropertyRef,
//!     op: Op,                                     // closed, unified
//!     value: Literal,
//!     handler: HandlerRef,                        // Plain | Typed { TypeId }
//!     params: BTreeMap<String, Literal>,          // handler scratchpad
//! }
//! ```
//!
//! The producer/consumer migration is incremental: [`ResolvedPredicate::from_legacy_*`]
//! constructors exist so call sites can flip to the unified shape one
//! at a time. A follow-up step (introducing the logical plan) routes
//! the builder through `ResolvedPredicate` exclusively and deletes
//! the parallel variants.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::query::{ComparisonOp, Literal, Predicate, PropertyRef};
use crate::types::{TypeId, TypedOp, TypedPredicate};

/// Unified op set. The first ten variants are scalar ops shared with
/// both [`ComparisonOp`] and [`TypedOp`]; the remaining ones are
/// handler-dispatched. Plain handlers reject any non-scalar op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Op {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    Contains,
    StartsWith,
    EndsWith,
    // ── Handler-dispatched ops ───────────────────────────────────
    Search,
    SearchReranked,
    HybridSearch,
    EntitySearch,
    Near,
}

impl Op {
    /// `true` when the op is one that a Plain (untyped) field reference
    /// can render directly to Cypher without consulting a handler.
    pub const fn is_scalar(self) -> bool {
        matches!(
            self,
            Op::Eq
                | Op::Neq
                | Op::Gt
                | Op::Gte
                | Op::Lt
                | Op::Lte
                | Op::In
                | Op::Contains
                | Op::StartsWith
                | Op::EndsWith
        )
    }

    pub fn from_comparison(op: ComparisonOp) -> Self {
        match op {
            ComparisonOp::Eq => Op::Eq,
            ComparisonOp::Neq => Op::Neq,
            ComparisonOp::Gt => Op::Gt,
            ComparisonOp::Gte => Op::Gte,
            ComparisonOp::Lt => Op::Lt,
            ComparisonOp::Lte => Op::Lte,
            ComparisonOp::In => Op::In,
            ComparisonOp::Contains => Op::Contains,
            ComparisonOp::StartsWith => Op::StartsWith,
            ComparisonOp::EndsWith => Op::EndsWith,
        }
    }

    /// Try to recover a [`ComparisonOp`] from a unified op. Returns
    /// `None` for handler-dispatched ops that have no scalar
    /// equivalent.
    pub fn to_comparison(self) -> Option<ComparisonOp> {
        Some(match self {
            Op::Eq => ComparisonOp::Eq,
            Op::Neq => ComparisonOp::Neq,
            Op::Gt => ComparisonOp::Gt,
            Op::Gte => ComparisonOp::Gte,
            Op::Lt => ComparisonOp::Lt,
            Op::Lte => ComparisonOp::Lte,
            Op::In => ComparisonOp::In,
            Op::Contains => ComparisonOp::Contains,
            Op::StartsWith => ComparisonOp::StartsWith,
            Op::EndsWith => ComparisonOp::EndsWith,
            _ => return None,
        })
    }

    pub fn from_typed(op: TypedOp) -> Self {
        match op {
            TypedOp::Eq => Op::Eq,
            TypedOp::Neq => Op::Neq,
            TypedOp::Gt => Op::Gt,
            TypedOp::Gte => Op::Gte,
            TypedOp::Lt => Op::Lt,
            TypedOp::Lte => Op::Lte,
            TypedOp::In => Op::In,
            TypedOp::Contains => Op::Contains,
            TypedOp::StartsWith => Op::StartsWith,
            TypedOp::EndsWith => Op::EndsWith,
            TypedOp::Search => Op::Search,
            TypedOp::SearchReranked => Op::SearchReranked,
            TypedOp::HybridSearch => Op::HybridSearch,
            TypedOp::EntitySearch => Op::EntitySearch,
            TypedOp::Near => Op::Near,
        }
    }

    pub fn to_typed(self) -> TypedOp {
        match self {
            Op::Eq => TypedOp::Eq,
            Op::Neq => TypedOp::Neq,
            Op::Gt => TypedOp::Gt,
            Op::Gte => TypedOp::Gte,
            Op::Lt => TypedOp::Lt,
            Op::Lte => TypedOp::Lte,
            Op::In => TypedOp::In,
            Op::Contains => TypedOp::Contains,
            Op::StartsWith => TypedOp::StartsWith,
            Op::EndsWith => TypedOp::EndsWith,
            Op::Search => TypedOp::Search,
            Op::SearchReranked => TypedOp::SearchReranked,
            Op::HybridSearch => TypedOp::HybridSearch,
            Op::EntitySearch => TypedOp::EntitySearch,
            Op::Near => TypedOp::Near,
        }
    }
}

/// What renders this predicate.
///
/// `Plain` — built-in scalar emitter; the value goes into a single
/// `$pN` parameter and the op rendered as a Cypher infix operator.
///
/// `Typed` — dispatched through the [`crate::types::TypeRegistry`] at
/// emit time. The handler decides what Cypher fragments to splice
/// where.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandlerRef {
    Plain,
    Typed { type_id: TypeId },
}

/// Resolved, ready-to-emit predicate. The unified successor to the
/// [`Predicate`] / [`TypedPredicate`] split.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedPredicate {
    pub field: PropertyRef,
    pub op: Op,
    pub value: Literal,
    pub handler: HandlerRef,
    /// Handler-private scratchpad. Unused (empty) for `HandlerRef::Plain`.
    #[serde(default)]
    pub params: BTreeMap<String, Literal>,
}

impl ResolvedPredicate {
    /// Lift a plain (built-in scalar) predicate.
    pub fn from_legacy_plain(p: &Predicate) -> Self {
        Self {
            field: p.field.clone(),
            op: Op::from_comparison(p.op),
            value: p.value.clone(),
            handler: HandlerRef::Plain,
            params: BTreeMap::new(),
        }
    }

    /// Lift a typed (handler-dispatched) predicate.
    pub fn from_legacy_typed(p: &TypedPredicate) -> Self {
        Self {
            field: p.field.clone(),
            op: Op::from_typed(p.op),
            value: p.value.clone(),
            handler: HandlerRef::Typed {
                type_id: p.type_id.clone(),
            },
            params: p.params.clone(),
        }
    }

    /// Convert back to a plain [`Predicate`]. Fails (returns `None`)
    /// when the op is handler-dispatched (`Search`, `Near`, …) — only
    /// scalar ops have a faithful plain representation.
    pub fn to_legacy_plain(&self) -> Option<Predicate> {
        Some(Predicate {
            field: self.field.clone(),
            op: self.op.to_comparison()?,
            value: self.value.clone(),
        })
    }

    /// Convert back to a [`TypedPredicate`]. Requires that `self` is
    /// `HandlerRef::Typed`; returns `None` for plain predicates.
    pub fn to_legacy_typed(&self) -> Option<TypedPredicate> {
        match &self.handler {
            HandlerRef::Plain => None,
            HandlerRef::Typed { type_id } => Some(TypedPredicate {
                type_id: type_id.clone(),
                field: self.field.clone(),
                op: self.op.to_typed(),
                value: self.value.clone(),
                params: self.params.clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::query::{Alias, ComparisonOp, Literal, Predicate, PropertyRef};

    fn pr(a: &str, p: Option<&str>) -> PropertyRef {
        PropertyRef {
            alias: Alias::new(a),
            property: p.map(str::to_string),
        }
    }

    #[test]
    fn plain_roundtrip_through_resolved() {
        let p = Predicate {
            field: pr("p", Some("age")),
            op: ComparisonOp::Gt,
            value: Literal::Int(30),
        };
        let resolved = ResolvedPredicate::from_legacy_plain(&p);
        assert!(matches!(resolved.handler, HandlerRef::Plain));
        assert_eq!(resolved.op, Op::Gt);
        let back = resolved.to_legacy_plain().unwrap();
        assert_eq!(back.op, ComparisonOp::Gt);
        assert_eq!(back.value, Literal::Int(30));
    }

    #[test]
    fn typed_roundtrip_through_resolved() {
        let mut params = BTreeMap::new();
        params.insert("k".to_string(), Literal::String("v".into()));
        let p = TypedPredicate {
            type_id: TypeId::new("SemanticText"),
            field: pr("c", Some("name")),
            op: TypedOp::Search,
            value: Literal::String("apple".into()),
            params: params.clone(),
        };
        let resolved = ResolvedPredicate::from_legacy_typed(&p);
        assert!(
            matches!(resolved.handler, HandlerRef::Typed { ref type_id } if type_id.as_str() == "SemanticText")
        );
        assert_eq!(resolved.op, Op::Search);
        let back = resolved.to_legacy_typed().unwrap();
        assert_eq!(back.op, TypedOp::Search);
        assert_eq!(back.params.get("k").cloned(), params.get("k").cloned());
    }

    #[test]
    fn scalar_op_classification() {
        for op in [
            Op::Eq,
            Op::Neq,
            Op::Gt,
            Op::Gte,
            Op::Lt,
            Op::Lte,
            Op::In,
            Op::Contains,
            Op::StartsWith,
            Op::EndsWith,
        ] {
            assert!(op.is_scalar(), "expected {op:?} to be scalar");
        }
        for op in [Op::Search, Op::SearchReranked, Op::HybridSearch, Op::Near] {
            assert!(!op.is_scalar(), "expected {op:?} to be non-scalar");
        }
    }

    #[test]
    fn plain_has_no_typed_form() {
        let p = ResolvedPredicate::from_legacy_plain(&Predicate {
            field: pr("p", Some("name")),
            op: ComparisonOp::Eq,
            value: Literal::String("x".into()),
        });
        assert!(p.to_legacy_typed().is_none());
    }

    #[test]
    fn handler_dispatched_op_has_no_plain_form() {
        let p = TypedPredicate {
            type_id: TypeId::new("SemanticText"),
            field: pr("c", Some("name")),
            op: TypedOp::Search,
            value: Literal::String("x".into()),
            params: BTreeMap::new(),
        };
        let resolved = ResolvedPredicate::from_legacy_typed(&p);
        assert!(resolved.to_legacy_plain().is_none());
    }
}
