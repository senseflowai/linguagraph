//! Alias interning and field references.
//!
//! Today the AST carries [`Alias(String)`](super::query::Alias) and
//! [`PropertyRef`](super::query::PropertyRef) — both stringly-typed.
//! `PropertyRef::property: Option<String>` overloads three meanings
//! (an entity reference, a property reference, and a sort-key fallback),
//! and every clone walks a few heap allocations.
//!
//! This module introduces a forward-compatible alternative:
//!
//! * [`AliasId`] — a small `Copy` handle that future AST nodes can carry
//!   without cloning strings.
//! * [`Binding`] — what an alias resolves to: a node or an edge with a
//!   graph label.
//! * [`BindingTable`] — the alias environment of a single query,
//!   constructible from an existing [`super::query::ReadQuery`].
//! * [`Field`] — a typed field reference: either the entity an alias
//!   binds (`Entity`), or one of its properties (`Property`). The
//!   `Option<String>` overload of [`PropertyRef`] disappears.
//!
//! These types are *parallel* to the existing AST today. The migration
//! is incremental: future passes (logical plan, normalization) will
//! consume them while the resolver and builder continue to produce
//! `PropertyRef` until a follow-up step swaps the producer side.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::query::{Alias, PropertyRef, ReadQuery};

/// Interned alias handle. `Copy`, cheap to pass and compare.
///
/// Always paired with a [`BindingTable`] that owns the underlying
/// strings. An `AliasId` is only valid inside the table that produced
/// it; using it elsewhere will index the wrong binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AliasId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BindingKind {
    Node,
    Edge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    pub name: Box<str>,
    /// Graph label. Empty string when the binding is label-less
    /// (e.g. `TraversalQuery` with no explicit entity label).
    pub label: Box<str>,
    pub kind: BindingKind,
}

/// Alias environment for a single query.
///
/// Bindings are stored in declaration order so `iter()` walks `start →
/// traversals[0].edge → traversals[0].target → …` — exactly the order
/// the builder needs when emitting MATCH patterns.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BindingTable {
    bindings: Vec<Binding>,
    #[serde(skip)]
    by_name: HashMap<Box<str>, AliasId>,
}

impl BindingTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a binding. If the name already exists, returns the
    /// existing id without touching the table (last-write-wins is not
    /// the right policy here — duplicate alias detection lives in the
    /// resolver).
    pub fn intern(&mut self, name: &str, label: &str, kind: BindingKind) -> AliasId {
        if let Some(id) = self.by_name.get(name) {
            return *id;
        }
        let id = AliasId(self.bindings.len() as u32);
        let name: Box<str> = name.into();
        self.bindings.push(Binding {
            name: name.clone(),
            label: label.into(),
            kind,
        });
        self.by_name.insert(name, id);
        id
    }

    pub fn lookup(&self, name: &str) -> Option<AliasId> {
        self.by_name.get(name).copied()
    }

    pub fn get(&self, id: AliasId) -> &Binding {
        &self.bindings[id.0 as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item = (AliasId, &Binding)> {
        self.bindings
            .iter()
            .enumerate()
            .map(|(i, b)| (AliasId(i as u32), b))
    }

    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Derive a binding table by walking an existing [`ReadQuery`].
    /// Bindings appear in start-then-traversal order, matching the
    /// builder's MATCH emission.
    pub fn from_read_query(q: &ReadQuery) -> Self {
        let mut t = Self::new();
        t.intern(q.start.alias.as_str(), &q.start.label, BindingKind::Node);
        for tr in &q.traversals {
            t.intern(tr.edge_alias.as_str(), &tr.edge_label, BindingKind::Edge);
            t.intern(
                tr.target.alias.as_str(),
                &tr.target.label,
                BindingKind::Node,
            );
        }
        // Rebuild the `by_name` index after deserialization paths;
        // here it's already up to date.
        t
    }
}

/// Typed field reference.
///
/// Replaces the `Option<String>` overload of [`PropertyRef`]:
/// * [`Field::Entity`] — the alias itself (e.g. `p` in `RETURN p`),
/// * [`Field::Property`] — one of its properties (e.g. `p.age`).
///
/// The two variants are *the* two meanings; there is no third state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Field {
    Entity(AliasId),
    Property(AliasId, Box<str>),
}

impl Field {
    pub fn alias(&self) -> AliasId {
        match self {
            Field::Entity(a) | Field::Property(a, _) => *a,
        }
    }

    pub fn property(&self) -> Option<&str> {
        match self {
            Field::Entity(_) => None,
            Field::Property(_, p) => Some(p),
        }
    }

    /// Construct a [`Field`] from a stringly-typed [`PropertyRef`]
    /// against a binding table. Returns `None` when the alias is not
    /// bound — typically a resolver bug, hence the panic-free shape so
    /// callers can decide what to do.
    pub fn from_property_ref(p: &PropertyRef, table: &BindingTable) -> Option<Self> {
        let id = table.lookup(p.alias.as_str())?;
        Some(match &p.property {
            None => Field::Entity(id),
            Some(prop) => Field::Property(id, prop.as_str().into()),
        })
    }

    /// Inverse of [`Self::from_property_ref`]. Always succeeds; reads
    /// the name out of the binding table.
    pub fn to_property_ref(&self, table: &BindingTable) -> PropertyRef {
        let b = table.get(self.alias());
        PropertyRef {
            alias: Alias::new(b.name.to_string()),
            property: self.property().map(|p| p.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::query::{
        Action, Alias, Depth, Direction, EdgeTraversal, Node, ReadQuery, ReturnClause,
    };

    fn alias(s: &str) -> Alias {
        Alias::new(s)
    }

    fn rq() -> ReadQuery {
        ReadQuery {
            action: Action::Find,
            start: Node {
                label: "Person".into(),
                alias: alias("p"),
                prefix_label: None,
            },
            traversals: vec![EdgeTraversal {
                from_alias: alias("p"),
                edge_label: "KNOWS".into(),
                edge_alias: alias("r"),
                direction: Direction::Out,
                target: Node {
                    label: "Person".into(),
                    alias: alias("p2"),
                    prefix_label: None,
                },
                depth: Some(Depth { min: 1, max: 1 }),
                optional: false,
            }],
            filter: None,
            returns: vec![ReturnClause::Field {
                field: PropertyRef {
                    alias: alias("p"),
                    property: Some("name".into()),
                },
                alias: None,
            }],
            group_by: vec![],
            sort: vec![],
            limit: None,
        }
    }

    #[test]
    fn from_read_query_intern_order_matches_traversal_order() {
        let q = rq();
        let t = BindingTable::from_read_query(&q);
        let names: Vec<&str> = t.iter().map(|(_, b)| &*b.name).collect();
        assert_eq!(names, vec!["p", "r", "p2"]);
        assert_eq!(t.get(t.lookup("p").unwrap()).kind, BindingKind::Node);
        assert_eq!(t.get(t.lookup("r").unwrap()).kind, BindingKind::Edge);
        assert_eq!(t.get(t.lookup("p2").unwrap()).kind, BindingKind::Node);
        assert_eq!(&*t.get(t.lookup("p2").unwrap()).label, "Person");
    }

    #[test]
    fn field_roundtrips_through_property_ref() {
        let q = rq();
        let t = BindingTable::from_read_query(&q);

        let pr = PropertyRef {
            alias: alias("p"),
            property: Some("name".into()),
        };
        let f = Field::from_property_ref(&pr, &t).unwrap();
        assert!(matches!(f, Field::Property(_, ref p) if &**p == "name"));
        let back = f.to_property_ref(&t);
        assert_eq!(back, pr);

        let pr_entity = PropertyRef {
            alias: alias("r"),
            property: None,
        };
        let f = Field::from_property_ref(&pr_entity, &t).unwrap();
        assert!(matches!(f, Field::Entity(_)));
        assert_eq!(f.to_property_ref(&t), pr_entity);
    }

    #[test]
    fn unknown_alias_yields_none() {
        let t = BindingTable::from_read_query(&rq());
        let pr = PropertyRef {
            alias: alias("nope"),
            property: None,
        };
        assert!(Field::from_property_ref(&pr, &t).is_none());
    }
}
