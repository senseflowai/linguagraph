//! Resolves [`super::TypeHandler`] implementations by [`super::TypeId`].
//!
//! The registry is the single point where stringly-typed type names enter
//! the system. From the moment a handler is resolved, downstream code
//! never sees the name — it works with the trait object.

use std::collections::HashMap;
use std::sync::Arc;

use super::{TypeError, TypeHandler, TypeId};

/// Builder for a [`TypeRegistry`].
///
/// Handlers can be registered in any order; the only constraint is that
/// each `TypeId` is unique. We surface this as a builder rather than a
/// mutable registry so the registry itself can be cloned freely without
/// any concern about partial modifications.
#[derive(Default)]
pub struct RegistryBuilder {
    handlers: HashMap<TypeId, Arc<dyn TypeHandler>>,
}

impl RegistryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<H: TypeHandler + 'static>(mut self, handler: H) -> Self {
        let id = handler.type_id();
        self.handlers.insert(id, Arc::new(handler));
        self
    }

    pub fn register_arc(mut self, handler: Arc<dyn TypeHandler>) -> Self {
        let id = handler.type_id();
        self.handlers.insert(id, handler);
        self
    }

    pub fn build(self) -> TypeRegistry {
        TypeRegistry { handlers: self.handlers }
    }
}

impl std::fmt::Debug for RegistryBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryBuilder")
            .field(
                "handlers",
                &self.handlers.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Immutable, clone-cheap directory of handlers.
#[derive(Clone, Default)]
pub struct TypeRegistry {
    handlers: HashMap<TypeId, Arc<dyn TypeHandler>>,
}

impl TypeRegistry {
    pub fn empty() -> Self {
        Self::default()
    }

    /// Resolve a handler. Returns [`TypeError::UnknownType`] when no
    /// handler is registered under `id`.
    pub fn get(&self, id: &TypeId) -> Result<&Arc<dyn TypeHandler>, TypeError> {
        self.handlers
            .get(id)
            .ok_or_else(|| TypeError::UnknownType(id.0.clone()))
    }

    /// Like [`Self::get`] but takes a string. Convenient at trust
    /// boundaries (DSL parsing) where the type id is still a string.
    pub fn get_by_name(&self, name: &str) -> Result<&Arc<dyn TypeHandler>, TypeError> {
        let id = TypeId::new(name);
        self.handlers
            .get(&id)
            .ok_or_else(|| TypeError::UnknownType(name.to_string()))
    }

    pub fn contains(&self, id: &TypeId) -> bool {
        self.handlers.contains_key(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Arc<dyn TypeHandler>> {
        self.handlers.values()
    }

    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }
}

impl std::fmt::Debug for TypeRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeRegistry")
            .field("types", &self.handlers.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::context::{EmitCtx, IngestCtx, LowerCtx};
    use crate::types::{Capabilities, TypeError, TypeHandler, TypedPredicate};

    #[derive(Debug)]
    struct Stub(&'static str);

    impl TypeHandler for Stub {
        fn type_id(&self) -> TypeId {
            TypeId::new(self.0)
        }
        fn capabilities(&self) -> Capabilities {
            Capabilities::EXACT_MATCH
        }
        fn on_ingest(&self, _: &mut IngestCtx<'_>) -> Result<(), TypeError> {
            Ok(())
        }
        fn lower(&self, _: &mut LowerCtx<'_>) -> Result<TypedPredicate, TypeError> {
            unreachable!()
        }
        fn emit(&self, _: &mut EmitCtx<'_>, _: &TypedPredicate) -> Result<(), TypeError> {
            unreachable!()
        }
    }

    #[test]
    fn round_trip() {
        let reg = RegistryBuilder::new()
            .register(Stub("A"))
            .register(Stub("B"))
            .build();
        assert_eq!(reg.len(), 2);
        assert!(reg.contains(&TypeId::new("A")));
        assert!(reg.get(&TypeId::new("B")).is_ok());
        assert!(matches!(
            reg.get(&TypeId::new("Missing")),
            Err(TypeError::UnknownType(_))
        ));
    }

    #[test]
    fn last_register_wins_for_same_id() {
        // Replace semantics make hot-reloading handlers safe.
        let reg = RegistryBuilder::new()
            .register(Stub("Dup"))
            .register(Stub("Dup"))
            .build();
        assert_eq!(reg.len(), 1);
    }
}
