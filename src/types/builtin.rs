//! The canonical vocabulary of built-in type ids.
//!
//! [`BuiltinType`] is the **single source of truth** for the string ids
//! the [`TypeRegistry`](super::TypeRegistry) keys handlers by. Every
//! built-in handler registers under one of these ids, and every place
//! that needs to name a built-in type id (ontology mapping, prompt
//! advertisement, downstream normalization) derives the string from here
//! rather than hard-coding a literal.
//!
//! Adding a built-in type is a one-line change: add a variant here, then
//! register a handler for `BuiltinType::X.type_id()`.
//!
//! The variant names *are* the ids verbatim (PascalCase), so `strum`
//! gives us `as`/`from`-string for free with no `match` to maintain.

use strum::{Display, EnumIter, EnumString, IntoStaticStr, VariantNames};

use super::TypeId;

/// The built-in field-type ids known to every registry produced by
/// [`super::handlers::register_core`]. See the [module docs](self).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Display, EnumString, IntoStaticStr, VariantNames,
    EnumIter, serde::Serialize, serde::Deserialize,
)]
pub enum BuiltinType {
    /// Plain, normalized text (exact / contains matching).
    Text,
    /// Integer or float.
    Number,
    /// Boolean.
    Boolean,
    /// Calendar date.
    Date,
    /// Instant (date + time).
    Timestamp,
    /// Embedded, vector-searchable free text.
    SemanticText,
}

impl BuiltinType {
    /// The id as a static string (e.g. `"SemanticText"`).
    pub fn id(self) -> &'static str {
        self.into()
    }

    /// The id as a [`TypeId`] suitable for registry lookup / registration.
    pub fn type_id(self) -> TypeId {
        TypeId::new(self.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::{IntoEnumIterator, VariantNames};

    #[test]
    fn id_matches_variant_name() {
        assert_eq!(BuiltinType::SemanticText.id(), "SemanticText");
        assert_eq!(BuiltinType::Text.id(), "Text");
        assert_eq!(BuiltinType::Number.id(), "Number");
    }

    #[test]
    fn from_str_round_trips_every_variant() {
        for variant in BuiltinType::iter() {
            assert_eq!(variant.id().parse::<BuiltinType>().unwrap(), variant);
        }
    }

    #[test]
    fn variants_lists_every_id() {
        assert_eq!(BuiltinType::VARIANTS.len(), BuiltinType::iter().count());
    }
}
