//! Static prompt sections.
//!
//! Each section is a `&'static str` literal. The builder concatenates
//! them in a fixed order with the analyser's summary spliced between.
//! Keeping them as bare strings (not a Tera/Handlebars template) means
//! the generated prompt is byte-for-byte deterministic and the golden
//! tests stay simple.

/// Lead-in that frames the LLM's job.
pub const PREAMBLE: &str = "You are a senior data engineer. Given a JSON document, \
emit a *linguagraph mapping* JSON that describes how to ingest the document into a \
property graph. Output **only** the JSON mapping — no prose, no Markdown fences, no \
trailing commentary.";

/// Mapping schema specification.
pub const MAPPING_SPEC: &str = r#"# Mapping schema

Output a single JSON object of this exact shape:

```
{
  "entities": [
    {
      "type": "EntityName",          // PascalCase, singular
      "source_path": "$.path[*]",    // JSONPath that yields one row per element
      "primary_key": "$.path[*].id", // JSONPath of the unique identifier
      "properties": [
        {
          "name": "field_name",
          "type": "<one of the available field types>",
          "source_path": "$.path[*].field"
        }
      ]
    }
  ],
  "relationships": [
    {
      "type": "REL_NAME",            // SCREAMING_SNAKE_CASE
      "from": "EntityA",
      "to": "EntityB"
    }
  ]
}
```
"#;

/// Default catalogue of field types when no live registry is available.
/// Matches what linguagraph ships out of the box.
pub const DEFAULT_TYPES_CATALOGUE: &str = r#"# Available field types

- **SemanticText** — long natural-language fields (descriptions, bios,
  reviews, free-form notes). Stored on the node and additionally
  embedded for vector / hybrid search.
- **Keyword** — short categorical / enum-like values (status,
  industry, role). Use when the value-set is small and bounded.
- **DateTime** — ISO-8601 timestamps.
- **Number** — integers, floats.
- **Boolean** — true/false.
- **Text** — short free-form strings that are not categorical and
  not long enough to warrant SemanticText. The default for ambiguous
  fields.
"#;

/// Strict output rules. Repeated towards the end so a long prompt
/// doesn't bury them.
pub const RULES: &str = r#"# Rules

1. **Output JSON only.** No prose, no comments, no Markdown code fences.
2. **One entity per array of objects** in the input. Singletons go in
   `entities` too, with `source_path` set to the object's JSONPath.
3. **Use `[*]`** in `source_path` when iterating array elements.
4. **Always set `primary_key`** when the input has a unique-looking
   field (`id`, `_id`, `uuid`, `<thing>_id`). Without one, choose the
   field that is most likely unique and document the choice in the
   property's name only — never invent values.
5. **Prefer `SemanticText`** for any field that is natural-language
   prose (descriptions, summaries, reviews). Only use `Text` when the
   field is short *and* clearly not categorical.
6. **Prefer `Keyword`** for short, low-cardinality strings (status,
   industry, role). Don't make Keyword the default for free-form
   strings.
7. **PascalCase, singular** for entity `type`; **SCREAMING_SNAKE_CASE**
   for relationship `type`.
8. **Don't duplicate entities.** If the same array path appears twice,
   merge the field set into a single entity.
9. **Normalize names**: drop hyphens/underscores, capitalise initials,
   pluralise to singular.
"#;

/// One worked example showing a flat document and the expected output.
pub const EXAMPLE: &str = r#"# Example

Input JSON:

```
{
  "companies": [
    {
      "id": 1,
      "name": "Stripe",
      "description": "Payments API for the internet.",
      "industry": "Fintech",
      "founded_at": "2010-09-01"
    }
  ]
}
```

Expected output:

```
{
  "entities": [
    {
      "type": "Company",
      "source_path": "$.companies[*]",
      "primary_key": "$.companies[*].id",
      "properties": [
        { "name": "name",        "type": "Text",         "source_path": "$.companies[*].name" },
        { "name": "description", "type": "SemanticText", "source_path": "$.companies[*].description" },
        { "name": "industry",    "type": "Keyword",      "source_path": "$.companies[*].industry" },
        { "name": "founded_at",  "type": "DateTime",     "source_path": "$.companies[*].founded_at" }
      ]
    }
  ],
  "relationships": []
}
```
"#;

/// Final reminder pinned to the end of the prompt.
pub const TAIL: &str = "Now produce the mapping JSON for the input below. \
Output the JSON mapping only.";
