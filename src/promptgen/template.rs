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
      "to": "EntityB",
      // Optional foreign-key join. REQUIRED when EntityA and EntityB come
      // from separate top-level arrays and are linked by an id value.
      "from_key": "$.entityA[*].b_id",   // JSONPath of the FK on EntityA
      "to_key": "$.entityB[*].id"        // JSONPath of the key on EntityB
                                         // (defaults to EntityB's primary_key)
    }
  ]
}
```
"#;

/// Default catalogue of field types when no live registry is available.
/// Matches what linguagraph ships out of the box.
pub const DEFAULT_TYPES_CATALOGUE: &str = r#"# Available field types

There are exactly two textual types — pick one:

- **Keyword** — a plain string matched by standard Cypher operators
  (`=`, `!=`, `<`, `>`, `=~` regex, `CONTAINS`, …). Use it for identifiers,
  codes, statuses, and short categorical / enum-like labels — anything you
  would match exactly or filter on. Stored verbatim; never embedded.
- **Text** — free-form, natural-language text (names, descriptions,
  summaries, reviews, notes). Stored on the node and embedded for vector /
  hybrid semantic search. The default for any text that isn't a Keyword.
- **Number** — integers, floats.
- **Boolean** — true/false.
- **DateTime** — ISO-8601 dates / timestamps.
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
5. **Use `Text`** for any natural-language / free-form field (names,
   descriptions, summaries, reviews). It is embedded for semantic search.
   `Text` is the default for textual fields that aren't keywords.
6. **Use `Keyword`** for identifiers, codes, statuses, and short
   categorical / enum-like values (status, industry, role) — anything you
   would match exactly, filter, or compare. Keywords are stored verbatim
   and never embedded.
7. **PascalCase, singular** for entity `type`; **SCREAMING_SNAKE_CASE**
   for relationship `type`.
8. **Don't duplicate entities.** If the same array path appears twice,
   merge the field set into a single entity.
9. **Normalize names**: drop hyphens/underscores, capitalise initials,
   pluralise to singular.
10. **Foreign-key relationships.** Scan every entity for foreign-key
    fields — any field ending in `_id`/`_ref`, **including ones nested in
    sub-objects** (e.g. `events[*].origin.camera_id`). When such a field
    references another entity's id, you MUST emit a relationship with
    `from_key` (the full JSONPath of the foreign key, e.g.
    `$.events[*].origin.camera_id`) and `to_key` (the JSONPath of the
    referenced entity's `primary_key`, e.g. `$.cameras[*].id`). Without
    these keys the relationship cannot link the correct objects. Only a
    nested child array physically inside its parent is linked positionally
    and may omit `from_key`/`to_key`.
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
        { "name": "name",        "type": "Text",     "source_path": "$.companies[*].name" },
        { "name": "description", "type": "Text",     "source_path": "$.companies[*].description" },
        { "name": "industry",    "type": "Keyword",  "source_path": "$.companies[*].industry" },
        { "name": "founded_at",  "type": "DateTime", "source_path": "$.companies[*].founded_at" }
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
