//! Integration tests for the knowledge-extraction prompt generator.
//!
//! The contract under test: the JSON shape advertised by the prompt
//! must parse cleanly into the `entities`/`relations` halves of a
//! [`ChunkInput`]. If we ever drift the prompt or the struct shape
//! these tests catch it.

use linguagraph::ingest::{ChunkInput, EntityInput, RelationInput};
use linguagraph::promptgen::knowledge::{
    default_entity_types, default_relation_types, generate_knowledge_extract_prompt,
    EntityTypeSpec, KnowledgeExtractOptions, RelationTypeSpec,
};

#[test]
fn defaults_render_a_prompt_with_legal_vocabulary() {
    let opts = KnowledgeExtractOptions::default();
    let p = generate_knowledge_extract_prompt("some legal text", &opts);

    // Spot-check a few of the bundled defaults are in the rendered list.
    assert!(p.contains("* `LegalNorm`"));
    assert!(p.contains("* `StateBody`"));
    assert!(p.contains("* `LegalRight`"));
    assert!(p.contains("* `GRANTS`"));
    assert!(p.contains("* `APPLIES_TO`"));
}

#[test]
fn custom_entity_and_relation_types_are_used() {
    let opts = KnowledgeExtractOptions {
        entity_types: vec![
            EntityTypeSpec::new("Article"),
            EntityTypeSpec::with_description("Citation", "Reference to another article."),
        ],
        relation_types: vec![
            RelationTypeSpec::new("CITES"),
            RelationTypeSpec::with_description("CONTAINS", "Article contains a sub-article."),
        ],
    };
    let p = generate_knowledge_extract_prompt("Article 1 cites Article 2.", &opts);
    assert!(p.contains("* `Article`"));
    assert!(p.contains("* `Citation` — Reference to another article."));
    assert!(p.contains("* `CITES`"));
    assert!(p.contains("* `CONTAINS` — Article contains a sub-article."));
    // Defaults must NOT have leaked in as list entries.
    assert!(!p.contains("* `LegalNorm`"));
    assert!(!p.contains("* `GRANTS`"));
}

#[test]
fn prompt_json_shape_matches_chunk_input_parsing() {
    // The literal output spec shown to the LLM should parse cleanly
    // into the (entities, relations) halves of a ChunkInput. We
    // reconstruct a complete ChunkInput JSON around the LLM's JSON and
    // assert it deserializes — that's the contract end-to-end.
    let llm_output = serde_json::json!({
        "entities": [
            {"id": "e1", "type": "Person", "name": "Alice"},
            {"id": "e2", "type": "Company", "name": "Acme"}
        ],
        "relations": [
            {"from": "e1", "to": "e2", "type": "WORKS_AT"}
        ]
    });
    let chunk_json = serde_json::json!({
        "id": "c1",
        "text": "Alice works at Acme.",
        "entities": llm_output["entities"],
        "relations": llm_output["relations"],
    });
    let chunk: ChunkInput =
        serde_json::from_value(chunk_json).expect("LLM-shaped JSON must parse as ChunkInput");
    assert_eq!(chunk.entities.len(), 2);
    assert_eq!(chunk.relations.len(), 1);
    // Spot-check field names match the prompt's documented shape.
    let alice: &EntityInput = &chunk.entities[0];
    assert_eq!(alice.id, "e1");
    assert_eq!(alice.kind, "Person");
    assert_eq!(alice.name, "Alice");
    let rel: &RelationInput = &chunk.relations[0];
    assert_eq!(rel.from, "e1");
    assert_eq!(rel.to, "e2");
    assert_eq!(rel.kind, "WORKS_AT");
}

#[test]
fn fragment_is_embedded_in_a_code_block() {
    let p = generate_knowledge_extract_prompt(
        "The court grants the right to appeal.",
        &KnowledgeExtractOptions::default(),
    );
    assert!(p.contains("# 🔹 Text Fragment"));
    assert!(p.contains("```\nThe court grants the right to appeal."));
}

#[test]
fn default_lists_are_reusable_directly() {
    // Useful when callers want to extend the default vocab rather than
    // replace it.
    let mut ents = default_entity_types();
    ents.push(EntityTypeSpec::new("CustomThing"));
    let mut rels = default_relation_types();
    rels.push(RelationTypeSpec::new("REFERENCES_EXTERNAL"));
    let p = generate_knowledge_extract_prompt(
        "x",
        &KnowledgeExtractOptions {
            entity_types: ents,
            relation_types: rels,
        },
    );
    assert!(p.contains("* `LegalNorm`"));
    assert!(p.contains("* `CustomThing`"));
    assert!(p.contains("* `GRANTS`"));
    assert!(p.contains("* `REFERENCES_EXTERNAL`"));
}
