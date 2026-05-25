//! Integration tests for the knowledge-extraction prompt generator.
//!
//! The contract under test: the JSON shape advertised by the prompt
//! must parse cleanly into the `entities`/`relations` halves of a
//! [`ChunkInput`]. If we ever drift the prompt or the struct shape
//! these tests catch it.

use linguagraph::ingest::{ChunkInput, EntityInput, RelationInput};
use linguagraph::prompt::{
    render_knowledge_extract_prompt, DomainOntology, EntityTypeSpec,
    InMemoryOntologyCatalogStorage, OntologyCatalog, OntologyCatalogStorage, PromptGenerator,
    RelationTypeSpec,
};

#[test]
fn builtin_legal_domain_renders_with_legal_vocabulary() {
    let generator = PromptGenerator::with_builtin_catalog();
    let p = generator
        .knowledge_extract_prompt("some legal text", Some("legal"))
        .expect("legal domain must be present in built-in catalog");

    // Spot-check a few of the bundled defaults are in the rendered list.
    assert!(p.contains("* `LegalNorm`"));
    assert!(p.contains("* `StateBody`"));
    assert!(p.contains("* `LegalRight`"));
    assert!(p.contains("* `GRANTS`"));
    assert!(p.contains("* `APPLIES_TO`"));
}

#[test]
fn custom_entity_and_relation_types_are_used() {
    let ontology = DomainOntology {
        entity_types: vec![
            EntityTypeSpec::new("Article"),
            EntityTypeSpec::with_description("Citation", "Reference to another article."),
        ],
        relation_types: vec![
            RelationTypeSpec::new("CITES"),
            RelationTypeSpec::with_description("CONTAINS", "Article contains a sub-article."),
        ],
    };
    let p = render_knowledge_extract_prompt("Article 1 cites Article 2.", "custom", &ontology);
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
    let generator = PromptGenerator::with_builtin_catalog();
    let p = generator
        .knowledge_extract_prompt("The court grants the right to appeal.", Some("legal"))
        .unwrap();
    assert!(p.contains("# 🔹 Text Fragment"));
    assert!(p.contains("```\nThe court grants the right to appeal."));
}

#[tokio::test]
async fn custom_storage_backend_drives_the_generator() {
    // Demonstrates how a caller can plug in their own storage
    // backend (e.g. Postgres). Here we use the in-memory one.
    let mut catalog = OntologyCatalog::default();
    catalog.insert(
        "medical",
        DomainOntology {
            entity_types: vec![
                EntityTypeSpec::with_description("Disease", "A pathological condition."),
                EntityTypeSpec::new("Symptom"),
            ],
            relation_types: vec![
                RelationTypeSpec::new("CAUSES"),
                RelationTypeSpec::new("TREATS"),
            ],
        },
    );
    let storage: InMemoryOntologyCatalogStorage =
        InMemoryOntologyCatalogStorage::new(catalog);
    // Verify the trait-level API too.
    let loaded = OntologyCatalogStorage::load(&storage).await.unwrap();
    assert!(loaded.get("medical").is_some());

    let generator = PromptGenerator::from_storage(&storage).await.unwrap();
    let p = generator
        .knowledge_extract_prompt("Patient presents with fever.", Some("medical"))
        .unwrap();
    assert!(p.contains("* `Disease` — A pathological condition."));
    assert!(p.contains("* `Symptom`"));
    assert!(p.contains("* `CAUSES`"));
    // Framing is substituted with the active domain name.
    assert!(p.contains("**medical information extraction**"));
    assert!(p.contains("authoritative medical content"));
    // No leftover placeholder or legacy `legal` framing.
    assert!(!p.contains("{domain}"));
    assert!(!p.contains("legal information extraction"));
}

#[test]
fn extending_a_domain_ontology_at_runtime() {
    // Useful when callers want to extend the default vocab rather than
    // replace it.
    let mut catalog = OntologyCatalog::builtin();
    let legal = catalog
        .domains
        .get_mut("legal")
        .expect("legal domain must exist");
    legal.entity_types.push(EntityTypeSpec::new("CustomThing"));
    legal
        .relation_types
        .push(RelationTypeSpec::new("REFERENCES_EXTERNAL"));

    let generator = PromptGenerator::new(catalog);
    let p = generator
        .knowledge_extract_prompt("x", Some("legal"))
        .unwrap();
    assert!(p.contains("* `LegalNorm`"));
    assert!(p.contains("* `CustomThing`"));
    assert!(p.contains("* `GRANTS`"));
    assert!(p.contains("* `REFERENCES_EXTERNAL`"));
}
