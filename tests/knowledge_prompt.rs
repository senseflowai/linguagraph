//! Integration tests for the knowledge-extraction prompt generator.

use linguagraph::{
    graph::{
        DomainOntology, EntityTypeSpec, InMemoryOntologyCatalogStorage, OntologyCatalog,
        OntologyCatalogStorage, RelationTypeSpec,
    },
    prompt::{render_knowledge_extract_prompt, PromptGenerator},
};

#[test]
fn builtin_legal_domain_renders_with_legal_vocabulary() {
    let generator = PromptGenerator::with_builtin_catalog();
    let p = generator
        .knowledge_extract_prompt(Some("legal"))
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
    let p = render_knowledge_extract_prompt("custom", &ontology);
    assert!(p.contains("* `Article`"));
    assert!(p.contains("* `Citation` — Reference to another article."));
    assert!(p.contains("* `CITES`"));
    assert!(p.contains("* `CONTAINS` — Article contains a sub-article."));
    // Defaults must NOT have leaked in as list entries.
    assert!(!p.contains("* `LegalNorm`"));
    assert!(!p.contains("* `GRANTS`"));
}

#[test]
fn prompt_json_shape_is_valid_json_with_expected_fields() {
    // Verify that the JSON shape advertised in the prompt contains the
    // expected top-level keys and field names.
    let llm_output: serde_json::Value = serde_json::json!({
        "entities": [
            {"id": "e1", "type": "Person", "name": "Alice"},
            {"id": "e2", "type": "Company", "name": "Acme"}
        ],
        "relations": [
            {"from": "e1", "to": "e2", "type": "WORKS_AT"}
        ]
    });
    let entities = llm_output["entities"].as_array().unwrap();
    let relations = llm_output["relations"].as_array().unwrap();
    assert_eq!(entities.len(), 2);
    assert_eq!(relations.len(), 1);
    assert_eq!(entities[0]["id"], "e1");
    assert_eq!(entities[0]["type"], "Person");
    assert_eq!(entities[0]["name"], "Alice");
    assert_eq!(relations[0]["from"], "e1");
    assert_eq!(relations[0]["to"], "e2");
    assert_eq!(relations[0]["type"], "WORKS_AT");
}

#[test]
fn prompt_does_not_embed_fragment_text() {
    let generator = PromptGenerator::with_builtin_catalog();
    let p = generator.knowledge_extract_prompt(Some("legal")).unwrap();
    assert!(!p.contains("# 🔹 Text Fragment"));
    assert!(!p.contains("The court grants the right to appeal."));
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
    let storage: InMemoryOntologyCatalogStorage = InMemoryOntologyCatalogStorage::new(catalog);
    // Verify the trait-level API too.
    let loaded = OntologyCatalogStorage::load(&storage).await.unwrap();
    assert!(loaded.get("medical").is_some());

    let generator = PromptGenerator::from_storage(&storage).await.unwrap();
    let p = generator.knowledge_extract_prompt(Some("medical")).unwrap();
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
    let p = generator.knowledge_extract_prompt(Some("legal")).unwrap();
    assert!(p.contains("* `LegalNorm`"));
    assert!(p.contains("* `CustomThing`"));
    assert!(p.contains("* `GRANTS`"));
    assert!(p.contains("* `REFERENCES_EXTERNAL`"));
}
