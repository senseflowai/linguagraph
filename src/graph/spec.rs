use crate::embeddings::{EmbedError, Embedder, Reranker};
use crate::graph::PropertyType;
use crate::graph::{
    FileGraphSpecificationStorage, GraphSpecificationStorage, GraphSpecificationStorageError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SpecRecord {
    Entity(EntitySpecRecord),
    Property(PropertySpecRecord),
}

impl SpecRecord {
    pub fn key(&self) -> String {
        match self {
            SpecRecord::Entity(record) => record.key(),
            SpecRecord::Property(record) => record.key(),
        }
    }

    pub fn entity_name(&self) -> &str {
        match self {
            SpecRecord::Entity(record) => &record.name,
            SpecRecord::Property(record) => &record.entity,
        }
    }

    pub fn as_entity(&self) -> Option<&EntitySpecRecord> {
        match self {
            SpecRecord::Entity(record) => Some(record),
            SpecRecord::Property(_) => None,
        }
    }

    pub fn as_property(&self) -> Option<&PropertySpecRecord> {
        match self {
            SpecRecord::Entity(_) => None,
            SpecRecord::Property(record) => Some(record),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntitySpecRecord {
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl EntitySpecRecord {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            embedding: None,
        }
    }

    pub fn key(&self) -> String {
        self.name.clone()
    }

    pub fn embedding(&self) -> Option<&[f32]> {
        self.embedding.as_deref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntitySpecMatch<'a> {
    pub record: &'a EntitySpecRecord,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertySpecRecord {
    pub entity: String,
    pub name: String,
    pub r#type: PropertyType,
    pub description: String,
}

impl PropertySpecRecord {
    pub fn new(
        entity: impl Into<String>,
        name: impl Into<String>,
        r#type: PropertyType,
        description: impl Into<String>,
    ) -> Self {
        Self {
            entity: entity.into(),
            name: name.into(),
            r#type,
            description: description.into(),
        }
    }

    pub fn key(&self) -> String {
        property_key(&self.entity, &self.name)
    }

    pub fn type_id(&self) -> &'static str {
        match self.r#type {
            PropertyType::String => "Text",
            PropertyType::Text => "SemanticText",
            PropertyType::Number => "Number",
            PropertyType::Boolean => "Boolean",
            PropertyType::DateTime | PropertyType::Timestamp => "Timestamp",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GraphSpecification {
    records: HashMap<String, SpecRecord>,
}

impl GraphSpecification {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn load() -> Result<Self, GraphSpecificationStorageError> {
        FileGraphSpecificationStorage::default().load().await
    }

    pub async fn save_to_file(&self) -> Result<(), GraphSpecificationStorageError> {
        FileGraphSpecificationStorage::default().save(self).await
    }

    pub fn with_record(mut self, record: impl Into<SpecRecord>) -> Self {
        self.insert(record);
        self
    }

    pub fn with_entity(self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.with_record(EntitySpecRecord::new(name, description))
    }

    pub fn with_property(
        self,
        entity: impl Into<String>,
        name: impl Into<String>,
        r#type: PropertyType,
        description: impl Into<String>,
    ) -> Self {
        self.with_record(PropertySpecRecord::new(entity, name, r#type, description))
    }

    pub fn insert(&mut self, record: impl Into<SpecRecord>) -> Option<SpecRecord> {
        let record = record.into();
        self.records.insert(record.key(), record)
    }

    pub fn compute(&mut self, embedder: &dyn Embedder) -> Result<(), EmbedError> {
        let mut jobs: Vec<(String, String)> = self
            .records
            .values()
            .filter_map(SpecRecord::as_entity)
            .filter(|record| record.embedding().is_none())
            .map(|record| (record.key(), self.entity_embedding_text(record)))
            .collect();
        jobs.sort_by(|a, b| a.0.cmp(&b.0));

        if jobs.is_empty() {
            return Ok(());
        }

        let texts: Vec<&str> = jobs.iter().map(|(_, text)| text.as_str()).collect();
        let embeddings = embedder.embed_batch(&texts)?;
        if embeddings.len() != jobs.len() {
            return Err(EmbedError::Backend(format!(
                "embedder returned {} vectors for {} entity specs",
                embeddings.len(),
                jobs.len()
            )));
        }

        for ((key, _), embedding) in jobs.into_iter().zip(embeddings.into_iter()) {
            if let Some(SpecRecord::Entity(record)) = self.records.get_mut(&key) {
                record.embedding = Some(embedding);
            }
        }
        Ok(())
    }

    pub fn find(
        &self,
        text: impl AsRef<str>,
        threshold: f32,
        embedder: &dyn Embedder,
        reranker: Option<&dyn Reranker>,
        reranking_threshold: f64,
    ) -> Result<Vec<EntitySpecMatch<'_>>, EmbedError> {
        let text = format!(
            "User query:{}\nTask: Identify database schema elements need for answering this query",
            text.as_ref()
        );
        let query = embedder.embed(text.as_str())?;
        let mut matches = Vec::new();

        for record in self.records.values().filter_map(SpecRecord::as_entity) {
            let Some(embedding) = record.embedding() else {
                continue;
            };
            if embedding.len() != query.len() {
                return Err(EmbedError::Backend(format!(
                    "embedding dimension mismatch for entity '{}': entity vector has {}, query vector has {}",
                    record.name,
                    embedding.len(),
                    query.len()
                )));
            }
            let score = cosine_similarity(&query, embedding);
            if score >= threshold {
                matches.push(EntitySpecMatch { record, score });
            }
        }

        if let Some(reranker) = reranker {
            if !matches.is_empty() {
                let documents: Vec<String> = matches
                    .iter()
                    .map(|m| self.entity_embedding_text(m.record))
                    .collect();
                let scores = reranker.rerank(text.as_str(), &documents)?;
                if scores.len() != matches.len() {
                    return Err(EmbedError::Backend(format!(
                        "reranker returned {} scores for {} entity specs",
                        scores.len(),
                        matches.len()
                    )));
                }
                matches = matches
                    .into_iter()
                    .zip(scores.into_iter())
                    .filter_map(|(mut m, score)| {
                        if score >= reranking_threshold {
                            m.score = score as f32;
                            Some(m)
                        } else {
                            None
                        }
                    })
                    .collect();
            }
        }

        matches.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.record.name.cmp(&b.record.name))
        });
        Ok(matches)
    }

    pub fn merge(&mut self, other: &GraphSpecification) {
        for record in other.records.values() {
            self.insert(record.clone());
        }
    }

    pub fn merged(mut self, other: &GraphSpecification) -> Self {
        self.merge(other);
        self
    }

    pub fn add_entity(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Option<SpecRecord> {
        self.insert(EntitySpecRecord::new(name, description))
    }

    pub fn add_property(
        &mut self,
        entity: impl Into<String>,
        name: impl Into<String>,
        r#type: PropertyType,
        description: impl Into<String>,
    ) -> Option<SpecRecord> {
        self.insert(PropertySpecRecord::new(entity, name, r#type, description))
    }

    pub fn get(&self, key: impl AsRef<str>) -> Option<&SpecRecord> {
        self.records.get(key.as_ref())
    }

    pub fn get_entity(&self, entity: impl AsRef<str>) -> Option<&EntitySpecRecord> {
        self.get(entity.as_ref()).and_then(SpecRecord::as_entity)
    }

    pub fn get_property(
        &self,
        entity: impl AsRef<str>,
        property: impl AsRef<str>,
    ) -> Option<&PropertySpecRecord> {
        self.get(property_key(entity.as_ref(), property.as_ref()))
            .and_then(SpecRecord::as_property)
    }

    pub fn get_type(&self, entity: impl AsRef<str>, property: impl AsRef<str>) -> Option<&str> {
        self.get_property(entity, property)
            .map(PropertySpecRecord::type_id)
    }

    pub fn get_query_type(
        &self,
        entity: impl AsRef<str>,
        property: impl AsRef<str>,
    ) -> Option<&str> {
        let property = self.get_property(entity, property)?;
        match property.r#type {
            PropertyType::String => Some(property.type_id()),
            PropertyType::Text => Some(property.type_id()),
            PropertyType::Number
            | PropertyType::Boolean
            | PropertyType::DateTime
            | PropertyType::Timestamp => None,
        }
    }

    pub fn records_for_entity(&self, entity: impl AsRef<str>) -> Vec<&SpecRecord> {
        let entity = entity.as_ref();
        self.records
            .values()
            .filter(|record| record.entity_name() == entity)
            .collect()
    }

    pub fn properties_for_entity(&self, entity: impl AsRef<str>) -> Vec<&PropertySpecRecord> {
        let entity = entity.as_ref();
        self.records
            .values()
            .filter_map(SpecRecord::as_property)
            .filter(|record| record.entity == entity)
            .collect()
    }

    pub fn get_properties(&self, entity: String) -> Vec<&PropertySpecRecord> {
        self.properties_for_entity(entity)
    }

    pub fn contains_key(&self, key: impl AsRef<str>) -> bool {
        self.records.contains_key(key.as_ref())
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn records(&self) -> &HashMap<String, SpecRecord> {
        &self.records
    }

    fn entity_embedding_text(&self, entity: &EntitySpecRecord) -> String {
        let mut properties = self.properties_for_entity(&entity.name);
        properties.sort_by(|a, b| a.name.cmp(&b.name));

        let mut out = format!("{} - {}\n", entity.name, entity.description);
        if !properties.is_empty() {
            out.push_str("Properties:");
            for property in properties {
                out.push_str(&format!(
                    "\n- {} ({:?}): {}",
                    property.name, property.r#type, property.description
                ));
            }
        }
        out
    }
}

impl From<EntitySpecRecord> for SpecRecord {
    fn from(record: EntitySpecRecord) -> Self {
        SpecRecord::Entity(record)
    }
}

impl From<PropertySpecRecord> for SpecRecord {
    fn from(record: PropertySpecRecord) -> Self {
        SpecRecord::Property(record)
    }
}

fn property_key(entity: &str, property: &str) -> String {
    format!("{entity}.{property}")
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        a_norm += x * x;
        b_norm += y * y;
    }

    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm.sqrt() * b_norm.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use crate::embeddings::{EmbedError, Embedder, MockEmbedder};

    use super::*;

    #[test]
    fn adds_records_with_expected_keys() {
        let spec = GraphSpecification::new()
            .with_entity("Company", "A legal organization.")
            .with_property(
                "Company",
                "name",
                PropertyType::Text,
                "Human-readable company name.",
            );

        assert_eq!(spec.len(), 2);
        assert!(spec.contains_key("Company"));
        assert!(spec.contains_key("Company.name"));
    }

    #[test]
    fn retrieves_by_raw_key_and_typed_accessors() {
        let mut spec = GraphSpecification::new();
        spec.add_entity("Person", "A human.");
        spec.add_property("Person", "age", PropertyType::Number, "Age in years.");

        let raw = spec.get("Person.age").unwrap();
        assert_eq!(raw.as_property().unwrap().r#type, PropertyType::Number);

        let entity = spec.get_entity("Person").unwrap();
        assert_eq!(entity.description, "A human.");

        let property = spec.get_property("Person", "age").unwrap();
        assert_eq!(property.description, "Age in years.");
    }

    #[test]
    fn retrieves_all_records_for_entity() {
        let spec = GraphSpecification::new()
            .with_entity("Person", "A human.")
            .with_property("Person", "name", PropertyType::Text, "Display name.")
            .with_property("Person", "age", PropertyType::Number, "Age in years.")
            .with_entity("Company", "A legal organization.");

        let person_records = spec.records_for_entity("Person");
        assert_eq!(person_records.len(), 3);

        let person_properties = spec.properties_for_entity("Person");
        assert_eq!(person_properties.len(), 2);
        assert!(person_properties.iter().any(|record| record.name == "name"));
        assert!(person_properties.iter().any(|record| record.name == "age"));

        let via_owned_entity_name = spec.get_properties("Person".to_string());
        assert_eq!(via_owned_entity_name.len(), 2);
    }

    #[test]
    fn inserting_same_key_replaces_record() {
        let mut spec = GraphSpecification::new();
        assert!(spec.add_entity("Person", "Old description.").is_none());

        let old = spec.add_entity("Person", "New description.").unwrap();
        assert_eq!(old.as_entity().unwrap().description, "Old description.");
        assert_eq!(
            spec.get_entity("Person").unwrap().description,
            "New description."
        );
    }

    #[test]
    fn merge_adds_new_records_and_replaces_existing_keys() {
        let mut base = GraphSpecification::new()
            .with_entity("Person", "Old person.")
            .with_property("Person", "name", PropertyType::Text, "Old name.")
            .with_entity("Company", "Company.");
        let incoming = GraphSpecification::new()
            .with_entity("Person", "New person.")
            .with_property("Person", "age", PropertyType::Number, "Age.")
            .with_property("Company", "name", PropertyType::Text, "Company name.");

        base.merge(&incoming);

        assert_eq!(base.len(), 5);
        assert_eq!(
            base.get_entity("Person").unwrap().description,
            "New person."
        );
        assert_eq!(
            base.get_property("Person", "name").unwrap().description,
            "Old name."
        );
        assert_eq!(
            base.get_property("Person", "age").unwrap().description,
            "Age."
        );
        assert_eq!(
            base.get_property("Company", "name").unwrap().description,
            "Company name."
        );
    }

    #[test]
    fn merged_returns_combined_spec_without_mutating_inputs() {
        let base = GraphSpecification::new().with_entity("Person", "Base.");
        let incoming = GraphSpecification::new().with_entity("Company", "Incoming.");

        let combined = base.clone().merged(&incoming);

        assert!(base.get_entity("Company").is_none());
        assert!(incoming.get_entity("Person").is_none());
        assert!(combined.get_entity("Person").is_some());
        assert!(combined.get_entity("Company").is_some());
    }

    #[test]
    fn compute_embeds_entity_specs() {
        let mut spec = GraphSpecification::new()
            .with_entity("Person", "A human being.")
            .with_property("Person", "name", PropertyType::Text, "Display name.")
            .with_entity("Company", "A legal organization.");
        let embedder = MockEmbedder::new(8);

        spec.compute(&embedder).unwrap();

        assert_eq!(
            spec.get_entity("Person")
                .unwrap()
                .embedding()
                .unwrap()
                .len(),
            8
        );
        assert_eq!(
            spec.get_entity("Company")
                .unwrap()
                .embedding()
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn compute_uses_entity_and_property_text() {
        #[derive(Debug, Default)]
        struct RecordingEmbedder {
            texts: Mutex<Vec<String>>,
        }

        impl Embedder for RecordingEmbedder {
            fn dim(&self) -> usize {
                2
            }

            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
                self.texts
                    .lock()
                    .unwrap()
                    .extend(texts.iter().map(|text| (*text).to_string()));
                Ok(texts.iter().map(|_| vec![0.0, 1.0]).collect())
            }
        }

        let mut spec = GraphSpecification::new()
            .with_entity("Person", "A human being.")
            .with_property("Person", "name", PropertyType::Text, "Display name.")
            .with_property("Person", "age", PropertyType::Number, "Age in years.");
        let embedder = RecordingEmbedder::default();

        spec.compute(&embedder).unwrap();

        let texts = embedder.texts.lock().unwrap();
        assert_eq!(texts.len(), 1);
        assert!(texts[0].contains("Person - A human being."));
        assert!(texts[0].contains("age (Number): Age in years."));
        assert!(texts[0].contains("name (Text): Display name."));
    }

    #[test]
    fn find_returns_entities_above_threshold_by_cosine_similarity() {
        #[derive(Debug)]
        struct KeywordEmbedder;

        impl Embedder for KeywordEmbedder {
            fn dim(&self) -> usize {
                2
            }

            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
                Ok(texts
                    .iter()
                    .map(|text| {
                        let lower = text.to_ascii_lowercase();
                        if lower.contains("company")
                            || lower.contains("organization")
                            || lower.contains("business")
                        {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect())
            }
        }

        let embedder = KeywordEmbedder;
        let mut spec = GraphSpecification::new()
            .with_entity("Person", "A human being.")
            .with_property("Person", "name", PropertyType::Text, "Display name.")
            .with_entity("Company", "A legal organization.")
            .with_property("Company", "name", PropertyType::Text, "Company name.");
        spec.compute(&embedder).unwrap();

        let matches = spec
            .find("business organization", 0.75, &embedder, None, 0.0)
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record.name, "Company");
        assert_eq!(matches[0].score, 1.0);
    }

    #[test]
    fn find_filters_embedding_matches_by_reranker_threshold() {
        #[derive(Debug)]
        struct KeywordEmbedder;

        impl Embedder for KeywordEmbedder {
            fn dim(&self) -> usize {
                2
            }

            fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
                Ok(texts
                    .iter()
                    .map(|text| {
                        let lower = text.to_ascii_lowercase();
                        if lower.contains("camera") {
                            vec![1.0, 0.0]
                        } else {
                            vec![0.0, 1.0]
                        }
                    })
                    .collect())
            }
        }

        #[derive(Debug)]
        struct PreferCameraReranker;

        impl Reranker for PreferCameraReranker {
            fn rerank(&self, _: &str, documents: &[String]) -> Result<Vec<f64>, EmbedError> {
                Ok(documents
                    .iter()
                    .map(|document| {
                        if document.contains("Camera -") {
                            0.9
                        } else {
                            0.2
                        }
                    })
                    .collect())
            }
        }

        let embedder = KeywordEmbedder;
        let reranker = PreferCameraReranker;
        let mut spec = GraphSpecification::new()
            .with_entity("Camera", "A camera")
            .with_entity("CameraArchive", "Old camera records");
        spec.compute(&embedder).unwrap();

        let matches = spec
            .find("camera", 0.0, &embedder, Some(&reranker), 0.8)
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record.name, "Camera");
        assert_eq!(matches[0].score, 0.9);
    }

    #[test]
    fn find_ignores_entities_without_computed_embeddings() {
        let spec = GraphSpecification::new()
            .with_entity("Person", "A human being.")
            .with_entity("Company", "A legal organization.");
        let embedder = MockEmbedder::new(8);

        let matches = spec.find("human", 0.0, &embedder, None, 0.0).unwrap();

        assert!(matches.is_empty());
    }
}
