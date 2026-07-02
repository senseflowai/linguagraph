//! Interactive TTY refinement of a generated [`Mapping`].
//!
//! Walks the user through each entity to confirm or override the
//! `primary_key`, choose which properties to keep (and their field
//! types), and confirm / add / drop relationships. After editing, the
//! result is re-validated and verified against the data.
//!
//! Gated behind the `interactive` feature (pulls in `dialoguer`). The
//! flow is synchronous and blocks on stdin — fine for a CLI invocation.

use dialoguer::{Confirm, Input, MultiSelect, Select};
use serde_json::Value;

use crate::graph::DomainOntology;
use crate::mapper::{self, Mapping, PropertyMapping, RelationshipMapping};
use crate::promptgen::{analyze, InferredType};

use super::collections::CollectionInfo;
use super::{enforce_ontology, MapGenError};

/// Interactively choose which top-level collections to map.
///
/// Presents a checkbox menu (all pre-selected) and returns the chosen
/// collection names. Loops until at least one is selected.
pub fn select_collections(items: &[CollectionInfo]) -> Result<Vec<String>, MapGenError> {
    let labels: Vec<String> = items
        .iter()
        .map(|c| format!("{} ({} item(s))", c.name, c.len))
        .collect();
    let defaults = vec![true; labels.len()];
    loop {
        let picked = MultiSelect::new()
            .with_prompt("Collections to map (space toggles, enter confirms)")
            .items(&labels)
            .defaults(&defaults)
            .interact()
            .map_err(ui_err)?;
        if picked.is_empty() {
            println!("  select at least one collection.");
            continue;
        }
        return Ok(picked.into_iter().map(|i| items[i].name.clone()).collect());
    }
}

/// Field-type vocabulary offered when (re)typing a property.
const FIELD_TYPES: &[&str] = &["Keyword", "Text", "Number", "Boolean", "DateTime", "List"];

fn ui_err(e: dialoguer::Error) -> MapGenError {
    MapGenError::Interactive(e.to_string())
}

/// Interactively refine `mapping` against `data`, keeping it conformant
/// with `ontology`. Returns the edited, re-verified mapping.
pub fn refine_interactively(
    mut mapping: Mapping,
    data: &Value,
    ontology: &DomainOntology,
) -> Result<Mapping, MapGenError> {
    let summary = analyze(data);

    println!("\n=== Interactive mapping refinement ===");
    println!(
        "{} entit(y/ies), {} relationship(s) proposed.\n",
        mapping.entities.len(),
        mapping.relationships.len()
    );

    for ent in &mut mapping.entities {
        println!("── Entity: {} ({})", ent.kind, ent.source_path);

        // Candidate primary-key paths: the analyser's field paths for the
        // matching entity, plus this entity's own property paths.
        let mut candidates: Vec<String> = Vec::new();
        if let Some(es) = summary
            .entities
            .iter()
            .find(|e| e.source_path == ent.source_path)
        {
            candidates.extend(es.fields.iter().map(|f| f.source_path.clone()));
        }
        for p in &ent.properties {
            if !candidates.contains(&p.source_path) {
                candidates.push(p.source_path.clone());
            }
        }

        // primary_key selection.
        let mut items: Vec<String> = vec![format!("Keep current: {}", ent.primary_key)];
        items.extend(candidates.iter().cloned());
        items.push("Enter custom JSONPath…".to_string());
        let choice = Select::new()
            .with_prompt("  primary_key")
            .items(&items)
            .default(0)
            .interact()
            .map_err(ui_err)?;
        if choice == items.len() - 1 {
            let custom: String = Input::new()
                .with_prompt("  custom primary_key JSONPath")
                .with_initial_text(ent.primary_key.clone())
                .interact_text()
                .map_err(ui_err)?;
            if !custom.trim().is_empty() {
                ent.primary_key = custom.trim().to_string();
            }
        } else if choice > 0 {
            ent.primary_key = candidates[choice - 1].clone();
        }

        // Property keep/drop.
        if !ent.properties.is_empty() {
            let labels: Vec<String> = ent
                .properties
                .iter()
                .map(|p| {
                    format!(
                        "{} [{}] ← {}",
                        p.name,
                        p.field_type.as_deref().unwrap_or("?"),
                        p.source_path
                    )
                })
                .collect();
            let defaults = vec![true; labels.len()];
            let kept = MultiSelect::new()
                .with_prompt("  properties to keep (space toggles, enter confirms)")
                .items(&labels)
                .defaults(&defaults)
                .interact()
                .map_err(ui_err)?;
            let keep_set: std::collections::HashSet<usize> = kept.into_iter().collect();
            let mut idx = 0;
            ent.properties.retain(|_| {
                let keep = keep_set.contains(&idx);
                idx += 1;
                keep
            });

            // Optional per-property retype.
            if !ent.properties.is_empty()
                && Confirm::new()
                    .with_prompt("  adjust property types?")
                    .default(false)
                    .interact()
                    .map_err(ui_err)?
            {
                for p in &mut ent.properties {
                    let cur = p.field_type.as_deref().unwrap_or("Text");
                    let default_idx = FIELD_TYPES.iter().position(|t| *t == cur).unwrap_or(1);
                    let sel = Select::new()
                        .with_prompt(format!("    type for `{}`", p.name))
                        .items(FIELD_TYPES)
                        .default(default_idx)
                        .interact()
                        .map_err(ui_err)?;
                    p.field_type = Some(FIELD_TYPES[sel].to_string());
                }
            }
        }

        // Add properties the analyser saw but the mapping omitted.
        if let Some(es) = summary
            .entities
            .iter()
            .find(|e| e.source_path == ent.source_path)
        {
            let missing: Vec<_> = es
                .fields
                .iter()
                .filter(|f| {
                    !ent.properties
                        .iter()
                        .any(|p| p.source_path == f.source_path)
                })
                .collect();
            if !missing.is_empty()
                && Confirm::new()
                    .with_prompt(format!("  add any of {} unused field(s)?", missing.len()))
                    .default(false)
                    .interact()
                    .map_err(ui_err)?
            {
                let labels: Vec<String> = missing
                    .iter()
                    .map(|f| format!("{} ({})", f.name, f.source_path))
                    .collect();
                let picked = MultiSelect::new()
                    .with_prompt("  fields to add")
                    .items(&labels)
                    .interact()
                    .map_err(ui_err)?;
                for i in picked {
                    let f = missing[i];
                    ent.properties.push(PropertyMapping {
                        name: f.name.clone(),
                        source_path: f.source_path.clone(),
                        description: None,
                        field_type: Some(inferred_to_field_type(f.inferred_type).to_string()),
                    });
                }
            }
        }
        println!();
    }

    // Relationship review.
    if !mapping.relationships.is_empty() {
        println!("── Relationships");
        let mut keep: Vec<RelationshipMapping> = Vec::new();
        for rel in std::mem::take(&mut mapping.relationships) {
            let ok = Confirm::new()
                .with_prompt(format!(
                    "  keep `{}` ({} → {})?",
                    rel.kind, rel.from, rel.to
                ))
                .default(true)
                .interact()
                .map_err(ui_err)?;
            if ok {
                keep.push(rel);
            }
        }
        mapping.relationships = keep;
    }

    // Add new relationships.
    let kinds: Vec<String> = mapping.entities.iter().map(|e| e.kind.clone()).collect();
    if !kinds.is_empty() {
        while Confirm::new()
            .with_prompt("  add a relationship?")
            .default(false)
            .interact()
            .map_err(ui_err)?
        {
            let from = Select::new()
                .with_prompt("    from")
                .items(&kinds)
                .default(0)
                .interact()
                .map_err(ui_err)?;
            let to = Select::new()
                .with_prompt("    to")
                .items(&kinds)
                .default(0)
                .interact()
                .map_err(ui_err)?;
            let kind: String = Input::new()
                .with_prompt("    relationship type (UPPER_SNAKE)")
                .interact_text()
                .map_err(ui_err)?;
            let kind = kind.trim();
            if !kind.is_empty() {
                // Optional foreign-key join: required when `from`/`to`
                // come from separate top-level arrays linked by an id.
                // Leave the from-key blank to use array-context alignment.
                let from_key: String = Input::new()
                    .with_prompt("    from_key JSONPath (FK on `from`; blank = none)")
                    .allow_empty(true)
                    .interact_text()
                    .map_err(ui_err)?;
                let (from_key, to_key) = if from_key.trim().is_empty() {
                    (None, None)
                } else {
                    let to_key: String = Input::new()
                        .with_prompt("    to_key JSONPath (blank = target primary_key)")
                        .allow_empty(true)
                        .interact_text()
                        .map_err(ui_err)?;
                    let to_key = to_key.trim();
                    (
                        Some(from_key.trim().to_string()),
                        if to_key.is_empty() {
                            None
                        } else {
                            Some(to_key.to_string())
                        },
                    )
                };
                mapping.relationships.push(RelationshipMapping {
                    kind: kind.to_ascii_uppercase(),
                    from: kinds[from].clone(),
                    to: kinds[to].clone(),
                    from_key,
                    to_key,
                });
            }
        }
    }

    // Re-validate and re-verify after edits.
    enforce_ontology(&mut mapping, ontology)?;
    mapping.validate().map_err(MapGenError::Parse)?;
    mapper::extract(&mapping, data).map_err(MapGenError::Verify)?;

    println!("✓ Refined mapping is valid.\n");
    Ok(mapping)
}

fn inferred_to_field_type(t: InferredType) -> &'static str {
    match t {
        InferredType::Keyword => "Keyword",
        InferredType::DateTime => "DateTime",
        InferredType::Number => "Number",
        InferredType::Boolean => "Boolean",
        InferredType::Text | InferredType::Unknown | InferredType::Identifier => "Text",
    }
}
