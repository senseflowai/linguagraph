//! Detect, filter, and sample the top-level *collections* of an input
//! JSON document for `generate-mapping`.
//!
//! A "collection" is a top-level array of objects — one per key of a root
//! object, or the root array itself. The user picks which collections to
//! map; only those are analysed, verified, and inlined into the prompt.
//!
//! The three helpers here are pure functions over [`serde_json::Value`]:
//!
//! * [`detect_collections`] — enumerate the selectable collections;
//! * [`filter_collections`] — keep only the chosen collections (all rows);
//! * [`sample_arrays`] — shrink every array to a structural sample for the
//!   prompt, preserving the document shape.

use serde_json::{Map, Value};

/// Name used for the single collection of a root-level array, or a root
/// object that is itself a singleton record (no array children).
pub const ROOT_COLLECTION: &str = "(root)";

/// A selectable top-level collection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectionInfo {
    /// JSON key (for a root object) or [`ROOT_COLLECTION`] for a root
    /// array / singleton-document root.
    pub name: String,
    /// JSONPath to the collection, e.g. `$.cameras` or `$`.
    pub path: String,
    /// Number of items (array length, or 1 for a singleton document).
    pub len: usize,
}

/// Enumerate the top-level collections of `data`.
///
/// * root object → one entry per key whose value is a non-empty array
///   containing at least one object;
/// * root array (of objects) → a single [`ROOT_COLLECTION`] entry;
/// * root object with no array-of-object children → a single
///   [`ROOT_COLLECTION`] entry covering the whole document (the analyser's
///   "Document" singleton case);
/// * scalar / empty root → empty vec (the caller reports "nothing to map").
pub fn detect_collections(data: &Value) -> Vec<CollectionInfo> {
    match data {
        Value::Array(items) => {
            if items.iter().any(Value::is_object) {
                vec![CollectionInfo {
                    name: ROOT_COLLECTION.to_string(),
                    path: "$".to_string(),
                    len: items.len(),
                }]
            } else {
                Vec::new()
            }
        }
        Value::Object(map) => {
            let mut out: Vec<CollectionInfo> = Vec::new();
            for (k, v) in map {
                if let Value::Array(items) = v {
                    if items.iter().any(Value::is_object) {
                        out.push(CollectionInfo {
                            name: k.clone(),
                            path: format!("$.{k}"),
                            len: items.len(),
                        });
                    }
                }
            }
            // Root object with no array-of-object children: treat the
            // whole document as a single collection (the singleton
            // "Document" the analyser surfaces).
            if out.is_empty() && !map.is_empty() {
                out.push(CollectionInfo {
                    name: ROOT_COLLECTION.to_string(),
                    path: "$".to_string(),
                    len: 1,
                });
            }
            out
        }
        _ => Vec::new(),
    }
}

/// Return a copy of `data` containing only the named collections.
///
/// * root object → an object with only the selected keys (original key
///   order preserved), each with all of its items intact;
/// * root selected wholesale via [`ROOT_COLLECTION`], root array, or a
///   scalar → `data` is returned unchanged.
pub fn filter_collections(data: &Value, selected: &[String]) -> Value {
    match data {
        Value::Object(map) => {
            // A root chosen wholesale (singleton document) keeps everything.
            if selected.iter().any(|s| s == ROOT_COLLECTION) {
                return data.clone();
            }
            let mut out = Map::new();
            for (k, v) in map {
                if selected.iter().any(|s| s == k) {
                    out.insert(k.clone(), v.clone());
                }
            }
            Value::Object(out)
        }
        // Root array (or scalar): the selection is all-or-nothing, so the
        // document is kept as is.
        _ => data.clone(),
    }
}

/// Recursively copy `value`, capping every array to at most `max_items`
/// elements. Objects and the kept array elements are recursed into, so
/// nested arrays shrink too while the document structure is preserved.
pub fn sample_arrays(value: &Value, max_items: usize) -> Value {
    match value {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .take(max_items)
                .map(|v| sample_arrays(v, max_items))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), sample_arrays(v, max_items)))
                .collect(),
        ),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn names(cs: &[CollectionInfo]) -> Vec<String> {
        cs.iter().map(|c| c.name.clone()).collect()
    }

    #[test]
    fn detect_root_object_multiple_collections() {
        let data = json!({
            "cameras": [{"id": 1}, {"id": 2}],
            "places": [{"id": "p1"}]
        });
        let cs = detect_collections(&data);
        assert_eq!(names(&cs), vec!["cameras".to_string(), "places".to_string()]);
        let cam = cs.iter().find(|c| c.name == "cameras").unwrap();
        assert_eq!(cam.path, "$.cameras");
        assert_eq!(cam.len, 2);
    }

    #[test]
    fn detect_skips_scalar_arrays_and_non_arrays() {
        let data = json!({
            "tags": ["a", "b", "c"],      // array of scalars — not a collection
            "meta": {"version": 1},        // object — not a collection
            "cameras": [{"id": 1}]         // the only real collection
        });
        let cs = detect_collections(&data);
        assert_eq!(names(&cs), vec!["cameras".to_string()]);
    }

    #[test]
    fn detect_root_array_is_single_root_collection() {
        let data = json!([{"id": 1}, {"id": 2}, {"id": 3}]);
        let cs = detect_collections(&data);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].name, ROOT_COLLECTION);
        assert_eq!(cs[0].path, "$");
        assert_eq!(cs[0].len, 3);
    }

    #[test]
    fn detect_singleton_document_object() {
        let data = json!({"id": "x", "title": "hello"});
        let cs = detect_collections(&data);
        assert_eq!(cs.len(), 1);
        assert_eq!(cs[0].name, ROOT_COLLECTION);
        assert_eq!(cs[0].path, "$");
        assert_eq!(cs[0].len, 1);
    }

    #[test]
    fn detect_scalar_or_empty_root_is_empty() {
        assert!(detect_collections(&json!(42)).is_empty());
        assert!(detect_collections(&json!("text")).is_empty());
        assert!(detect_collections(&json!({})).is_empty());
        assert!(detect_collections(&json!([])).is_empty());
    }

    #[test]
    fn filter_keeps_selected_object_keys_only() {
        let data = json!({
            "cameras": [{"id": 1}],
            "places": [{"id": "p1"}],
            "meta": {"v": 1}
        });
        let filtered = filter_collections(&data, &["cameras".to_string()]);
        let obj = filtered.as_object().unwrap();
        assert!(obj.contains_key("cameras"));
        assert!(!obj.contains_key("places"));
        assert!(!obj.contains_key("meta"));
    }

    #[test]
    fn filter_root_array_is_unchanged() {
        let data = json!([{"id": 1}, {"id": 2}]);
        let filtered = filter_collections(&data, &[ROOT_COLLECTION.to_string()]);
        assert_eq!(filtered, data);
    }

    #[test]
    fn filter_singleton_document_via_root_keeps_everything() {
        let data = json!({"id": "x", "title": "hello"});
        let filtered = filter_collections(&data, &[ROOT_COLLECTION.to_string()]);
        assert_eq!(filtered, data);
    }

    #[test]
    fn sample_caps_top_level_and_nested_arrays() {
        let data = json!({
            "cameras": [
                {"id": 1, "lenses": [{"f": 1}, {"f": 2}, {"f": 3}]},
                {"id": 2},
                {"id": 3},
                {"id": 4}
            ]
        });
        let sampled = sample_arrays(&data, 2);
        let cams = sampled["cameras"].as_array().unwrap();
        assert_eq!(cams.len(), 2); // top-level capped
        let lenses = cams[0]["lenses"].as_array().unwrap();
        assert_eq!(lenses.len(), 2); // nested capped too
        // Keys and scalars preserved.
        assert_eq!(cams[0]["id"], json!(1));
    }

    #[test]
    fn sample_leaves_short_arrays_and_scalars_intact() {
        let data = json!({"xs": [1], "name": "a", "n": 7});
        let sampled = sample_arrays(&data, 2);
        assert_eq!(sampled, data);
    }

    #[test]
    fn sample_root_array() {
        let data = json!([{"id": 1}, {"id": 2}, {"id": 3}]);
        let sampled = sample_arrays(&data, 2);
        assert_eq!(sampled.as_array().unwrap().len(), 2);
    }
}
