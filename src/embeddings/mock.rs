//! Deterministic, dependency-free embedder used by tests and dry-runs.
//!
//! Hashes each input into a fixed-dimension vector with values in
//! `[-1, 1]`. Same input → same vector across runs, processes, and
//! machines, which is exactly what test snapshots need.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::{EmbedError, Embedder};

#[derive(Debug)]
pub struct MockEmbedder {
    dim: usize,
}

impl MockEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "MockEmbedder dim must be > 0");
        Self { dim }
    }
}

impl Embedder for MockEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(hash_to_vector(t, self.dim));
        }
        Ok(out)
    }
}

fn hash_to_vector(text: &str, dim: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut h = DefaultHasher::new();
        text.hash(&mut h);
        i.hash(&mut h);
        let n = h.finish();
        // Map u64 to roughly [-1, 1].
        let f = ((n & 0xFFFF) as f32 / 32_768.0) - 1.0;
        v.push(f);
    }
    // L2-normalise so cosine similarity behaves sensibly downstream.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_is_deterministic() {
        let e = MockEmbedder::new(8);
        let a = e.embed("hello").unwrap();
        let b = e.embed("hello").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_inputs_produce_different_vectors() {
        let e = MockEmbedder::new(16);
        let a = e.embed("apple").unwrap();
        let b = e.embed("banana").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn batch_matches_single() {
        let e = MockEmbedder::new(8);
        let single = e.embed("foo").unwrap();
        let batch = e.embed_batch(&["foo", "bar"]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], single);
        assert_ne!(batch[0], batch[1]);
    }

    #[test]
    fn output_is_unit_normalised() {
        let e = MockEmbedder::new(64);
        let v = e.embed("test").unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "expected unit norm, got {norm}");
    }
}
