//! `llama-cpp-2`-backed embedder.
//!
//! Loads a GGUF embedding model once and reuses it across calls. The
//! [`LlamaBackend`] is a process-singleton (it can only be initialised
//! once) so we keep ours behind a `OnceCell`-style guard.
//!
//! Compiled only when the `llama` feature is enabled — keeps the default
//! build dependency-free.

use std::num::NonZeroU32;
use std::sync::Mutex;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use once_cell::sync::OnceCell;

use super::{EmbedError, Embedder};

/// Process-global llama backend. Calling `LlamaBackend::init()` more
/// than once returns an error from llama.cpp.
static BACKEND: OnceCell<LlamaBackend> = OnceCell::new();

fn backend() -> Result<&'static LlamaBackend, EmbedError> {
    BACKEND
        .get_or_try_init(|| LlamaBackend::init().map_err(|e| EmbedError::Backend(e.to_string())))
}

#[derive(Debug)]
pub struct LlamaEmbedder {
    /// Model is held behind a Mutex because `new_context` borrows it
    /// mutably under the hood (depending on llama-cpp-2 version) and we
    /// must serialise concurrent embed calls regardless.
    model: Mutex<LlamaModel>,
    dim: usize,
    n_ctx: u32,
}

impl LlamaEmbedder {
    /// Load a GGUF embedding model from `path`.
    pub fn load(path: &str) -> Result<Self, EmbedError> {
        let backend = backend()?;
        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(backend, path, &params)
            .map_err(|e| EmbedError::Io(format!("loading {path}: {e}")))?;
        // Use embedding-size from the model. Fall back to a sensible
        // default if the binding doesn't expose it directly; the Qdrant
        // collection is created lazily so a wrong guess only manifests
        // on the first insert.
        let dim = model_dim(&model);
        Ok(Self { model: Mutex::new(model), dim, n_ctx: 4096 })
    }

    fn embed_locked(
        model: &LlamaModel,
        texts: &[&str],
        n_ctx: u32,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let backend = backend()?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_seq_max(texts.len().max(1) as u32)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| EmbedError::Backend(e.to_string()))?;

        let token_lines: Vec<_> = texts
            .iter()
            .map(|t| model.str_to_token(t, AddBos::Always))
            .collect::<Result<_, _>>()
            .map_err(|e| EmbedError::Backend(format!("tokenize: {e}")))?;

        let max_tokens = 512;
        let mut batch = LlamaBatch::new(max_tokens, token_lines.len() as i32);
        let mut output: Vec<Vec<f32>> = Vec::with_capacity(token_lines.len());
        let mut seq_id: i32 = 0;

        for tokens in &token_lines {
            if (batch.n_tokens() as usize + tokens.len()) > max_tokens {
                flush(&mut ctx, &mut batch, seq_id, &mut output)?;
                seq_id = 0;
                batch.clear();
            }
            batch
                .add_sequence(tokens, seq_id, false)
                .map_err(|e| EmbedError::Backend(e.to_string()))?;
            seq_id += 1;
        }
        flush(&mut ctx, &mut batch, seq_id, &mut output)?;
        Ok(output)
    }
}

fn flush(
    ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
    batch: &mut LlamaBatch,
    n_seqs: i32,
    output: &mut Vec<Vec<f32>>,
) -> Result<(), EmbedError> {
    if n_seqs == 0 {
        return Ok(());
    }
    ctx.clear_kv_cache();
    ctx.decode(batch).map_err(|e| EmbedError::Backend(e.to_string()))?;
    for i in 0..n_seqs {
        let emb = ctx
            .embeddings_seq_ith(i)
            .map_err(|e| EmbedError::Backend(e.to_string()))?;
        output.push(l2_normalise(emb));
    }
    batch.clear();
    Ok(())
}

fn l2_normalise(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Best-effort embedding dimension lookup. Older `llama-cpp-2` versions
/// expose this as `n_embd()`; we treat it as a hint only.
fn model_dim(model: &LlamaModel) -> usize {
    // The exact accessor name has churned between releases; this
    // shim isolates the breakage to one place.
    model.n_embd() as usize
}

impl Embedder for LlamaEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let model = self
            .model
            .lock()
            .map_err(|e| EmbedError::Backend(format!("model mutex poisoned: {e}")))?;
        Self::embed_locked(&model, texts, self.n_ctx)
    }
}
