//! `llama-cpp-2`-backed embedder.
//!
//! Loads a GGUF embedding model once and reuses it across calls. The
//! [`LlamaBackend`] is a process-singleton (it can only be initialised
//! once) so we keep ours behind a `OnceCell`-style guard.
//!
//! Compiled only when the `llama` feature is enabled — keeps the default
//! build dependency-free.

use super::{EmbedError, Embedder};
use anyhow::Context;
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use once_cell::sync::OnceCell;
use std::num::NonZeroU32;
use std::sync::Mutex;

/// Process-global llama backend. Calling `LlamaBackend::init()` more
/// than once returns an error from llama.cpp.
static BACKEND: OnceCell<LlamaBackend> = OnceCell::new();

fn backend() -> Result<&'static LlamaBackend, EmbedError> {
    BACKEND.get_or_try_init(|| {
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));
        LlamaBackend::init().map_err(|e| EmbedError::Backend(e.to_string()))
    })
}

pub struct LLamaTokenizer {
    pub tokens: Vec<LlamaToken>,
}

impl LLamaTokenizer {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }

    pub fn tokenize(
        &mut self,
        model: &LlamaModel,
        prompt: &str,
    ) -> anyhow::Result<&Vec<LlamaToken>> {
        self.tokens = model
            .str_to_token(prompt, AddBos::Always)
            .with_context(|| format!("failed to tokenize {prompt}"))?;
        Ok(&self.tokens)
    }
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
        Ok(Self {
            model: Mutex::new(model),
            dim,
            n_ctx: 4096,
        })
    }

    fn tokenize_prompt(model: &LlamaModel, prompt: &str) -> anyhow::Result<LLamaTokenizer> {
        let mut tokenizer = LLamaTokenizer::new();

        let tokens = tokenizer.tokenize(model, prompt)?;
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        for token in tokens {
            model.token_to_piece(*token, &mut decoder, true, None)?;
        }

        Ok(tokenizer)
    }

    fn split_into_batches(
        tokenizers: Vec<LLamaTokenizer>,
        max_tokens: usize,
    ) -> Vec<Vec<LLamaTokenizer>> {
        let mut batches: Vec<Vec<LLamaTokenizer>> = Vec::new();
        let mut current_batch: Vec<LLamaTokenizer> = Vec::new();
        let mut current_tokens = 0;

        for tokenizer in tokenizers {
            let len = tokenizer.tokens.len();

            if len > max_tokens {
                if !current_batch.is_empty() {
                    batches.push(current_batch);
                    current_batch = Vec::new();
                    current_tokens = 0;
                }
                batches.push(vec![tokenizer]);
                continue;
            }

            if current_tokens + len > max_tokens {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_tokens = 0;
            }

            current_tokens += len;
            current_batch.push(tokenizer);
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        batches
    }

    fn embed_locked(
        model: &LlamaModel,
        texts: &[&str],
        n_ctx: u32,
    ) -> Result<Vec<Vec<f32>>, EmbedError> {
        let backend = backend()?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            // .with_n_seq_max(texts.len().max(1) as u32)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Unspecified);
        let mut ctx = model
            .new_context(backend, ctx_params)
            .map_err(|e| EmbedError::Backend(e.to_string()))?;

        let token_lines: Vec<_> = texts
            .iter()
            .map(|t| model.str_to_token(t, AddBos::Always))
            .collect::<Result<_, _>>()
            .map_err(|e| EmbedError::Backend(format!("tokenize: {e}")))?;

        let max_tokens = 512;

        let mut tokenizers: Vec<LLamaTokenizer> = Vec::new();

        for prompt in texts {
            let tokenizer = Self::tokenize_prompt(model, prompt)
                .map_err(|e| EmbedError::Backend(format!("tokenize: {e}")))?;
            tokenizers.push(tokenizer);
        }

        let tokenizers_batches = Self::split_into_batches(tokenizers, ctx.n_ctx() as usize);

        let mut output: Vec<Vec<f32>> = Vec::with_capacity(token_lines.len());

        ctx.clear_kv_cache();

        for tokenizer_batch in tokenizers_batches {
            for (_, tokenizer) in tokenizer_batch.iter().enumerate() {
                if tokenizer.tokens.len() <= max_tokens {
                    let mut batch = LlamaBatch::new(tokenizer.tokens.len(), 1);
                    batch
                        .add_sequence(&tokenizer.tokens, 0, false)
                        .map_err(|e| EmbedError::Backend(e.to_string()))?;
                    ctx.decode(&mut batch)
                        .map_err(|e| EmbedError::Backend(e.to_string()))?;
                    let emb = ctx
                        .embeddings_seq_ith(0)
                        .map_err(|e| EmbedError::Backend(e.to_string()))?;
                    output.push(l2_normalise(emb));
                    batch.clear();
                } else {
                    let mut batches = Vec::new();
                    for chunk in tokenizer.tokens.chunks(max_tokens) {
                        let mut batch = LlamaBatch::new(chunk.len(), 1);
                        batch
                            .add_sequence(&chunk, 0, false)
                            .map_err(|e| EmbedError::Backend(e.to_string()))?;
                        batches.push(batch);
                    }

                    for batch in &mut batches {
                        ctx.decode(batch)
                            .map_err(|e| EmbedError::Backend(e.to_string()))?;
                    }

                    let emb = ctx
                        .embeddings_seq_ith(0)
                        .map_err(|e| EmbedError::Backend(e.to_string()))?;
                    output.push(l2_normalise(emb));

                    for batch in &mut batches {
                        batch.clear();
                    }
                }
            }
        }

        Ok(output)
    }
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
