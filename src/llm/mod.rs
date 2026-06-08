//! Provider-agnostic LLM client abstraction.
//!
//! The rest of the crate stays free of provider plumbing: callers depend
//! on the [`LlmClient`] trait and pick a concrete backend at the edge.
//!
//! * [`MockLlmClient`] — deterministic, network-free; used by tests.
//! * [`OpenAiClient`] — talks to any OpenAI-compatible
//!   `/v1/chat/completions` endpoint (e.g. a self-hosted **vLLM**
//!   server). Gated behind the `openai` cargo feature so the default
//!   build can stay HTTP-free.
//!
//! The trait is intentionally minimal: a single [`LlmClient::complete`]
//! turning a `(system, user)` message pair into a completion string.
//! Higher-level orchestration (prompt assembly, repair loops, parsing)
//! lives in [`crate::mapgen`].

use async_trait::async_trait;
use thiserror::Error;

mod mock;
#[cfg(feature = "openai")]
mod openai;

pub use mock::MockLlmClient;
#[cfg(feature = "openai")]
pub use openai::OpenAiClient;

/// Errors surfaced by an [`LlmClient`].
#[derive(Debug, Error)]
pub enum LlmError {
    /// Transport-level failure (connection refused, timeout, …). Held as
    /// a string so the enum compiles without the HTTP dependency.
    #[error("LLM transport error: {0}")]
    Http(String),

    /// The endpoint returned a non-2xx status.
    #[error("LLM API error (status {status}): {message}")]
    Api { status: u16, message: String },

    /// The response body could not be decoded into the expected shape.
    #[error("failed to decode LLM response: {0}")]
    Decode(String),

    /// The configured API-key environment variable was required but unset.
    #[error("missing API key: environment variable `{0}` is not set")]
    MissingApiKey(String),

    /// The model returned an empty completion.
    #[error("LLM returned an empty completion")]
    EmptyResponse,
}

/// A minimal chat-style LLM client.
///
/// Implementations must be cheap to share (`Send + Sync`); callers
/// typically hold them behind an `Arc<dyn LlmClient>`.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Run a single completion. `system` carries the instructions /
    /// constraints; `user` carries the task-specific payload. Returns the
    /// raw completion text (callers are responsible for any
    /// post-processing such as stripping Markdown fences).
    async fn complete(&self, system: &str, user: &str) -> Result<String, LlmError>;
}
