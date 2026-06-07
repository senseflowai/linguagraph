//! OpenAI-compatible chat-completions client.
//!
//! Works against any server implementing the OpenAI `/v1/chat/completions`
//! contract — in particular a self-hosted **vLLM** OpenAI server. Gated
//! behind the `openai` feature so the default build pulls in no HTTP
//! stack.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use crate::config::LlmConfig;

use super::{LlmClient, LlmError};

/// HTTP client for an OpenAI-compatible chat-completions endpoint.
#[derive(Debug, Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    /// Base URL **without** a trailing slash, e.g. `http://localhost:8000/v1`.
    base_url: String,
    model: String,
    /// Optional bearer token. Absent is fine for local vLLM servers.
    api_key: Option<String>,
    temperature: f32,
    max_tokens: Option<u32>,
}

impl OpenAiClient {
    /// Build a client from [`LlmConfig`]. The API key is read from the
    /// environment variable named by `cfg.api_key_env` (absent is
    /// tolerated — local servers usually don't require one).
    pub fn from_config(cfg: &LlmConfig) -> Self {
        let api_key = std::env::var(&cfg.api_key_env)
            .ok()
            .filter(|k| !k.trim().is_empty());
        Self {
            http: reqwest::Client::new(),
            base_url: cfg.base_url.trim_end_matches('/').to_string(),
            model: cfg.model.clone(),
            api_key,
            temperature: cfg.temperature,
            max_tokens: cfg.max_tokens,
        }
    }

    /// Override the model after construction (used by CLI flags).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the base URL after construction (used by CLI flags).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[serde(default)]
    message: ChoiceMessage,
}

#[derive(Debug, Default, Deserialize)]
struct ChoiceMessage {
    #[serde(default)]
    content: String,
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn complete(&self, system: &str, user: &str) -> Result<String, LlmError> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = json!({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        });
        if let Some(max) = self.max_tokens {
            body["max_tokens"] = json!(max);
        }

        let mut req = self.http.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req.send().await.map_err(|e| LlmError::Http(e.to_string()))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        if !status.is_success() {
            return Err(LlmError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let parsed: ChatResponse =
            serde_json::from_str(&text).map_err(|e| LlmError::Decode(e.to_string()))?;

        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .filter(|c| !c.trim().is_empty())
            .ok_or(LlmError::EmptyResponse)
    }
}
