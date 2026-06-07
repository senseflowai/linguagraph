//! Deterministic, network-free [`LlmClient`] for tests.

use std::sync::Mutex;

use async_trait::async_trait;

use super::{LlmClient, LlmError};

/// A mock LLM that replays a queue of canned responses and records every
/// `(system, user)` prompt it was asked to complete.
///
/// Each call to [`LlmClient::complete`] pops the next queued response.
/// When the queue is exhausted the **last** response is replayed (so a
/// single happy-path response survives repair retries); if the queue was
/// empty to begin with, an [`LlmError::EmptyResponse`] is returned.
#[derive(Debug, Default)]
pub struct MockLlmClient {
    responses: Mutex<std::collections::VecDeque<String>>,
    last: Mutex<Option<String>>,
    /// Captured prompts, in call order: `(system, user)`.
    calls: Mutex<Vec<(String, String)>>,
}

impl MockLlmClient {
    /// Build a mock that replays `responses` in order.
    pub fn new<I, S>(responses: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            responses: Mutex::new(responses.into_iter().map(Into::into).collect()),
            last: Mutex::new(None),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Convenience constructor for a mock that always returns `response`.
    pub fn single(response: impl Into<String>) -> Self {
        Self::new([response.into()])
    }

    /// Number of times [`LlmClient::complete`] has been invoked.
    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    /// Snapshot of every captured `(system, user)` prompt pair.
    pub fn calls(&self) -> Vec<(String, String)> {
        self.calls.lock().unwrap().clone()
    }

    /// The `user` payload of the most recent call, if any.
    pub fn last_user_prompt(&self) -> Option<String> {
        self.calls.lock().unwrap().last().map(|(_, u)| u.clone())
    }
}

#[async_trait]
impl LlmClient for MockLlmClient {
    async fn complete(&self, system: &str, user: &str) -> Result<String, LlmError> {
        self.calls
            .lock()
            .unwrap()
            .push((system.to_string(), user.to_string()));

        let next = self.responses.lock().unwrap().pop_front();
        match next {
            Some(resp) => {
                *self.last.lock().unwrap() = Some(resp.clone());
                Ok(resp)
            }
            None => match self.last.lock().unwrap().clone() {
                Some(resp) => Ok(resp),
                None => Err(LlmError::EmptyResponse),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn replays_in_order_then_repeats_last() {
        let mock = MockLlmClient::new(["first", "second"]);
        assert_eq!(mock.complete("s", "a").await.unwrap(), "first");
        assert_eq!(mock.complete("s", "b").await.unwrap(), "second");
        // Queue exhausted → last response repeats.
        assert_eq!(mock.complete("s", "c").await.unwrap(), "second");
        assert_eq!(mock.call_count(), 3);
        assert_eq!(mock.last_user_prompt().as_deref(), Some("c"));
    }

    #[tokio::test]
    async fn empty_queue_errors() {
        let mock = MockLlmClient::new(Vec::<String>::new());
        assert!(matches!(
            mock.complete("s", "u").await,
            Err(LlmError::EmptyResponse)
        ));
    }
}
