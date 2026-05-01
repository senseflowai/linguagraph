//! In-memory test double for [`GraphClient`].
//!
//! Stores a queue of canned responses; each call to [`execute`] pops the
//! next one. Anything beyond the queued responses returns an empty result.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::builder::CypherQuery;

use super::result::QueryResult;
use super::{DbError, GraphClient};

#[derive(Debug, Default)]
pub struct MockClient {
    queue: Mutex<Vec<QueryResult>>,
    pub captured: Mutex<Vec<CypherQuery>>,
}

impl MockClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, result: QueryResult) {
        self.queue.lock().expect("mock queue poisoned").push(result);
    }
}

#[async_trait]
impl GraphClient for MockClient {
    async fn execute(&self, q: &CypherQuery) -> Result<QueryResult, DbError> {
        self.captured
            .lock()
            .expect("captured poisoned")
            .push(q.clone());
        let next = self.queue.lock().expect("mock queue poisoned").pop();
        Ok(next.unwrap_or_default())
    }
}
