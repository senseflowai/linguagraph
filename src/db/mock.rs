//! In-memory test double for [`GraphClient`].
//!
//! Stores a stack of canned responses; each call to [`execute`] pops the
//! most recently enqueued one (LIFO), so multi-query tests enqueue in
//! **reverse call order**. Anything beyond the queued responses returns
//! an empty result.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::builder::CypherQuery;
use crate::prompt::GraphSchema;

use super::result::QueryResult;
use super::{DbError, GraphClient};

#[derive(Debug, Default)]
pub struct MockClient {
    queue: Mutex<Vec<QueryResult>>,
    pub captured: Mutex<Vec<CypherQuery>>,
    schema: Mutex<GraphSchema>,
}

impl MockClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&self, result: QueryResult) {
        self.queue.lock().expect("mock queue poisoned").push(result);
    }

    pub fn set_schema(&self, schema: GraphSchema) {
        *self.schema.lock().expect("mock schema poisoned") = schema;
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

    async fn schema(&self) -> Result<GraphSchema, DbError> {
        Ok(self.schema.lock().expect("mock schema poisoned").clone())
    }
}
