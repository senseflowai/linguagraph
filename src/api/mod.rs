//! Public API surface for the `linguagraph` crate.
//!
//! This module is the stable contract a REST service (and the knowledge
//! graph UI behind it) codes against. The service owns HTTP transport,
//! end-user authorization, response caching, and rendering; this crate
//! owns the graph — compiling questions into safe Cypher, traversal,
//! search, knowledge extraction, deduplication, and provenance.
//!
//! # Design in one screen
//!
//! * **One cheap-to-clone handle.** [`LinguaGraph`] wraps the connection
//!   pool and config. Create it once per process; clone it per request.
//! * **Everything is tenant-scoped.** You cannot read or write without
//!   selecting a tenant: [`LinguaGraph::read`] / [`LinguaGraph::write`]
//!   return tenant-bound scopes. Isolation is enforced inside the library.
//! * **Read/write split at the type level.** [`GraphRead`] and
//!   [`GraphWrite`] are separate traits, so the read path can be handed a
//!   read-only capability (least privilege; routable to a replica).
//! * **Traits, not a concrete type.** The service depends on
//!   `dyn GraphRead` / `dyn GraphWrite`, which keeps it mockable and
//!   backend-agnostic.
//! * **Async, `Result<T, GraphError>` everywhere.** Cancellation is by
//!   dropping the future; large/long responses stream.
//! * **Cursor-only lists.** Anything potentially large returns a
//!   [`Page<T>`] with an opaque [`Cursor`].
//! * **Bounded traversals.** [`TraversalOptions`] requires `depth` and
//!   `limit`; an unbounded traversal is rejected.
//! * **Everything is `serde`.** Types cross the REST boundary with almost
//!   no re-marshalling.
//! * **Stable identifiers.** [`EntityId`] / [`RelationId`] survive
//!   restarts, so share links and cursors keep working.
//!
//! # Example: a neighbours endpoint
//!
//! ```no_run
//! use std::sync::Arc;
//! use linguagraph::api::{Direction, EntityId, GraphRead, LinguaGraph, TenantId, TraversalOptions};
//!
//! async fn neighbors(graph: LinguaGraph, tenant: String, id: String) {
//!     let scope = graph.read(TenantId(tenant)); // read-only; can go to a replica
//!     let page = scope
//!         .neighbors(
//!             &EntityId(id),
//!             TraversalOptions::one_hop(25), // depth + limit are mandatory
//!         )
//!         .await;
//!     // `page` is already `serde`-serializable — the service just wraps it.
//!     let _ = page;
//! }
//! ```

mod error;
mod handle;
mod model;
mod options;
mod traits;

pub use error::{GraphError, Result};

pub use model::{
    Confidence, ConfidenceLevel, Cost, Cursor, Diagnostics, Domain, Entity, EntityId,
    EntitySummary, EntityType, FacetDim, Facets, Filters, Locale, Page, Property, PropertyKind,
    Provenance, Relation, RelationId, RelationType, SourceId, SourceRef, Subgraph, TenantId,
    Timestamp, Value,
};

pub use options::{
    Answer, AnswerChunk, AnswerMeta, AnswerMode, AskOptions, DeleteReport, Direction, EntityHit,
    EntityTypeInfo, ExtractOptions, Extraction, GraphBatch, MergeCandidate, MergeDecision,
    MergePolicy, MergeSignal, NeighborsPage, Path, PathOptions, QueryPlan, ResolveReport,
    ReviewOptions, RunOptions, SearchMode, SearchOptions, TraversalOptions, TypeMatch,
    UpsertOptions, UpsertReport,
};

pub use traits::{Cache, Embedder, GraphRead, GraphWrite, SharedEmbedder};

pub use handle::{
    LinguaGraph, LinguaGraphBuilder, Limits, Ontology, PoolConfig, ReadScope, WriteScope,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::core::Pipeline;
    use crate::db::MockClient;
    use std::sync::Arc;

    fn test_config() -> Config {
        // Minimal config: a database block plus all-default sub-configs.
        crate::config::load_from_str(
            r#"
            [database]
            uri = "bolt://localhost:7687"
            user = ""
            password = ""
            "#,
        )
        .expect("valid test config")
    }

    fn test_graph() -> LinguaGraph {
        let pipeline = Pipeline::new(Arc::new(MockClient::new()), &test_config());
        LinguaGraph::from_pipeline(pipeline, Limits::default())
    }

    #[test]
    fn ids_serialize_transparently() {
        let id = EntityId("acme".into());
        assert_eq!(serde_json::to_string(&id).unwrap(), r#""acme""#);
        let back: EntityId = serde_json::from_str(r#""acme""#).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn confidence_levels_bucket_by_score() {
        assert_eq!(Confidence::from_score(0.9).level, ConfidenceLevel::High);
        assert_eq!(Confidence::from_score(0.6).level, ConfidenceLevel::Medium);
        assert_eq!(Confidence::from_score(0.2).level, ConfidenceLevel::Review);
    }

    #[test]
    fn read_and_write_scopes_bind_their_tenant() {
        let graph = test_graph();
        let read = graph.read(TenantId("ws_1".into()));
        assert_eq!(read.tenant().as_str(), "ws_1");
        let write = graph.write(TenantId("ws_2".into()));
        assert_eq!(write.tenant().as_str(), "ws_2");
    }

    #[test]
    fn dyn_dispatch_is_object_safe() {
        // Compile-time proof the traits are object-safe (the whole point
        // of the `dyn GraphRead` / `dyn GraphWrite` service contract).
        let graph = test_graph();
        let _read: Box<dyn GraphRead> = Box::new(graph.read(TenantId("t".into())));
        let _write: Box<dyn GraphWrite> = Box::new(graph.write(TenantId("t".into())));
    }

    #[tokio::test]
    async fn unbounded_traversal_is_rejected() {
        let graph = test_graph();
        let scope = graph.read(TenantId("t".into()));
        let mut opts = TraversalOptions::one_hop(10);
        opts.limit = 0; // remove the bound
        let err = scope
            .neighbors(&EntityId("x".into()), opts)
            .await
            .unwrap_err();
        assert!(matches!(err, GraphError::UnboundedTraversal));
    }

    #[tokio::test]
    async fn empty_question_is_invalid() {
        let graph = test_graph();
        let scope = graph.read(TenantId("t".into()));
        let err = scope.ask("   ", AskOptions::default()).await.unwrap_err();
        assert!(matches!(err, GraphError::InvalidQuestion(_)));
    }

    #[tokio::test]
    async fn delete_by_unknown_source_reports_not_found() {
        // The MockClient returns an empty result, so the pipeline's
        // source discovery finds nothing → source_found = false.
        let graph = test_graph();
        let scope = graph.write(TenantId("t".into()));
        let report = scope
            .delete_by_source(&SourceId("nope".into()))
            .await
            .unwrap();
        assert!(!report.source_found);
    }
}
