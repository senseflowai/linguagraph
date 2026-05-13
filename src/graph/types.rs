use thiserror::Error;

/// Stable handle for an entity inside a [`Graph`](crate::graph::Graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityRef(pub(crate) usize);

impl EntityRef {
    pub(crate) fn from_index(index: usize) -> Self {
        Self(index)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

/// Stable handle for a relationship inside a [`Graph`](crate::graph::Graph).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelationRef(pub(crate) usize);

impl RelationRef {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GraphBuildError {
    #[error("unknown entity reference {0}")]
    UnknownEntityRef(usize),

    #[error("cannot add a Chunk: the GraphBuilder has no active Source. \
             Call GraphBuilder::with_source(name) or add_source(name) first.")]
    ChunkWithoutSource,
}
