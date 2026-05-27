//! Load + validate configuration. Supports environment overrides of the form
//! `LINGUAGRAPH__SECTION__FIELD=value` (double underscore separator).

use std::path::Path;

use thiserror::Error;
use tokio::fs;

use super::Config;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("I/O error reading config: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid TOML: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("invalid value for `{key}`: {message}")]
    InvalidValue { key: String, message: String },
}

/// Load config from a TOML file, then apply environment overrides.
pub async fn load(path: &Path) -> Result<Config, ConfigError> {
    let raw = fs::read_to_string(path).await?;
    let mut cfg: Config = load_from_str(&raw)?;
    apply_env_overrides(&mut cfg);
    validate(&cfg)?;
    Ok(cfg)
}

pub fn load_from_str(raw: &str) -> Result<Config, ConfigError> {
    let cfg: Config = toml::from_str(raw)?;
    Ok(cfg)
}

fn apply_env_overrides(cfg: &mut Config) {
    if let Ok(v) = std::env::var("LINGUAGRAPH__DATABASE__URI") {
        cfg.database.uri = v;
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__DATABASE__USER") {
        cfg.database.user = v;
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__DATABASE__PASSWORD") {
        cfg.database.password = v;
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__DATABASE__MAX_CONNECTIONS") {
        if let Ok(n) = v.parse() {
            cfg.database.max_connections = n;
        }
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__LLM__MODEL") {
        cfg.llm.model = v;
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__LLM__PROVIDER") {
        cfg.llm.provider = v;
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__QUERY__MAX_TRAVERSAL_DEPTH") {
        if let Ok(n) = v.parse() {
            cfg.query.max_traversal_depth = n;
        }
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__QUERY__DEFAULT_LIMIT") {
        if let Ok(n) = v.parse() {
            cfg.query.default_limit = n;
        }
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__PROMPT__ONTOLOGIES_PATH") {
        cfg.prompt.ontologies_path = Some(v);
    }
    if let Ok(v) = std::env::var("LINGUAGRAPH__PROMPT__DEFAULT_DOMAIN") {
        cfg.prompt.default_domain = Some(v);
    }
}

fn validate(cfg: &Config) -> Result<(), ConfigError> {
    if cfg.database.uri.is_empty() {
        return Err(ConfigError::InvalidValue {
            key: "database.uri".into(),
            message: "must not be empty".into(),
        });
    }
    if cfg.database.max_connections == 0 {
        return Err(ConfigError::InvalidValue {
            key: "database.max_connections".into(),
            message: "must be > 0".into(),
        });
    }
    if cfg.query.max_traversal_depth == 0 {
        return Err(ConfigError::InvalidValue {
            key: "query.max_traversal_depth".into(),
            message: "must be > 0".into(),
        });
    }
    if !(0.0..=2.0).contains(&cfg.llm.temperature) {
        return Err(ConfigError::InvalidValue {
            key: "llm.temperature".into(),
            message: "must be in [0.0, 2.0]".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_config() {
        let toml = r#"
            [database]
            uri = "bolt://localhost:7687"
            user = "memgraph"
            password = "memgraph"
        "#;
        let cfg = load_from_str(toml).unwrap();
        assert_eq!(cfg.database.uri, "bolt://localhost:7687");
        assert_eq!(cfg.database.max_connections, 16);
        assert_eq!(cfg.query.default_limit, 100);
        assert_eq!(cfg.graph_specification.embedding_model, None);
        assert_eq!(cfg.graph_specification.reranking_model, None);
        assert_eq!(cfg.graph_specification.embedding_dim, 384);
        assert_eq!(cfg.graph_specification.reranking_threshold, 0.3);
    }

    #[test]
    fn parses_graph_specification_embedding_config() {
        let toml = r#"
            [database]
            uri = "bolt://localhost:7687"
            user = "memgraph"
            password = "memgraph"

            [graph_specification]
            embedding_model = "models/spec-embed.gguf"
            reranking_model = "models/spec-rerank.gguf"
            embedding_dim = 768
            reranking_threshold = 0.41
        "#;
        let cfg = load_from_str(toml).unwrap();
        assert_eq!(
            cfg.graph_specification.embedding_model.as_deref(),
            Some("models/spec-embed.gguf")
        );
        assert_eq!(
            cfg.graph_specification.reranking_model.as_deref(),
            Some("models/spec-rerank.gguf")
        );
        assert_eq!(cfg.graph_specification.embedding_dim, 768);
        assert_eq!(cfg.graph_specification.reranking_threshold, 0.41);
    }

    #[test]
    fn rejects_empty_uri() {
        let mut cfg = load_from_str(
            r#"
            [database]
            uri = "bolt://x"
            user = "x"
            password = "x"
        "#,
        )
        .unwrap();
        cfg.database.uri.clear();
        assert!(validate(&cfg).is_err());
    }
}
