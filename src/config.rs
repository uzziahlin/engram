use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level configuration for the engram memory runtime.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub semantic: SemanticConfig,
    #[serde(default)]
    pub reflection: ReflectionConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "StorageConfig::default_database_path")]
    pub database_path: PathBuf,
    /// query_log rows older than this are pruned by `engram maintain`.
    /// query_log grows by one row per search and has no natural bound.
    #[serde(default = "StorageConfig::default_query_log_retention_days")]
    pub query_log_retention_days: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: Self::default_database_path(),
            query_log_retention_days: Self::default_query_log_retention_days(),
        }
    }
}

impl StorageConfig {
    fn default_database_path() -> PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".engram")
            .join("memory.db")
    }

    fn default_query_log_retention_days() -> u64 {
        90
    }
}

/// Expand a leading `~` to the user's home directory.
///
/// Handles both `~` and `~/path` forms; any other input is returned unchanged.
/// Does not expand `~user` (another user's home) — that is out of scope and
/// not exposed by the `dirs` crate.
fn expand_tilde(path: &std::path::Path) -> PathBuf {
    let Some(s) = path.to_str() else {
        return path.to_path_buf();
    };
    if s == "~" {
        return dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    }
    if let Some(rest) = s.strip_prefix("~/") {
        return dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(rest);
    }
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_tilde_expands_home_prefix() {
        let home = dirs::home_dir().expect("home dir should be available in test env");
        assert_eq!(expand_tilde(std::path::Path::new("~")), home);
        assert_eq!(
            expand_tilde(std::path::Path::new("~/.engram/memory.db")),
            home.join(".engram/memory.db")
        );
    }

    #[test]
    fn expand_tilde_passes_through_absolute_and_relative_paths() {
        assert_eq!(
            expand_tilde(std::path::Path::new("/var/data/memory.db")),
            PathBuf::from("/var/data/memory.db")
        );
        assert_eq!(
            expand_tilde(std::path::Path::new("relative/path.db")),
            PathBuf::from("relative/path.db")
        );
    }

    #[test]
    fn load_from_file_expands_tilde_in_database_path() {
        let dir = std::env::temp_dir();
        let config_path = dir.join("engram_tilde_test_config.toml");
        std::fs::write(
            &config_path,
            "[storage]\ndatabase_path = \"~/.engram/memory.db\"\n",
        )
        .unwrap();
        let config = Config::load_from_file(&config_path).expect("config should load");
        let _ = std::fs::remove_file(&config_path);

        let home = dirs::home_dir().expect("home dir should be available in test env");
        assert_eq!(
            config.storage.database_path,
            home.join(".engram").join("memory.db"),
            "tilde must be expanded to an absolute home path"
        );
    }

    #[test]
    fn retrieval_half_life_defaults_to_30_days() {
        assert_eq!(RetrievalConfig::default().recency_half_life_days, 30);
    }

    #[test]
    fn mcp_worker_threads_defaults_to_1() {
        // Default 1 = sequential request processing (safe for stdio MCP where
        // a single client pipelines dependent requests, e.g. create-then-search).
        // Raising worker_threads opts into concurrent processing.
        assert_eq!(McpConfig::default().worker_threads, 1);
    }

    #[test]
    fn semantic_defaults_off_with_minilm() {
        let c = SemanticConfig::default();
        assert!(!c.enabled);
        assert_eq!(c.model_id, "sentence-transformers/all-MiniLM-L6-v2");
        assert_eq!(c.rrf_k, 60.0);
        assert_eq!(c.top_k, 50);
        assert!(c.model_path.is_none());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    /// Default `limit` for search/list tools when the caller omits it.
    #[serde(default = "RetrievalConfig::default_limit")]
    pub default_limit: usize,
    #[serde(default = "RetrievalConfig::default_recency_half_life_days")]
    pub recency_half_life_days: u64,
    /// Intent-aware ranking: classified intent keywords adjust the reranker's
    /// type/recency/importance weights. Every type is always searched — routing
    /// is soft, it never narrows the sources (recall first).
    #[serde(default = "RetrievalConfig::default_intent_routing")]
    pub intent_routing: bool,
    /// Graph-based second-tier retrieval: memories sharing an entity (file/
    /// tool) with the top search hits are appended as low-rank context.
    #[serde(default = "RetrievalConfig::default_graph_expansion")]
    pub graph_expansion: bool,
    /// Global ranking-signal weights. Defaults reproduce the pre-config
    /// hard-coded values, so an absent `[retrieval]` section is a no-op.
    /// `weight_relevance` scales the BM25 score; the other three are the base
    /// values the per-intent planner adjusts (max-increments) on top of.
    #[serde(default = "RetrievalConfig::default_weight_relevance")]
    pub weight_relevance: f32,
    #[serde(default = "RetrievalConfig::default_weight_recency")]
    pub weight_recency: f32,
    #[serde(default = "RetrievalConfig::default_weight_importance")]
    pub weight_importance: f32,
    #[serde(default = "RetrievalConfig::default_weight_type")]
    pub weight_type: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_limit: Self::default_limit(),
            recency_half_life_days: Self::default_recency_half_life_days(),
            intent_routing: Self::default_intent_routing(),
            graph_expansion: Self::default_graph_expansion(),
            weight_relevance: Self::default_weight_relevance(),
            weight_recency: Self::default_weight_recency(),
            weight_importance: Self::default_weight_importance(),
            weight_type: Self::default_weight_type(),
        }
    }
}

impl RetrievalConfig {
    fn default_limit() -> usize {
        10
    }
    fn default_recency_half_life_days() -> u64 {
        30
    }
    fn default_intent_routing() -> bool {
        true
    }
    fn default_graph_expansion() -> bool {
        true
    }
    // Defaults sum to 1.0: relevance is the dominant signal (the query terms
    // matched), the static priors only break ties. The planner renormalizes
    // after per-intent adjustments, so custom values may use any scale.
    fn default_weight_relevance() -> f32 {
        0.5
    }
    fn default_weight_recency() -> f32 {
        0.15
    }
    fn default_weight_importance() -> f32 {
        0.2
    }
    fn default_weight_type() -> f32 {
        0.15
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    #[serde(default = "ContextConfig::default_context_window_tokens")]
    pub context_window_tokens: usize,
    #[serde(default = "ContextConfig::default_memory_budget_percent")]
    pub memory_budget_percent: u8,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            context_window_tokens: Self::default_context_window_tokens(),
            memory_budget_percent: Self::default_memory_budget_percent(),
        }
    }
}

impl ContextConfig {
    fn default_context_window_tokens() -> usize {
        200_000
    }
    fn default_memory_budget_percent() -> u8 {
        15
    }
}

/// MCP server config. stdio is the only transport implemented; `transport`
/// was removed after lingering as a dead knob — re-add it together with an
/// actual HTTP transport, not before.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default = "McpConfig::default_worker_threads")]
    pub worker_threads: usize,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            worker_threads: Self::default_worker_threads(),
        }
    }
}

impl McpConfig {
    fn default_worker_threads() -> usize {
        1
    }
}

/// Semantic / embedding retrieval config. Disabled by default to keep the
/// release binary self-contained; only takes effect with `--features semantic`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "SemanticConfig::default_model_id")]
    pub model_id: String,
    /// Override model directory (air-gapped / user-provided). When None, the
    /// model is fetched on first run into `~/.engram/models/<model_id>`.
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default = "SemanticConfig::default_rrf_k")]
    pub rrf_k: f32,
    #[serde(default = "SemanticConfig::default_top_k")]
    pub top_k: usize,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model_id: Self::default_model_id(),
            model_path: None,
            rrf_k: Self::default_rrf_k(),
            top_k: Self::default_top_k(),
        }
    }
}

impl SemanticConfig {
    fn default_model_id() -> String {
        "sentence-transformers/all-MiniLM-L6-v2".to_string()
    }
    fn default_rrf_k() -> f32 {
        60.0
    }
    fn default_top_k() -> usize {
        50
    }
}

/// Reflection engine config. The reflection pass scans active failures, groups
/// them by tag, and proposes a preventive procedural rule whenever a tag recurs
/// at least `min_occurrences` times.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionConfig {
    /// Minimum active failures sharing a tag before a preventive rule is
    /// proposed. Default 3.
    #[serde(default = "ReflectionConfig::default_min_occurrences")]
    pub min_occurrences: usize,
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self {
            min_occurrences: Self::default_min_occurrences(),
        }
    }
}

impl ReflectionConfig {
    fn default_min_occurrences() -> usize {
        3
    }
}

/// Filesystem-access guard config. `repo_path` arguments come from the MCP
/// client (ultimately the agent, which can be steered by prompt injection);
/// see `src/path_guard.rs` for the enforcement rules.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// When non-empty, `ingest_commits`/`collect_sources` only accept
    /// `repo_path` values under one of these roots (canonical comparison).
    /// Empty (default) = no allowlist; the home directory and filesystem
    /// root are rejected unconditionally either way.
    #[serde(default)]
    pub allowed_roots: Vec<PathBuf>,
}

impl Config {
    /// Load configuration from a TOML file, falling back to defaults for missing fields.
    pub fn load_from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&content)?;
        // User-facing config paths may use the `~` home shorthand (e.g. "~/.engram/memory.db").
        // `~` is a shell convention, not an OS path feature — if left unexpanded, SQLite treats
        // it as a relative path and creates a literal `~/` directory under the process cwd.
        // Expand it at the config boundary so the rest of the system sees an absolute path.
        config.storage.database_path = expand_tilde(&config.storage.database_path);
        config.security.allowed_roots = config
            .security
            .allowed_roots
            .iter()
            .map(|p| expand_tilde(p))
            .collect();
        // Validate on every load path, not just `load()` — embedders and tests
        // that use `load_from_file` directly must not bypass range checks.
        config.validate()?;
        Ok(config)
    }

    /// Load configuration with priority: CLI > env vars > config file > defaults.
    /// For MVP, this supports config file and defaults only.
    pub fn load() -> anyhow::Result<Self> {
        let config_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".engram")
            .join("config.toml");

        if !config_path.exists() {
            return Ok(Self::default());
        }

        let config = Self::load_from_file(&config_path)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration fields are within acceptable ranges.
    fn validate(&self) -> anyhow::Result<()> {
        if self.context.memory_budget_percent == 0 || self.context.memory_budget_percent > 50 {
            anyhow::bail!(
                "context.memory_budget_percent must be between 1 and 50, got {}",
                self.context.memory_budget_percent
            );
        }
        if self.context.context_window_tokens < 1000 {
            anyhow::bail!(
                "context.context_window_tokens must be >= 1000, got {}",
                self.context.context_window_tokens
            );
        }
        if self.retrieval.default_limit == 0 || self.retrieval.default_limit > 1000 {
            anyhow::bail!(
                "retrieval.default_limit must be between 1 and 1000, got {}",
                self.retrieval.default_limit
            );
        }
        // Weight sanity: finite, non-negative, not all-zero. NaN would poison
        // every partial_cmp in the reranker (NaN == NaN compares Equal), and a
        // negative weight silently inverts the ranking.
        let r = &self.retrieval;
        for (name, v) in [
            ("weight_relevance", r.weight_relevance),
            ("weight_recency", r.weight_recency),
            ("weight_importance", r.weight_importance),
            ("weight_type", r.weight_type),
        ] {
            if !v.is_finite() {
                anyhow::bail!("retrieval.{name} must be a finite number, got {v}");
            }
            if v < 0.0 {
                anyhow::bail!("retrieval.{name} must be >= 0, got {v}");
            }
        }
        if r.weight_relevance + r.weight_recency + r.weight_importance + r.weight_type <= 0.0 {
            anyhow::bail!(
                "retrieval weight_* must not all be zero (sum = {})",
                r.weight_relevance + r.weight_recency + r.weight_importance + r.weight_type
            );
        }
        if self.mcp.worker_threads == 0 || self.mcp.worker_threads > 64 {
            anyhow::bail!(
                "mcp.worker_threads must be between 1 and 64, got {}",
                self.mcp.worker_threads
            );
        }
        if self.reflection.min_occurrences == 0 {
            anyhow::bail!("reflection.min_occurrences must be >= 1, got 0",);
        }
        if self.semantic.enabled || cfg!(feature = "semantic") {
            if !self.semantic.rrf_k.is_finite() || self.semantic.rrf_k <= 0.0 {
                anyhow::bail!(
                    "semantic.rrf_k must be a positive number, got {}",
                    self.semantic.rrf_k
                );
            }
            if self.semantic.top_k == 0 {
                anyhow::bail!("semantic.top_k must be >= 1, got 0");
            }
        }
        Ok(())
    }
}
