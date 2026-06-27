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
    pub graph: GraphConfig,
    #[serde(default)]
    pub mcp: McpConfig,
    #[serde(default)]
    pub semantic: SemanticConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    #[serde(default = "StorageConfig::default_database_path")]
    pub database_path: PathBuf,
    #[serde(default = "StorageConfig::default_wal_mode")]
    pub wal_mode: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            database_path: Self::default_database_path(),
            wal_mode: Self::default_wal_mode(),
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

    fn default_wal_mode() -> bool {
        true
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
    #[serde(default = "RetrievalConfig::default_limit")]
    pub default_limit: usize,
    #[serde(default = "RetrievalConfig::default_fallback_timeout_ms")]
    pub fallback_timeout_ms: u64,
    #[serde(default = "RetrievalConfig::default_fallback_max_results")]
    pub fallback_max_results: usize,
    #[serde(default = "RetrievalConfig::default_recency_half_life_days")]
    pub recency_half_life_days: u64,
    /// Route `search_memory` to only the memory types implied by the classified
    /// intent (General intent still searches all four). Turn off to always query
    /// every type regardless of intent.
    #[serde(default = "RetrievalConfig::default_intent_routing")]
    pub intent_routing: bool,
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
            fallback_timeout_ms: Self::default_fallback_timeout_ms(),
            fallback_max_results: Self::default_fallback_max_results(),
            recency_half_life_days: Self::default_recency_half_life_days(),
            intent_routing: Self::default_intent_routing(),
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
    fn default_fallback_timeout_ms() -> u64 {
        50
    }
    fn default_fallback_max_results() -> usize {
        100
    }
    fn default_recency_half_life_days() -> u64 {
        30
    }
    fn default_intent_routing() -> bool {
        true
    }
    fn default_weight_relevance() -> f32 {
        0.4
    }
    fn default_weight_recency() -> f32 {
        0.2
    }
    fn default_weight_importance() -> f32 {
        0.4
    }
    fn default_weight_type() -> f32 {
        0.4
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    #[serde(default = "GraphConfig::default_max_nodes")]
    pub max_nodes: usize,
    #[serde(default = "GraphConfig::default_lazy_loading_threshold")]
    pub lazy_loading_threshold: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: Self::default_max_nodes(),
            lazy_loading_threshold: Self::default_lazy_loading_threshold(),
        }
    }
}

impl GraphConfig {
    fn default_max_nodes() -> usize {
        10_000
    }
    fn default_lazy_loading_threshold() -> usize {
        100_000
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default = "McpConfig::default_transport")]
    pub transport: String,
    #[serde(default = "McpConfig::default_worker_threads")]
    pub worker_threads: usize,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            transport: Self::default_transport(),
            worker_threads: Self::default_worker_threads(),
        }
    }
}

impl McpConfig {
    fn default_transport() -> String {
        "stdio".to_string()
    }
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
        if self.context.memory_budget_percent > 50 {
            anyhow::bail!(
                "context.memory_budget_percent must be <= 50, got {}",
                self.context.memory_budget_percent
            );
        }
        if self.retrieval.default_limit == 0 || self.retrieval.default_limit > 1000 {
            anyhow::bail!(
                "retrieval.default_limit must be between 1 and 1000, got {}",
                self.retrieval.default_limit
            );
        }
        if self.graph.max_nodes == 0 {
            anyhow::bail!("graph.max_nodes must be > 0, got {}", self.graph.max_nodes);
        }
        Ok(())
    }
}
