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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    #[serde(default = "RetrievalConfig::default_limit")]
    pub default_limit: usize,
    #[serde(default = "RetrievalConfig::default_fallback_timeout_ms")]
    pub fallback_timeout_ms: u64,
    #[serde(default = "RetrievalConfig::default_fallback_max_results")]
    pub fallback_max_results: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            default_limit: Self::default_limit(),
            fallback_timeout_ms: Self::default_fallback_timeout_ms(),
            fallback_max_results: Self::default_fallback_max_results(),
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextConfig {
    #[serde(default = "ContextConfig::default_memory_budget_percent")]
    pub memory_budget_percent: u8,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            memory_budget_percent: Self::default_memory_budget_percent(),
        }
    }
}

impl ContextConfig {
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
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            transport: Self::default_transport(),
        }
    }
}

impl McpConfig {
    fn default_transport() -> String {
        "stdio".to_string()
    }
}

impl Config {
    /// Load configuration from a TOML file, falling back to defaults for missing fields.
    pub fn load_from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration with priority: CLI > env vars > config file > defaults.
    /// For MVP, this supports config file and defaults only.
    pub fn load() -> anyhow::Result<Self> {
        let config_path = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".engram")
            .join("config.toml");

        if config_path.exists() {
            Self::load_from_file(&config_path)
        } else {
            Ok(Self::default())
        }
    }
}
