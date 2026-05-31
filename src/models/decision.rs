use serde::{Deserialize, Serialize};

/// Decision memory stores architectural reasoning:
/// why a technology was chosen, why a strategy changed, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMemory {
    pub id: String,
    pub project_id: String,
    pub title: String,
    pub context: String,
    pub rationale: String,
    pub tradeoffs: String,
    pub related_files: Vec<String>,
    pub tags: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
