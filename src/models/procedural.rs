use serde::{Deserialize, Serialize};

/// Procedural memory stores workflows and coding conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralMemory {
    pub id: String,
    pub project_id: String,
    pub workflow_name: String,
    pub steps: Vec<String>,
    pub related_tools: Vec<String>,
    pub tags: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
