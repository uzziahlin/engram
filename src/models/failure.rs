use serde::{Deserialize, Serialize};

/// Failure memory stores incidents and debugging patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureMemory {
    pub id: String,
    pub project_id: String,
    pub incident: String,
    pub root_cause: String,
    pub fix: String,
    pub prevention: String,
    pub severity: u8,
    pub tags: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
