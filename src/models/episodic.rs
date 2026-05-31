use serde::{Deserialize, Serialize};

/// Episodic memory stores task history: debugging sessions, refactors,
/// migrations, deployments, feature implementations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    pub id: String,
    pub project_id: String,
    pub session_id: String,
    pub summary: String,
    pub content: String,
    pub files_touched: Vec<String>,
    pub related_commits: Vec<String>,
    pub importance: f32,
    pub tags: Vec<String>,
    pub created_at: i64,
    pub updated_at: i64,
}
