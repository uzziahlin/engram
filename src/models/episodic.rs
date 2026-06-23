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

impl EpisodicMemory {
    /// Text used to compute this memory's embedding. Single source of truth for
    /// both the write path and `reindex`; changing it changes stored vectors.
    pub fn embedding_text(&self) -> String {
        format!("{}\n{}", self.summary, self.content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_text_joins_summary_and_content() {
        let m = EpisodicMemory {
            id: "i".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "OAuth token refresh loop".into(),
            content: "credential renewal cycle".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        };
        assert_eq!(
            m.embedding_text(),
            "OAuth token refresh loop\ncredential renewal cycle"
        );
    }
}
