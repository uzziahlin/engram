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

impl FailureMemory {
    /// See `EpisodicMemory::embedding_text`.
    pub fn embedding_text(&self) -> String {
        format!("{}\n{}", self.incident, self.root_cause)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_text_joins_incident_and_root_cause() {
        let m = FailureMemory {
            id: "i".into(),
            project_id: "p".into(),
            incident: "search crash".into(),
            root_cause: "fts5 column filter".into(),
            fix: "f".into(),
            prevention: "pr".into(),
            severity: 3,
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        };
        assert_eq!(m.embedding_text(), "search crash\nfts5 column filter");
    }
}
