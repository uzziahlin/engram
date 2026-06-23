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

impl DecisionMemory {
    /// See `EpisodicMemory::embedding_text`.
    pub fn embedding_text(&self) -> String {
        format!("{}\n{}", self.title, self.rationale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_text_joins_title_and_rationale() {
        let m = DecisionMemory {
            id: "i".into(),
            project_id: "p".into(),
            title: "pick candle".into(),
            context: "ctx".into(),
            rationale: "pure rust offline".into(),
            tradeoffs: "to".into(),
            related_files: vec![],
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        };
        assert_eq!(m.embedding_text(), "pick candle\npure rust offline");
    }
}
