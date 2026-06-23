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

impl ProceduralMemory {
    /// See `EpisodicMemory::embedding_text`.
    pub fn embedding_text(&self) -> String {
        format!("{}\n{}", self.workflow_name, self.steps.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_text_joins_workflow_name_and_steps() {
        let m = ProceduralMemory {
            id: "i".into(),
            project_id: "p".into(),
            workflow_name: "release".into(),
            steps: vec!["build".into(), "tag".into(), "publish".into()],
            related_tools: vec![],
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        };
        assert_eq!(m.embedding_text(), "release\nbuild tag publish");
    }
}
