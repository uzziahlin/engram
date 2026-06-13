use crate::models::MemoryIntent;

/// Retrieval plan defining which memory sources to query and ranking weights.
#[derive(Debug, Clone)]
pub struct RetrievalPlan {
    pub sources: Vec<MemorySource>,
    pub recency_weight: f32,
    pub importance_weight: f32,
    pub graph_weight: f32,
}

/// Memory sources available for retrieval.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemorySource {
    Episodic,
    Decision,
    Failure,
    Procedural,
}

/// Retrieval planner that selects memory sources and ranking weights based on intent.
pub struct RetrievalPlanner;

impl Default for RetrievalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl RetrievalPlanner {
    pub fn new() -> Self {
        Self
    }

    /// Generate a retrieval plan based on classified intents.
    pub fn plan(&self, intents: &[MemoryIntent]) -> RetrievalPlan {
        let mut sources = Vec::new();
        let mut recency_weight = 0.2;
        let mut importance_weight = 0.4;
        let mut graph_weight = 0.4;

        for intent in intents {
            match intent {
                MemoryIntent::Debugging => {
                    if !sources.contains(&MemorySource::Failure) {
                        sources.push(MemorySource::Failure);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    importance_weight = 0.6;
                    graph_weight = 0.8;
                }
                MemoryIntent::Architecture => {
                    if !sources.contains(&MemorySource::Decision) {
                        sources.push(MemorySource::Decision);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    importance_weight = 0.6;
                    graph_weight = 0.8;
                }
                MemoryIntent::Workflow => {
                    if !sources.contains(&MemorySource::Procedural) {
                        sources.push(MemorySource::Procedural);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    recency_weight = 0.3;
                }
                MemoryIntent::Refactor => {
                    if !sources.contains(&MemorySource::Decision) {
                        sources.push(MemorySource::Decision);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    graph_weight = 0.6;
                }
                MemoryIntent::Deployment => {
                    if !sources.contains(&MemorySource::Procedural) {
                        sources.push(MemorySource::Procedural);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    if !sources.contains(&MemorySource::Failure) {
                        sources.push(MemorySource::Failure);
                    }
                    recency_weight = 0.4;
                }
                MemoryIntent::Incident => {
                    if !sources.contains(&MemorySource::Failure) {
                        sources.push(MemorySource::Failure);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    importance_weight = 0.7;
                    graph_weight = 0.9;
                }
                MemoryIntent::General => {
                    // General queries search all sources
                    sources = vec![
                        MemorySource::Episodic,
                        MemorySource::Decision,
                        MemorySource::Failure,
                        MemorySource::Procedural,
                    ];
                }
            }
        }

        // Deduplicate sources
        sources.sort();
        sources.dedup();

        RetrievalPlan {
            sources,
            recency_weight,
            importance_weight,
            graph_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugging_plan() {
        let planner = RetrievalPlanner::new();
        let plan = planner.plan(&[MemoryIntent::Debugging]);
        assert!(plan.sources.contains(&MemorySource::Failure));
        assert!(plan.sources.contains(&MemorySource::Episodic));
        assert!(plan.importance_weight > 0.5);
    }

    #[test]
    fn test_architecture_plan() {
        let planner = RetrievalPlanner::new();
        let plan = planner.plan(&[MemoryIntent::Architecture]);
        assert!(plan.sources.contains(&MemorySource::Decision));
    }

    #[test]
    fn test_general_plan_searches_all() {
        let planner = RetrievalPlanner::new();
        let plan = planner.plan(&[MemoryIntent::General]);
        assert_eq!(plan.sources.len(), 4);
    }

    #[test]
    fn test_compound_intent_plan() {
        let planner = RetrievalPlanner::new();
        let plan = planner.plan(&[MemoryIntent::Debugging, MemoryIntent::Incident]);
        assert!(plan.sources.contains(&MemorySource::Failure));
        assert!(plan.graph_weight > 0.7);
    }
}
