use crate::models::MemoryIntent;

/// Global ranking-signal weights, sourced from `[retrieval]` config.
///
/// `plan()` uses these as the base values and only ever raises a weight
/// (max-increment) per intent, so a single intent reproduces the prior
/// hard-coded constants and compound intents keep the strongest signal
/// instead of last-wins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanWeights {
    pub relevance: f32,
    pub recency: f32,
    pub importance: f32,
    pub type_weight: f32,
}

impl Default for PlanWeights {
    /// Defaults reproduce the pre-config hard-coded values (no-op when the
    /// `[retrieval]` section is absent).
    fn default() -> Self {
        Self {
            relevance: 0.4,
            recency: 0.2,
            importance: 0.4,
            type_weight: 0.4,
        }
    }
}

/// Retrieval plan defining which memory sources to query and ranking weights.
#[derive(Debug, Clone)]
pub struct RetrievalPlan {
    pub sources: Vec<MemorySource>,
    pub relevance_weight: f32,
    pub recency_weight: f32,
    pub importance_weight: f32,
    pub type_weight: f32,
}

/// Memory sources available for retrieval.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemorySource {
    Episodic,
    Decision,
    Failure,
    Procedural,
}

impl MemorySource {
    /// Map to the `memory_type` string used across storage / FTS
    /// (`"episodic"`, `"decision"`, `"failure"`, `"procedural"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            MemorySource::Episodic => "episodic",
            MemorySource::Decision => "decision",
            MemorySource::Failure => "failure",
            MemorySource::Procedural => "procedural",
        }
    }

    /// All four sources, in canonical order.
    pub fn all() -> &'static [MemorySource] {
        &[
            MemorySource::Episodic,
            MemorySource::Decision,
            MemorySource::Failure,
            MemorySource::Procedural,
        ]
    }
}

/// Retrieval planner that selects memory sources and ranking weights based on intent.
pub struct RetrievalPlanner {
    base: PlanWeights,
}

impl RetrievalPlanner {
    pub fn new(base: PlanWeights) -> Self {
        Self { base }
    }

    /// Generate a retrieval plan based on classified intents.
    pub fn plan(&self, intents: &[MemoryIntent]) -> RetrievalPlan {
        let mut sources = Vec::new();
        // Start from configured base weights. Each intent only ever raises a
        // weight (max), so a single intent reproduces the prior hard-coded
        // value and compound intents keep the strongest signal instead of
        // last-wins overwriting earlier intents.
        let mut recency_weight = self.base.recency;
        let mut importance_weight = self.base.importance;
        let mut type_weight = self.base.type_weight;

        for intent in intents {
            match intent {
                MemoryIntent::Debugging => {
                    if !sources.contains(&MemorySource::Failure) {
                        sources.push(MemorySource::Failure);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    importance_weight = importance_weight.max(0.6);
                    type_weight = type_weight.max(0.8);
                }
                MemoryIntent::Architecture => {
                    if !sources.contains(&MemorySource::Decision) {
                        sources.push(MemorySource::Decision);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    importance_weight = importance_weight.max(0.6);
                    type_weight = type_weight.max(0.8);
                }
                MemoryIntent::Workflow => {
                    if !sources.contains(&MemorySource::Procedural) {
                        sources.push(MemorySource::Procedural);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    recency_weight = recency_weight.max(0.3);
                }
                MemoryIntent::Refactor => {
                    if !sources.contains(&MemorySource::Decision) {
                        sources.push(MemorySource::Decision);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    type_weight = type_weight.max(0.6);
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
                    recency_weight = recency_weight.max(0.4);
                }
                MemoryIntent::Incident => {
                    if !sources.contains(&MemorySource::Failure) {
                        sources.push(MemorySource::Failure);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    importance_weight = importance_weight.max(0.7);
                    type_weight = type_weight.max(0.9);
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
            relevance_weight: self.base.relevance,
            recency_weight,
            importance_weight,
            type_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugging_plan() {
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::Debugging]);
        assert!(plan.sources.contains(&MemorySource::Failure));
        assert!(plan.sources.contains(&MemorySource::Episodic));
        assert!(plan.importance_weight > 0.5);
    }

    #[test]
    fn test_architecture_plan() {
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::Architecture]);
        assert!(plan.sources.contains(&MemorySource::Decision));
    }

    #[test]
    fn test_general_plan_searches_all() {
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::General]);
        assert_eq!(plan.sources.len(), 4);
    }

    #[test]
    fn test_compound_intent_plan() {
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::Debugging, MemoryIntent::Incident]);
        assert!(plan.sources.contains(&MemorySource::Failure));
        assert!(plan.type_weight > 0.7);
    }

    #[test]
    fn test_compound_intent_takes_max_weight() {
        // Debugging sets importance=0.6/type=0.8; Incident sets importance=0.7/type=0.9.
        // Compound intent must keep the STRONGEST signal (max), not last-wins.
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::Debugging, MemoryIntent::Incident]);
        assert!(
            (plan.importance_weight - 0.7).abs() < 1e-6,
            "importance must be max(0.6, 0.7) = 0.7, got {}",
            plan.importance_weight
        );
        assert!(
            (plan.type_weight - 0.9).abs() < 1e-6,
            "type must be max(0.8, 0.9) = 0.9, got {}",
            plan.type_weight
        );
        // Single intent reproduces its target value (max against the lower base).
        let plan = planner.plan(&[MemoryIntent::Debugging]);
        assert!((plan.importance_weight - 0.6).abs() < 1e-6);
        assert!((plan.relevance_weight - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_base_weights_come_from_config() {
        // Custom base weights propagate as the floor; an intent's target still
        // wins when it exceeds the base, otherwise the base is kept.
        let planner = RetrievalPlanner::new(PlanWeights {
            relevance: 0.5,
            recency: 0.3,
            importance: 0.5,
            type_weight: 0.5,
        });
        let plan = planner.plan(&[MemoryIntent::Debugging]);
        assert!((plan.relevance_weight - 0.5).abs() < 1e-6);
        // Debugging targets importance 0.6 > base 0.5 → 0.6.
        assert!((plan.importance_weight - 0.6).abs() < 1e-6);
        // Debugging targets type 0.8 > base 0.5 → 0.8.
        assert!((plan.type_weight - 0.8).abs() < 1e-6);
    }
}
