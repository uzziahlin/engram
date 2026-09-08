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
    /// Defaults sum to 1.0, matching `[retrieval]` config defaults. Relevance
    /// (the only query-dependent signal) dominates; the static priors only
    /// break ties. `plan()` renormalizes after per-intent adjustments, so
    /// custom config values may use any positive scale.
    fn default() -> Self {
        Self {
            relevance: 0.5,
            recency: 0.15,
            importance: 0.2,
            type_weight: 0.15,
        }
    }
}

impl PlanWeights {
    /// Normalize so the four weights sum to 1. No-op for the defaults;
    /// keeps user config on any positive scale meaningful.
    fn normalized(self) -> Self {
        let sum = self.relevance + self.recency + self.importance + self.type_weight;
        if sum <= 0.0 || !sum.is_finite() {
            return Self::default();
        }
        Self {
            relevance: self.relevance / sum,
            recency: self.recency / sum,
            importance: self.importance / sum,
            type_weight: self.type_weight / sum,
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
        // weight (max), so a single intent reproduces its target value and
        // compound intents keep the strongest signal instead of last-wins.
        //
        // Intents raise RELEVANCE (and sometimes recency) — the classified
        // intent means the query terms matched that domain, which is evidence
        // the BM25 score is meaningful. The type weight is NEVER raised: the
        // static type prior already favors failure/decision records
        // unconditionally, and amplifying a query-independent prior under
        // intent is exactly the inversion the 2026-09 review flagged (intents
        // used to raise only the priors, letting them outvote relevance).
        let mut relevance_weight = self.base.relevance;
        let mut recency_weight = self.base.recency;
        // Per-record importance and the type prior are never intent-adjusted:
        // they are properties of the record, not of the query.
        let importance_weight = self.base.importance;
        let type_weight = self.base.type_weight;

        for intent in intents {
            match intent {
                MemoryIntent::Debugging => {
                    if !sources.contains(&MemorySource::Failure) {
                        sources.push(MemorySource::Failure);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    relevance_weight = relevance_weight.max(0.6);
                }
                MemoryIntent::Architecture => {
                    if !sources.contains(&MemorySource::Decision) {
                        sources.push(MemorySource::Decision);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    relevance_weight = relevance_weight.max(0.6);
                }
                MemoryIntent::Workflow => {
                    if !sources.contains(&MemorySource::Procedural) {
                        sources.push(MemorySource::Procedural);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    relevance_weight = relevance_weight.max(0.55);
                    recency_weight = recency_weight.max(0.25);
                }
                MemoryIntent::Refactor => {
                    if !sources.contains(&MemorySource::Decision) {
                        sources.push(MemorySource::Decision);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    relevance_weight = relevance_weight.max(0.55);
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
                    relevance_weight = relevance_weight.max(0.55);
                    recency_weight = recency_weight.max(0.3);
                }
                MemoryIntent::Incident => {
                    if !sources.contains(&MemorySource::Failure) {
                        sources.push(MemorySource::Failure);
                    }
                    if !sources.contains(&MemorySource::Episodic) {
                        sources.push(MemorySource::Episodic);
                    }
                    relevance_weight = relevance_weight.max(0.65);
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

        // Renormalize after intent adjustments so the final weights always
        // sum to 1 — comparable across intents and directly interpretable
        // as "share of the final score".
        let w = PlanWeights {
            relevance: relevance_weight,
            recency: recency_weight,
            importance: importance_weight,
            type_weight,
        }
        .normalized();

        RetrievalPlan {
            sources,
            relevance_weight: w.relevance,
            recency_weight: w.recency,
            importance_weight: w.importance,
            type_weight: w.type_weight,
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
        // Debugging raises relevance (the query matched debugging terms).
        assert!(plan.relevance_weight > PlanWeights::default().relevance);
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
        // Incident raises relevance more than Debugging; max wins.
        assert!(plan.relevance_weight > PlanWeights::default().relevance);
        // The type weight's share never grows under intent (static priors
        // must not be amplified — the relevance raise takes the share).
        assert!(plan.type_weight <= PlanWeights::default().type_weight + 1e-6);
    }

    #[test]
    fn test_weights_always_normalized() {
        // After any intent adjustment the four weights must sum to 1 so the
        // final score is interpretable and comparable across intents.
        let planner = RetrievalPlanner::new(PlanWeights::default());
        for intents in [
            vec![MemoryIntent::General],
            vec![MemoryIntent::Debugging],
            vec![MemoryIntent::Incident],
            vec![MemoryIntent::Deployment, MemoryIntent::Workflow],
        ] {
            let plan = planner.plan(&intents);
            let sum = plan.relevance_weight
                + plan.recency_weight
                + plan.importance_weight
                + plan.type_weight;
            assert!((sum - 1.0).abs() < 1e-5, "weights must sum to 1, got {sum}");
        }
    }

    #[test]
    fn test_intent_raises_relevance_not_just_priors() {
        // The 2026-09 review found intents only raised the query-INDEPENDENT
        // priors (importance/type/recency), letting static priors dominate
        // ranking. Intents must now raise relevance first.
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let base = PlanWeights::default();
        for intents in [
            vec![MemoryIntent::Debugging],
            vec![MemoryIntent::Architecture],
            vec![MemoryIntent::Incident],
        ] {
            let plan = planner.plan(&intents);
            assert!(
                plan.relevance_weight > base.relevance,
                "{intents:?} must raise relevance above base {}",
                base.relevance
            );
        }
    }

    #[test]
    fn test_custom_base_weights_normalized() {
        // Custom base weights on any scale are renormalized to sum 1.
        let planner = RetrievalPlanner::new(PlanWeights {
            relevance: 2.0,
            recency: 1.0,
            importance: 1.0,
            type_weight: 0.0,
        });
        let plan = planner.plan(&[MemoryIntent::General]);
        assert!((plan.relevance_weight - 0.5).abs() < 1e-6);
        assert!((plan.type_weight - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_relevance_dominates_type_prior() {
        // Even under the strongest type-raising intent (Incident), the max
        // possible type-weight contribution must stay well below the max
        // relevance contribution, so a strong BM25 match on the "wrong" type
        // still outranks a weak match on the "right" type.
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::Incident]);
        assert!(
            plan.relevance_weight > 2.0 * plan.type_weight,
            "relevance weight {} must dominate type weight {}",
            plan.relevance_weight,
            plan.type_weight
        );
    }
}
