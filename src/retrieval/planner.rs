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

/// Retrieval plan: ranking weights adjusted for the classified intent.
///
/// There is deliberately NO source-routing field anymore: the computed
/// "which memory types to query" was never consumed (search always fans out
/// to all four types and the intent only adjusts weights) — a soft-weight
/// design chosen because hard narrowing cost recall. Keeping the dead field
/// invited readers to believe routing existed.
#[derive(Debug, Clone)]
pub struct RetrievalPlan {
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

    /// Parse the `memory_type` string from MCP/CLI input. NOT named
    /// `from_str` on purpose: avoids clippy `should_implement_trait`.
    pub fn from_type_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "episodic" => Ok(MemorySource::Episodic),
            "decision" => Ok(MemorySource::Decision),
            "failure" => Ok(MemorySource::Failure),
            "procedural" => Ok(MemorySource::Procedural),
            other => anyhow::bail!(
                "unknown memory_type {other:?}; valid: episodic, decision, failure, procedural"
            ),
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
        // Start from configured base weights. Each intent only ever raises a
        // weight (max), so a single intent reproduces its target value and
        // compound intents keep the strongest signal instead of last-wins.
        //
        // Intents raise RELEVANCE — the classified intent means the query
        // terms matched that domain, which is evidence the BM25 score is
        // meaningful. The type weight is NEVER raised: the static type prior
        // already favors failure/decision records unconditionally, and
        // amplifying a query-independent prior under intent is exactly the
        // inversion the 2026-09 review flagged (intents used to raise only
        // the priors, letting them outvote relevance).
        //
        // Recency is raised ONLY for Workflow queries ("how do we deploy?")
        // where stale procedures are genuinely less valuable. Deployment no
        // longer raises it: CJK keywords like 「迁移」 classify as Deployment,
        // and "数据库迁移方案" queries are usually after historical DECISIONS —
        // raising recency there actively suppressed the records being asked
        // for (a 30-day half-life leaves a 40-day-old decision at ~0.40).
        let mut relevance_weight = self.base.relevance;
        let mut recency_weight = self.base.recency;
        // Per-record importance and the type prior are never intent-adjusted:
        // they are properties of the record, not of the query.
        let importance_weight = self.base.importance;
        let type_weight = self.base.type_weight;

        for intent in intents {
            match intent {
                MemoryIntent::Debugging => {
                    relevance_weight = relevance_weight.max(0.6);
                }
                MemoryIntent::Architecture => {
                    relevance_weight = relevance_weight.max(0.6);
                }
                MemoryIntent::Workflow => {
                    relevance_weight = relevance_weight.max(0.55);
                    recency_weight = recency_weight.max(0.25);
                }
                MemoryIntent::Refactor => {
                    relevance_weight = relevance_weight.max(0.55);
                }
                MemoryIntent::Deployment => {
                    relevance_weight = relevance_weight.max(0.55);
                }
                MemoryIntent::Incident => {
                    relevance_weight = relevance_weight.max(0.65);
                }
                MemoryIntent::General => {}
            }
        }

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
        // Debugging raises relevance (the query matched debugging terms).
        assert!(plan.relevance_weight > PlanWeights::default().relevance);
    }

    #[test]
    fn test_architecture_plan() {
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::Architecture]);
        assert!(plan.relevance_weight > PlanWeights::default().relevance);
    }

    #[test]
    fn test_general_plan_is_base_weights() {
        // General intent leaves the configured base weights untouched.
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::General]);
        let base = PlanWeights::default().normalized();
        assert!((plan.relevance_weight - base.relevance).abs() < 1e-6);
        assert!((plan.recency_weight - base.recency).abs() < 1e-6);
    }

    #[test]
    fn deployment_does_not_raise_recency() {
        // 「迁移」-style queries classify as Deployment but usually want
        // historical decisions; raising recency suppressed exactly those.
        // Only Workflow may raise recency. Compare against the General plan
        // (base weights, renormalized) so the check measures a real raise,
        // not the share shrink that follows any relevance raise.
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let general = planner.plan(&[MemoryIntent::General]);
        for intents in [
            vec![MemoryIntent::Deployment],
            vec![MemoryIntent::Deployment, MemoryIntent::Refactor],
        ] {
            let plan = planner.plan(&intents);
            assert!(
                plan.recency_weight <= general.recency_weight + 1e-6,
                "{intents:?} must not raise recency (got {}, base {})",
                plan.recency_weight,
                general.recency_weight
            );
        }
        let wf = planner.plan(&[MemoryIntent::Workflow]);
        assert!(wf.recency_weight > general.recency_weight);
    }

    #[test]
    fn test_compound_intent_plan() {
        let planner = RetrievalPlanner::new(PlanWeights::default());
        let plan = planner.plan(&[MemoryIntent::Debugging, MemoryIntent::Incident]);
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
