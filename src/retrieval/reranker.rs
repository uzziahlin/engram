use crate::retrieval::bm25::SearchResult;
use crate::retrieval::planner::RetrievalPlan;

/// Reranker for combining and ranking retrieval results
/// based on recency, importance, and type weights.
pub struct Reranker;

impl Default for Reranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker {
    pub fn new() -> Self {
        Self
    }

    /// Rerank search results.
    ///
    ///   final = W_REL * relevance
    ///         + recency_weight  * recency_decay(now, created_at; half_life)
    ///         + importance_weight * importance
    ///         + type_weight     * type_prior
    pub fn rerank(
        &self,
        results: &mut [SearchResult],
        plan: &RetrievalPlan,
        now_timestamp: i64,
        half_life_seconds: f32,
    ) {
        if results.is_empty() {
            return;
        }

        const W_REL: f32 = 0.4;

        for result in results.iter_mut() {
            let relevance = result.relevance_score;

            // Real exponential half-life decay against the actual clock.
            let age = (now_timestamp - result.created_at).max(0) as f32;
            let recency_decay = if half_life_seconds > 0.0 {
                0.5_f32.powf(age / half_life_seconds)
            } else {
                1.0
            };

            // Type prior (memory-type importance), distinct from per-record importance.
            let type_prior: f32 = match result.memory_type.as_str() {
                "failure" => 0.9,
                "decision" => 0.7,
                "episodic" => 0.5,
                "procedural" => 0.3,
                _ => 0.4,
            };

            let final_score = W_REL * relevance
                + plan.recency_weight * recency_decay
                + plan.importance_weight * result.importance
                + plan.type_weight * type_prior;

            // Store the unclamped score: it is the sort key, and clamping here
            // would flatten the top of the ranking (default weights let finals
            // reach ~1.46). The output layer (server.rs) clamps for display.
            result.relevance_score = final_score;
        }

        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Deduplicate results by ID.
    pub fn deduplicate(&self, results: &mut Vec<SearchResult>) {
        let mut seen = std::collections::HashSet::new();
        results.retain(|r| seen.insert(r.id.clone()));
    }

    /// Apply fallback resource limits.
    pub fn apply_fallback_limits(
        &self,
        results: &mut Vec<SearchResult>,
        max_input: usize,
        max_output: usize,
    ) {
        results.truncate(max_output.min(max_input));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retrieval::planner::MemorySource;

    fn make_plan() -> RetrievalPlan {
        RetrievalPlan {
            sources: vec![MemorySource::Episodic],
            recency_weight: 0.2,
            importance_weight: 0.6,
            type_weight: 0.4,
        }
    }

    #[test]
    fn test_rerank_sorts_by_combined_score() {
        let reranker = Reranker::new();
        let plan = make_plan();

        let mut results = vec![
            SearchResult {
                id: "1".into(),
                memory_type: "episodic".into(),
                summary: "old important".into(),
                relevance_score: 0.9,
                importance: 0.5,
                created_at: 1000,
            },
            SearchResult {
                id: "2".into(),
                memory_type: "episodic".into(),
                summary: "recent unimportant".into(),
                relevance_score: 0.2,
                importance: 0.5,
                created_at: 2000,
            },
        ];

        reranker.rerank(&mut results, &plan, 2000, 30.0 * 86400.0);

        // Both should have scores now, and they should be sorted descending
        assert!(results[0].relevance_score >= results[1].relevance_score);
    }

    #[test]
    fn test_deduplicate() {
        let reranker = Reranker::new();
        let mut results = vec![
            SearchResult {
                id: "1".into(),
                memory_type: "episodic".into(),
                summary: "first".into(),
                relevance_score: 0.9,
                importance: 0.5,
                created_at: 1000,
            },
            SearchResult {
                id: "1".into(),
                memory_type: "episodic".into(),
                summary: "duplicate".into(),
                relevance_score: 0.8,
                importance: 0.5,
                created_at: 1000,
            },
            SearchResult {
                id: "2".into(),
                memory_type: "episodic".into(),
                summary: "unique".into(),
                relevance_score: 0.7,
                importance: 0.5,
                created_at: 1000,
            },
        ];

        reranker.deduplicate(&mut results);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "1");
        assert_eq!(results[1].id, "2");
    }

    #[test]
    fn test_fallback_limits() {
        let reranker = Reranker::new();

        let mut results: Vec<SearchResult> = (0..50)
            .map(|i| SearchResult {
                id: format!("{i}"),
                memory_type: "episodic".into(),
                summary: format!("result {i}"),
                relevance_score: 0.5,
                importance: 0.5,
                created_at: 1000,
            })
            .collect();

        reranker.apply_fallback_limits(&mut results, 25, 10);
        assert_eq!(results.len(), 10);
    }

    fn result(id: &str, ty: &str, rel: f32, imp: f32, created_at: i64) -> SearchResult {
        SearchResult {
            id: id.into(),
            memory_type: ty.into(),
            summary: String::new(),
            relevance_score: rel,
            importance: imp,
            created_at,
        }
    }

    #[test]
    fn importance_breaks_ties_at_equal_relevance() {
        let reranker = Reranker::new();
        let mut plan = make_plan();
        plan.importance_weight = 0.6;
        let now = 1_000_000;
        let half_life = 30.0 * 86400.0;
        let mut results = vec![
            result("low", "episodic", 0.5, 0.1, now),
            result("high", "episodic", 0.5, 0.9, now),
        ];
        reranker.rerank(&mut results, &plan, now, half_life);
        assert_eq!(results[0].id, "high");
    }

    #[test]
    fn high_relevance_ranks_above_low_at_saturated_top() {
        // Regression: default weights let finals exceed 1.0; clamping the sort
        // key flattened the top so two high-importance results saturated to 1.0
        // and fell back to BM25 order. With unclamped sort, the higher-relevance
        // item must win even when both would have clamped to 1.0.
        let reranker = Reranker::new();
        let plan = make_plan(); // recency 0.2, importance 0.6, type 0.4
        let now = 1_000_000;
        let half_life = 30.0 * 86400.0;
        let mut results = vec![
            result("low-rel", "episodic", 0.6, 0.9, now),
            result("high-rel", "episodic", 0.9, 0.9, now),
        ];
        reranker.rerank(&mut results, &plan, now, half_life);
        // Both finals exceed 1.0 (1.18 and 1.30) — unclamped, high-rel wins.
        assert_eq!(results[0].id, "high-rel");
        assert!(results[0].relevance_score > results[1].relevance_score);
    }

    #[test]
    fn all_old_results_are_not_treated_as_fresh() {
        let reranker = Reranker::new();
        let mut plan = make_plan();
        plan.recency_weight = 1.0;
        plan.importance_weight = 0.0;
        plan.type_weight = 0.0;
        let now = 1_000_000_000;
        let half_life = 30.0 * 86400.0;
        let year = 365 * 86400;
        let mut results = vec![
            result("older", "episodic", 0.0, 0.0, now - 2 * year),
            result("old", "episodic", 0.0, 0.0, now - year),
        ];
        reranker.rerank(&mut results, &plan, now, half_life);
        for r in &results {
            assert!(r.relevance_score < 0.1, "old memory {} wrongly fresh: {}", r.id, r.relevance_score);
        }
    }
}
