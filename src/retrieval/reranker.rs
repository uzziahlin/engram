use crate::retrieval::bm25::SearchResult;
use crate::retrieval::planner::RetrievalPlan;

/// Reranker for combining and ranking retrieval results
/// based on recency, importance, and graph weights.
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

    /// Rerank search results based on the retrieval plan's weights.
    ///
    /// Scoring formula:
    ///   final_score = relevance * 0.4
    ///               + recency_normalized * plan.recency_weight
    ///               + relevance * plan.importance_weight
    ///               + type_boost * plan.graph_weight
    pub fn rerank(&self, results: &mut [SearchResult], plan: &RetrievalPlan, _now_timestamp: i64) {
        if results.is_empty() {
            return;
        }

        // Find the newest and oldest timestamps for normalization
        let max_ts = results.iter().map(|r| r.created_at).max().unwrap_or(0);
        let min_ts = results.iter().map(|r| r.created_at).min().unwrap_or(0);
        let ts_range = (max_ts - min_ts).max(1) as f32;

        for result in results.iter_mut() {
            let recency_normalized = if ts_range > 0.0 {
                (result.created_at - min_ts) as f32 / ts_range
            } else {
                1.0
            };

            let base_relevance = result.relevance_score;

            // Static type-based boost reflecting memory type importance
            let type_boost: f32 = match result.memory_type.as_str() {
                "failure" => 0.9,
                "decision" => 0.7,
                "episodic" => 0.5,
                "procedural" => 0.3,
                _ => 0.4,
            };

            // Combined score
            let final_score = base_relevance * 0.4
                + recency_normalized * plan.recency_weight
                + base_relevance * plan.importance_weight
                + type_boost * plan.graph_weight;

            result.relevance_score = final_score.min(1.0);
        }

        // Sort by final score descending
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
            graph_weight: 0.4,
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
                created_at: 1000,
            },
            SearchResult {
                id: "2".into(),
                memory_type: "episodic".into(),
                summary: "recent unimportant".into(),
                relevance_score: 0.2,
                created_at: 2000,
            },
        ];

        reranker.rerank(&mut results, &plan, 2000);

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
                created_at: 1000,
            },
            SearchResult {
                id: "1".into(),
                memory_type: "episodic".into(),
                summary: "duplicate".into(),
                relevance_score: 0.8,
                created_at: 1000,
            },
            SearchResult {
                id: "2".into(),
                memory_type: "episodic".into(),
                summary: "unique".into(),
                relevance_score: 0.7,
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
                created_at: 1000,
            })
            .collect();

        reranker.apply_fallback_limits(&mut results, 25, 10);
        assert_eq!(results.len(), 10);
    }
}
