use crate::models::{DecisionMemory, EpisodicMemory, FailureMemory, ProceduralMemory};
use crate::storage::{MemoryRepository, ScoredMemory};
use anyhow::Result;

/// A single search result with relevance score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub memory_type: String,
    pub summary: String,
    pub relevance_score: f32,
    pub importance: f32,
    pub created_at: i64,
}

/// Trait for extracting searchable fields from memory types.
trait Searchable {
    fn search_id(&self) -> &str;
    fn search_summary(&self) -> &str;
    fn search_created_at(&self) -> i64;
    fn search_importance(&self) -> f32;
}

impl Searchable for EpisodicMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_summary(&self) -> &str {
        &self.summary
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        self.importance.clamp(0.0, 1.0)
    }
}

impl Searchable for DecisionMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_summary(&self) -> &str {
        &self.title
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        0.5
    }
}

impl Searchable for FailureMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_summary(&self) -> &str {
        &self.incident
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        (self.severity as f32 / 5.0).clamp(0.0, 1.0)
    }
}

impl Searchable for ProceduralMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_summary(&self) -> &str {
        &self.workflow_name
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        0.5
    }
}

/// BM25 retrieval engine using SQLite FTS5.
pub struct BM25Retriever;

impl Default for BM25Retriever {
    fn default() -> Self {
        Self::new()
    }
}

impl BM25Retriever {
    pub fn new() -> Self {
        Self
    }

    /// Sigmoid normalization scaling factor.
    /// Controls how quickly the normalization curve saturates.
    const SIGMOID_SCALE: f64 = 5.0;

    /// Normalize FTS5 bm25() score to [0, 1].
    /// bm25() returns negative values; more negative = better match.
    /// We negate and apply sigmoid-like normalization.
    fn normalize_bm25(score: f64) -> f32 {
        let positive = (-score).max(0.0);
        (1.0 - (-positive / Self::SIGMOID_SCALE).exp()) as f32
    }

    /// Convert scored memories of any type into SearchResults.
    fn to_results<T: Searchable>(
        scored_mems: Vec<ScoredMemory<T>>,
        memory_type: &str,
    ) -> Vec<SearchResult> {
        scored_mems
            .into_iter()
            .map(|scored| SearchResult {
                id: scored.memory.search_id().to_string(),
                memory_type: memory_type.into(),
                summary: scored.memory.search_summary().to_string(),
                relevance_score: Self::normalize_bm25(scored.bm25_score),
                importance: scored.memory.search_importance(),
                created_at: scored.memory.search_created_at(),
            })
            .collect()
    }

    /// Search all memory types via FTS5 BM25 for a given query.
    /// Returns combined results sorted by relevance.
    pub fn search_all(
        repo: &MemoryRepository,
        query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        results.extend(Self::to_results(
            repo.search_episodic(query, project_id, limit)?,
            "episodic",
        ));
        results.extend(Self::to_results(
            repo.search_decisions(query, project_id, limit)?,
            "decision",
        ));
        results.extend(Self::to_results(
            repo.search_failures(query, project_id, limit)?,
            "failure",
        ));
        results.extend(Self::to_results(
            repo.search_procedural(query, project_id, limit)?,
            "procedural",
        ));

        // Sort by relevance score descending
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    /// Search a specific memory type.
    pub fn search_by_type(
        repo: &MemoryRepository,
        query: &str,
        project_id: &str,
        memory_type: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        match memory_type {
            "episodic" => Ok(Self::to_results(
                repo.search_episodic(query, project_id, limit)?,
                "episodic",
            )),
            "decision" => Ok(Self::to_results(
                repo.search_decisions(query, project_id, limit)?,
                "decision",
            )),
            "failure" => Ok(Self::to_results(
                repo.search_failures(query, project_id, limit)?,
                "failure",
            )),
            "procedural" => Ok(Self::to_results(
                repo.search_procedural(query, project_id, limit)?,
                "procedural",
            )),
            _ => Ok(Vec::new()),
        }
    }

    /// Materialize `SearchResult`s for explicit `(memory_type, id)` pairs by
    /// loading them from the repo. Used to bring vector-only hits into the
    /// fused candidate set. `relevance_score` is 0.0 (these were not BM25 hits);
    /// the reranker recomputes the final score. Importance is normalized the
    /// same way as the BM25 path (via `Searchable`).
    pub fn fetch_by_ids(
        repo: &MemoryRepository,
        ids: &[(String, String)],
    ) -> Result<Vec<SearchResult>> {
        let mut out = Vec::new();
        for (memory_type, id) in ids {
            let mut sr = match memory_type.as_str() {
                "episodic" => repo.get_episodic(id)?.map(|m| {
                    Self::to_results(vec![ScoredMemory { memory: m, bm25_score: 0.0 }], "episodic")
                }),
                "decision" => repo.get_decision(id)?.map(|m| {
                    Self::to_results(vec![ScoredMemory { memory: m, bm25_score: 0.0 }], "decision")
                }),
                "failure" => repo.get_failure(id)?.map(|m| {
                    Self::to_results(vec![ScoredMemory { memory: m, bm25_score: 0.0 }], "failure")
                }),
                "procedural" => repo.get_procedural(id)?.map(|m| {
                    Self::to_results(vec![ScoredMemory { memory: m, bm25_score: 0.0 }], "procedural")
                }),
                _ => None,
            };
            if let Some(v) = sr.as_mut() {
                out.append(v);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{EpisodicMemory, FailureMemory, DecisionMemory, ProceduralMemory};
    use crate::storage::ScoredMemory;

    fn episodic(importance: f32) -> EpisodicMemory {
        EpisodicMemory {
            id: "e1".into(), project_id: "p".into(), session_id: "s".into(),
            summary: "sum".into(), content: "c".into(), files_touched: vec![],
            related_commits: vec![], importance, tags: vec![], created_at: 0, updated_at: 0,
        }
    }
    fn failure(severity: u8) -> FailureMemory {
        FailureMemory {
            id: "f1".into(), project_id: "p".into(), incident: "boom".into(),
            root_cause: "rc".into(), fix: "fx".into(), prevention: "pv".into(),
            severity, tags: vec![], created_at: 0, updated_at: 0,
        }
    }
    fn decision() -> DecisionMemory {
        DecisionMemory {
            id: "d1".into(), project_id: "p".into(), title: "t".into(), context: "ctx".into(),
            rationale: "r".into(), tradeoffs: "to".into(), related_files: vec![], tags: vec![],
            created_at: 0, updated_at: 0,
        }
    }
    fn procedural() -> ProceduralMemory {
        ProceduralMemory {
            id: "pr1".into(), project_id: "p".into(), workflow_name: "wf".into(),
            steps: vec![], related_tools: vec![], tags: vec![], created_at: 0, updated_at: 0,
        }
    }

    #[test]
    fn importance_maps_per_type() {
        // episodic: passthrough
        let r = BM25Retriever::to_results(vec![ScoredMemory { memory: episodic(0.8), bm25_score: -1.0 }], "episodic");
        assert!((r[0].importance - 0.8).abs() < 1e-6);
        // failure: severity / 5
        let r = BM25Retriever::to_results(vec![ScoredMemory { memory: failure(5), bm25_score: -1.0 }], "failure");
        assert!((r[0].importance - 1.0).abs() < 1e-6);
        let r = BM25Retriever::to_results(vec![ScoredMemory { memory: failure(1), bm25_score: -1.0 }], "failure");
        assert!((r[0].importance - 0.2).abs() < 1e-6);
        // decision / procedural: neutral 0.5
        let r = BM25Retriever::to_results(vec![ScoredMemory { memory: decision(), bm25_score: -1.0 }], "decision");
        assert!((r[0].importance - 0.5).abs() < 1e-6);
        let r = BM25Retriever::to_results(vec![ScoredMemory { memory: procedural(), bm25_score: -1.0 }], "procedural");
        assert!((r[0].importance - 0.5).abs() < 1e-6);
    }
}
