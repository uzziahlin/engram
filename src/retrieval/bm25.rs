use crate::models::{EpisodicMemory, DecisionMemory, FailureMemory, ProceduralMemory};
use crate::storage::{MemoryRepository, ScoredMemory};
use anyhow::Result;

/// A single search result with relevance score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    pub memory_type: String,
    pub summary: String,
    pub relevance_score: f32,
    pub created_at: i64,
}

/// Trait for extracting searchable fields from memory types.
trait Searchable {
    fn search_id(&self) -> &str;
    fn search_summary(&self) -> &str;
    fn search_created_at(&self) -> i64;
}

impl Searchable for EpisodicMemory {
    fn search_id(&self) -> &str { &self.id }
    fn search_summary(&self) -> &str { &self.summary }
    fn search_created_at(&self) -> i64 { self.created_at }
}

impl Searchable for DecisionMemory {
    fn search_id(&self) -> &str { &self.id }
    fn search_summary(&self) -> &str { &self.title }
    fn search_created_at(&self) -> i64 { self.created_at }
}

impl Searchable for FailureMemory {
    fn search_id(&self) -> &str { &self.id }
    fn search_summary(&self) -> &str { &self.incident }
    fn search_created_at(&self) -> i64 { self.created_at }
}

impl Searchable for ProceduralMemory {
    fn search_id(&self) -> &str { &self.id }
    fn search_summary(&self) -> &str { &self.workflow_name }
    fn search_created_at(&self) -> i64 { self.created_at }
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
        scored_mems.into_iter().map(|scored| SearchResult {
            id: scored.memory.search_id().to_string(),
            memory_type: memory_type.into(),
            summary: scored.memory.search_summary().to_string(),
            relevance_score: Self::normalize_bm25(scored.bm25_score),
            created_at: scored.memory.search_created_at(),
        }).collect()
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
            repo.search_episodic(query, project_id, limit)?, "episodic",
        ));
        results.extend(Self::to_results(
            repo.search_decisions(query, project_id, limit)?, "decision",
        ));
        results.extend(Self::to_results(
            repo.search_failures(query, project_id, limit)?, "failure",
        ));
        results.extend(Self::to_results(
            repo.search_procedural(query, project_id, limit)?, "procedural",
        ));

        // Sort by relevance score descending
        results.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal));
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
                repo.search_episodic(query, project_id, limit)?, "episodic",
            )),
            "decision" => Ok(Self::to_results(
                repo.search_decisions(query, project_id, limit)?, "decision",
            )),
            "failure" => Ok(Self::to_results(
                repo.search_failures(query, project_id, limit)?, "failure",
            )),
            "procedural" => Ok(Self::to_results(
                repo.search_procedural(query, project_id, limit)?, "procedural",
            )),
            _ => Ok(Vec::new()),
        }
    }
}
