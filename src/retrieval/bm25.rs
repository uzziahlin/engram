use crate::storage::MemoryRepository;
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

    /// Normalize FTS5 bm25() score to [0, 1].
    /// bm25() returns negative values; more negative = better match.
    /// We negate and apply sigmoid-like normalization.
    fn normalize_bm25(score: f64) -> f32 {
        let positive = (-score).max(0.0);
        (1.0 - (-positive / 5.0).exp()) as f32
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

        // Search episodic memories
        let episodic = repo.search_episodic(query, project_id, limit)?;
        for scored in episodic {
            let relevance = Self::normalize_bm25(scored.bm25_score);
            results.push(SearchResult {
                id: scored.memory.id,
                memory_type: "episodic".into(),
                summary: scored.memory.summary,
                relevance_score: relevance,
                created_at: scored.memory.created_at,
            });
        }

        // Search decision memories
        let decisions = repo.search_decisions(query, project_id, limit)?;
        for scored in decisions {
            let relevance = Self::normalize_bm25(scored.bm25_score);
            results.push(SearchResult {
                id: scored.memory.id,
                memory_type: "decision".into(),
                summary: scored.memory.title,
                relevance_score: relevance,
                created_at: scored.memory.created_at,
            });
        }

        // Search failure memories
        let failures = repo.search_failures(query, project_id, limit)?;
        for scored in failures {
            let relevance = Self::normalize_bm25(scored.bm25_score);
            results.push(SearchResult {
                id: scored.memory.id,
                memory_type: "failure".into(),
                summary: scored.memory.incident,
                relevance_score: relevance,
                created_at: scored.memory.created_at,
            });
        }

        // Search procedural memories
        let procedural = repo.search_procedural(query, project_id, limit)?;
        for scored in procedural {
            let relevance = Self::normalize_bm25(scored.bm25_score);
            results.push(SearchResult {
                id: scored.memory.id,
                memory_type: "procedural".into(),
                summary: scored.memory.workflow_name,
                relevance_score: relevance,
                created_at: scored.memory.created_at,
            });
        }

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
            "episodic" => {
                let scored_mems = repo.search_episodic(query, project_id, limit)?;
                Ok(scored_mems.into_iter().map(|scored| SearchResult {
                    id: scored.memory.id,
                    memory_type: "episodic".into(),
                    summary: scored.memory.summary,
                    relevance_score: Self::normalize_bm25(scored.bm25_score),
                    created_at: scored.memory.created_at,
                }).collect())
            }
            "decision" => {
                let scored_mems = repo.search_decisions(query, project_id, limit)?;
                Ok(scored_mems.into_iter().map(|scored| SearchResult {
                    id: scored.memory.id,
                    memory_type: "decision".into(),
                    summary: scored.memory.title,
                    relevance_score: Self::normalize_bm25(scored.bm25_score),
                    created_at: scored.memory.created_at,
                }).collect())
            }
            "failure" => {
                let scored_mems = repo.search_failures(query, project_id, limit)?;
                Ok(scored_mems.into_iter().map(|scored| SearchResult {
                    id: scored.memory.id,
                    memory_type: "failure".into(),
                    summary: scored.memory.incident,
                    relevance_score: Self::normalize_bm25(scored.bm25_score),
                    created_at: scored.memory.created_at,
                }).collect())
            }
            "procedural" => {
                let scored_mems = repo.search_procedural(query, project_id, limit)?;
                Ok(scored_mems.into_iter().map(|scored| SearchResult {
                    id: scored.memory.id,
                    memory_type: "procedural".into(),
                    summary: scored.memory.workflow_name,
                    relevance_score: Self::normalize_bm25(scored.bm25_score),
                    created_at: scored.memory.created_at,
                }).collect())
            }
            _ => Ok(Vec::new()),
        }
    }
}
