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

    /// Search all memory types via FTS5 BM25 for a given query.
    /// Returns combined results sorted by relevance.
    pub fn search_all(
        repo: &MemoryRepository,
        query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();

        // TODO: Phase 2 — extract FTS5 BM25 rank from repository layer for accurate scoring.
        // Current approach uses field-based heuristics normalized to [0, 1].

        // Search episodic memories — importance field is user-set [0, 1]
        let episodic = repo.search_episodic(query, project_id, limit)?;
        for mem in episodic {
            results.push(SearchResult {
                id: mem.id,
                memory_type: "episodic".into(),
                summary: mem.summary,
                relevance_score: mem.importance.clamp(0.0, 1.0),
                created_at: mem.created_at,
            });
        }

        // Search decision memories — high base relevance (decisions are high-signal)
        let decisions = repo.search_decisions(query, project_id, limit)?;
        for mem in decisions {
            results.push(SearchResult {
                id: mem.id,
                memory_type: "decision".into(),
                summary: mem.title,
                relevance_score: 0.8,
                created_at: mem.created_at,
            });
        }

        // Search failure memories
        let failures = repo.search_failures(query, project_id, limit)?;
        for mem in failures {
            results.push(SearchResult {
                id: mem.id,
                memory_type: "failure".into(),
                summary: mem.incident,
                relevance_score: mem.severity as f32 / 5.0,
                created_at: mem.created_at,
            });
        }

        // Search procedural memories
        let procedural = repo.search_procedural(query, project_id, limit)?;
        for mem in procedural {
            results.push(SearchResult {
                id: mem.id,
                memory_type: "procedural".into(),
                summary: mem.workflow_name,
                relevance_score: 0.6,
                created_at: mem.created_at,
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
                let mems = repo.search_episodic(query, project_id, limit)?;
                Ok(mems.into_iter().map(|mem| SearchResult {
                    id: mem.id,
                    memory_type: "episodic".into(),
                    summary: mem.summary,
                    relevance_score: mem.importance,
                    created_at: mem.created_at,
                }).collect())
            }
            "decision" => {
                let mems = repo.search_decisions(query, project_id, limit)?;
                Ok(mems.into_iter().map(|mem| SearchResult {
                    id: mem.id,
                    memory_type: "decision".into(),
                    summary: mem.title,
                    relevance_score: 0.7,
                    created_at: mem.created_at,
                }).collect())
            }
            "failure" => {
                let mems = repo.search_failures(query, project_id, limit)?;
                Ok(mems.into_iter().map(|mem| SearchResult {
                    id: mem.id,
                    memory_type: "failure".into(),
                    summary: mem.incident,
                    relevance_score: mem.severity as f32 / 5.0,
                    created_at: mem.created_at,
                }).collect())
            }
            "procedural" => {
                let mems = repo.search_procedural(query, project_id, limit)?;
                Ok(mems.into_iter().map(|mem| SearchResult {
                    id: mem.id,
                    memory_type: "procedural".into(),
                    summary: mem.workflow_name,
                    relevance_score: 0.6,
                    created_at: mem.created_at,
                }).collect())
            }
            _ => Ok(Vec::new()),
        }
    }
}
