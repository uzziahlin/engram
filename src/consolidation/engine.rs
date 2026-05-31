use crate::storage::MemoryRepository;
use anyhow::Result;
use std::collections::HashSet;

/// Consolidation engine (MVP stub).
/// Basic rule-based deduplication: content hash + time-window merging.
pub struct ConsolidationEngine;

impl Default for ConsolidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolidationEngine {
    pub fn new() -> Self {
        Self
    }

    /// Compute a simple content hash for deduplication.
    /// Uses the summary + content fields as the hash source.
    pub fn content_hash(summary: &str, content: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        summary.hash(&mut hasher);
        content.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Deduplicate episodic memories by content hash within a project.
    /// Returns the IDs of duplicates found.
    pub fn deduplicate_episodic(
        &self,
        repo: &MemoryRepository,
        project_id: &str,
    ) -> Result<Vec<String>> {
        let conn = repo.connection();
        let mut stmt = conn.prepare(
            "SELECT id, summary, content FROM episodic_memories WHERE project_id = ?1 ORDER BY created_at ASC",
        )?;

        let mut seen_hashes = HashSet::new();
        let mut duplicates = Vec::new();

        let rows = stmt.query_map(rusqlite::params![project_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;

        for row in rows {
            let (id, summary, content) = row?;
            let hash = Self::content_hash(&summary, &content);
            if !seen_hashes.insert(hash) {
                duplicates.push(id);
            }
        }

        Ok(duplicates)
    }

    /// Check if two texts are near-duplicates using Jaccard similarity.
    /// Returns true if similarity exceeds the threshold.
    pub fn jaccard_similarity(a: &str, b: &str, threshold: f64) -> bool {
        let set_a: HashSet<&str> = a.split_whitespace().collect();
        let set_b: HashSet<&str> = b.split_whitespace().collect();

        if set_a.is_empty() && set_b.is_empty() {
            return true;
        }

        let intersection = set_a.intersection(&set_b).count() as f64;
        let union = set_a.union(&set_b).count() as f64;

        if union == 0.0 {
            return false;
        }

        (intersection / union) >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = ConsolidationEngine::content_hash("summary", "content");
        let h2 = ConsolidationEngine::content_hash("summary", "content");
        assert_eq!(h1, h2);

        let h3 = ConsolidationEngine::content_hash("other", "content");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_jaccard_similarity() {
        assert!(ConsolidationEngine::jaccard_similarity(
            "hello world foo",
            "hello world bar",
            0.4,
        ));
        assert!(!ConsolidationEngine::jaccard_similarity(
            "completely different text",
            "totally unrelated words",
            0.5,
        ));
    }

    #[test]
    fn test_jaccard_identical() {
        assert!(ConsolidationEngine::jaccard_similarity("same text", "same text", 0.99));
    }
}
