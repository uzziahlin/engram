//! Reciprocal Rank Fusion (RRF) of multiple ranked id lists.
//!
//! Library-agnostic: fuses ranked id lists (e.g. BM25 ranks + vector ranks)
//! into one ranking by `score(id) = Σ 1/(k + rank)`, robust to score-scale
//! differences across sources. See
//! `docs/superpowers/plans/2026-06-22-semantic-search-spike.md` Task 2.

use std::collections::HashMap;

/// Fuse multiple ranked lists of ids into one, by RRF: `score(id) = Σ 1/(k + rank)`.
/// `rank` is the 0-based position within each input list. Returns ids sorted by
/// fused score descending.
pub fn rrf_fuse(lists: &[Vec<String>], k: f32) -> Vec<String> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32);
        }
    }
    let mut ids: Vec<(String, f32)> = scores.into_iter().collect();
    ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ids.into_iter().map(|(id, _)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuses_two_lists_by_reciprocal_rank() {
        let bm25 = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let vec_ = vec!["c".to_string(), "a".to_string(), "d".to_string()];
        let fused = rrf_fuse(&[bm25, vec_], 60.0);
        // 'a' appears high in both → should rank first; 'c' second; unique tails after.
        assert_eq!(fused[0], "a");
        assert!(fused.contains(&"d".to_string()));
    }

    #[test]
    fn empty_lists_yield_empty() {
        assert!(rrf_fuse(&[], 60.0).is_empty());
    }
}
