//! Reciprocal Rank Fusion (RRF) of multiple ranked id lists.
//!
//! Library-agnostic: fuses ranked id lists (e.g. BM25 ranks + vector ranks)
//! into one ranking by `score(id) = Σ 1/(k + rank)`, robust to score-scale
//! differences across sources. See
//! `docs/superpowers/plans/2026-06-22-semantic-search-spike.md` Task 2.

use std::collections::HashMap;

/// Reciprocal-rank damping constant, shared by [`rrf_fuse`] and
/// [`rrf_two_arm`]. 60 is the canonical value from the RRF literature: large
/// enough that rank differences within the top ~10 stay gentle, small enough
/// that rank 1 vs rank 100 still matters.
pub const RRF_K: f32 = 60.0;

/// Fuse multiple ranked lists of ids into one, by RRF: `score(id) = Σ 1/(k + rank)`.
/// `rank` is the 0-based position within each input list. Returns ids sorted by
/// fused score descending; equal scores break ties by id ascending so the
/// output is deterministic across runs (HashMap iteration order must not leak).
pub fn rrf_fuse(lists: &[Vec<String>], k: f32) -> Vec<String> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (k + rank as f32);
        }
    }
    let mut ids: Vec<(String, f32)> = scores.into_iter().collect();
    ids.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ids.into_iter().map(|(id, _)| id).collect()
}

/// Two-arm RRF relevance for semantic fusion: `1/(RRF_K + bm25_rank) +
/// 1/(RRF_K + cosine_rank)`, each arm contributing only when the candidate
/// actually ranked in that list. An ABSENT arm is missing evidence, not a
/// last-place rank: a vector-only candidate's semantics is its only signal
/// (and symmetrically for BM25-only candidates), so treating absence as
/// rank-∞ would bury exactly the candidates fusion exists to surface.
///
/// Normalized to [0, 1] by the theoretical best `2/RRF_K` (both arms at
/// rank 0 — ranks are 0-based, the same convention as [`rrf_fuse`], so the
/// normalizer is `RRF_K / 2`, not `(RRF_K + 1) / 2`). The normalizer is a
/// positive constant, hence order-preserving — deliberately chosen over a
/// per-candidate-set min-max, which degenerates to 0/0 when all candidates
/// tie and makes absolute values depend on the set. A single-signal
/// candidate therefore caps at 0.5: one arm carries at most half the
/// possible evidence.
///
/// Why ranks instead of the old `max(normalized_bm25, cosine)`: embedding
/// cosine has a heavy noise floor (MiniLM scores unrelated pairs 0.25-0.45,
/// right where calibrated BM25 scores for weak-but-real matches land), so
/// max-injection let pure noise override real lexical evidence. Ranks are
/// scale-blind by construction — a noise-floor cosine can at best earn its
/// own arm's rank, never overwrite the other arm's.
pub fn rrf_two_arm(bm25_rank: Option<usize>, cosine_rank: Option<usize>) -> f32 {
    let arm = |rank: Option<usize>| rank.map(|r| 1.0 / (RRF_K + r as f32)).unwrap_or(0.0);
    (arm(bm25_rank) + arm(cosine_rank)) * RRF_K / 2.0
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

    // ─── Two-arm RRF relevance (semantic fusion input) ──────────────
    // Injected-score unit tests: the whole point of extracting
    // `rrf_two_arm` as a pure function is that the fusion tradeoff is
    // testable WITHOUT downloading an embedding model.

    /// The shootout the old max-injection lost: a true lexical match with a
    /// weak calibrated BM25 score (0.35) plus a weak-but-present cosine
    /// (0.30, rank ~10 in the vector list) vs. an unrelated document riding
    /// MiniLM's cosine noise floor (0.42, vector rank 0, no BM25 hit).
    /// max() ranks the noise above the match; RRF ranks the match above the
    /// noise because the noise has no BM25 arm to add.
    #[test]
    fn rrf_ranks_true_match_above_cosine_noise_floor_where_max_did_not() {
        // Old semantics, computed inline for the side-by-side (old-vs-new)
        // claim: max(normalized_bm25, cosine.clamp(0,1)).
        let true_match = (0.35f32, 0.30f32); // (bm25, cosine)
        let noise = (0.0f32, 0.42f32); // no BM25 hit, cosine noise floor
        assert!(
            noise.1.max(noise.0) > true_match.1.max(true_match.0),
            "precondition: under max() the noise doc (0.42) outranks the match (0.35)"
        );
        // New semantics: strong BM25 + weak cosine vs absent BM25 + top cosine.
        let rrf_true = rrf_two_arm(Some(0), Some(10));
        let rrf_noise = rrf_two_arm(None, Some(0));
        assert!(
            rrf_true > rrf_noise,
            "RRF must rank the dual-evidence match ({rrf_true}) above the cosine-only noise doc ({rrf_noise})"
        );
    }

    /// Mirror image: a strong lexical match that the vector arm missed must
    /// still outrank weak BM25 matches — the BM25 arm alone carries it.
    #[test]
    fn rrf_ranks_strong_bm25_match_above_weaker_bm25_with_equal_cosine() {
        // Same cosine rank, different BM25 ranks → BM25 ordering preserved.
        assert!(rrf_two_arm(Some(0), Some(5)) > rrf_two_arm(Some(7), Some(5)));
        // Same BM25 rank, different cosine ranks → cosine ordering preserved.
        assert!(rrf_two_arm(Some(3), Some(0)) > rrf_two_arm(Some(3), Some(9)));
    }

    /// Single-signal candidates keep that signal's ordering: relevance is
    /// strictly decreasing in the present arm's rank, absent arm or not.
    #[test]
    fn rrf_preserves_order_within_a_single_arm() {
        let mut prev_bm25_only = rrf_two_arm(Some(0), None);
        for rank in 1..50 {
            let next = rrf_two_arm(Some(rank), None);
            assert!(
                next < prev_bm25_only,
                "BM25-only order broken at rank {rank}"
            );
            prev_bm25_only = next;
        }
        let mut prev_vec_only = rrf_two_arm(None, Some(0));
        for rank in 1..50 {
            let next = rrf_two_arm(None, Some(rank));
            assert!(
                next < prev_vec_only,
                "cosine-only order broken at rank {rank}"
            );
            prev_vec_only = next;
        }
    }

    /// Bounded [0, 1]; dual evidence outranks any single arm (a single arm
    /// caps at 0.5 — half the possible evidence).
    #[test]
    fn rrf_bounds_and_single_arm_ceiling() {
        assert!((rrf_two_arm(Some(0), Some(0)) - 1.0).abs() < 1e-6);
        assert!((rrf_two_arm(Some(0), None) - 0.5).abs() < 1e-6);
        assert!((rrf_two_arm(None, Some(0)) - 0.5).abs() < 1e-6);
        assert_eq!(rrf_two_arm(None, None), 0.0);
        assert!(rrf_two_arm(Some(0), Some(1)) < 1.0);
        // Dual evidence always beats either arm alone at the same ranks.
        assert!(rrf_two_arm(Some(0), Some(40)) > rrf_two_arm(Some(0), None));
        assert!(rrf_two_arm(Some(40), Some(0)) > rrf_two_arm(None, Some(0)));
    }
}
