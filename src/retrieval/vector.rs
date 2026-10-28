//! Pure vector math for semantic retrieval (library-agnostic, always compiled).

/// Cosine similarity of two equal-length vectors. Returns 0.0 on length mismatch
/// or zero-norm input (degenerate, treated as "no similarity").
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Return the top-`k` candidates by cosine similarity to `query`, highest
/// first, as `(id, cosine)` pairs — the score feeds the fused relevance.
/// `candidates` yields borrowed `(id, vector)` pairs so callers can score
/// straight out of a cache without cloning every blob. Ties broken by input order.
pub fn top_k_cosine<'a>(
    query: &[f32],
    candidates: impl IntoIterator<Item = (&'a str, &'a [f32])>,
    k: usize,
) -> Vec<(String, f32)> {
    let mut scored: Vec<(usize, &str, f32)> = candidates
        .into_iter()
        .enumerate()
        .map(|(i, (id, v))| (i, id, cosine(query, v)))
        .collect();
    // Sort by score desc; ties broken by original input index (stable order).
    scored.sort_by(|x, y| {
        y.2.partial_cmp(&x.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(x.0.cmp(&y.0))
    });
    scored
        .into_iter()
        .take(k)
        .map(|(_, id, score)| (id.to_string(), score))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_basics() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0); // length mismatch
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 1.0]), 0.0); // zero norm
    }

    #[test]
    fn top_k_orders_and_truncates() {
        let cands = [
            ("a".to_string(), vec![1.0, 0.0]),
            ("b".to_string(), vec![0.0, 1.0]),
            ("c".to_string(), vec![0.9, 0.1]),
        ];
        let got = top_k_cosine(
            &[1.0, 0.0],
            cands.iter().map(|(id, v)| (id.as_str(), v.as_slice())),
            2,
        );
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, "a");
        assert!(
            (got[0].1 - 1.0).abs() < 1e-6,
            "score must be returned for relevance injection"
        );
        assert_eq!(got[1].0, "c");
    }
}
