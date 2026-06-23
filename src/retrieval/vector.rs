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

/// Return the ids of the top-`k` candidates by cosine similarity to `query`,
/// highest first. `candidates` is (id, vector). Ties broken by input order.
pub fn top_k_cosine(query: &[f32], candidates: &[(String, Vec<f32>)], k: usize) -> Vec<String> {
    let mut scored: Vec<(usize, &String, f32)> = candidates
        .iter()
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
        .map(|(_, id, _)| id.clone())
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
        let cands = vec![
            ("a".into(), vec![1.0, 0.0]),
            ("b".into(), vec![0.0, 1.0]),
            ("c".into(), vec![0.9, 0.1]),
        ];
        let got = top_k_cosine(&[1.0, 0.0], &cands, 2);
        assert_eq!(got, vec!["a".to_string(), "c".to_string()]);
    }
}
