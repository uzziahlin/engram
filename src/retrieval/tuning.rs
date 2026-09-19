//! Feedback-driven importance tuning — closes the adoption-feedback loop that
//! v0.5 opened. `mark_relevance` and `get_memory` produce per-event adoption
//! signals; until now nothing consumed them automatically (the ±0.1 manual
//! bump was the only effect). This module turns accumulated usage evidence
//! into bounded, reviewable importance adjustments:
//!
//! - **promote**: a memory SERVED ≥3 times whose adoption rate is ≥50% and
//!   whose importance is still below 0.9 → +0.1. It keeps proving useful but
//!   the ranking doesn't yet reflect that.
//! - **demote**: a memory served ≥5 times, never adopted, importance above
//!   0.2 → −0.1. It keeps surfacing and being ignored.
//!
//! Deliberately conservative: fixed ±0.1 steps (the same size as a manual
//! `mark_relevance`), clamped bounds, and — like `consolidate`/`reflect` —
//! dry-run by default. `engram tune` previews; `--apply` writes.

use crate::storage::{MemoryKind, MemoryRepository};

use anyhow::Result;
use serde::Serialize;

/// Serving count before a memory is eligible for promotion.
const MIN_IMPRESSIONS_PROMOTE: i64 = 3;
/// Serving count before a never-adopted memory is eligible for demotion —
/// higher than the promote floor: absence of evidence is weaker evidence.
const MIN_IMPRESSIONS_DEMOTE: i64 = 5;
/// Adoption rate (adoptions / impressions) that qualifies as "keeps proving
/// useful".
const PROMOTE_ADOPTION_RATE: f64 = 0.5;
/// Adjustment magnitude — identical to one manual `mark_relevance` call, so
/// the automatic loop can never move a memory faster than explicit feedback.
const STEP: f32 = 0.1;
/// Bounds the tuner may not cross. Importance above 0.9 / below 0.2 stays
/// the exclusive domain of human judgment.
const PROMOTE_CEILING: f32 = 0.9;
const DEMOTE_FLOOR: f32 = 0.2;

/// One proposed importance adjustment, with the evidence behind it.
#[derive(Debug, Clone, Serialize)]
pub struct TuneProposal {
    pub memory_type: String,
    pub id: String,
    pub action: &'static str,
    pub current_importance: f32,
    pub proposed_importance: f32,
    pub impressions: i64,
    pub adoptions: i64,
    /// adoptions / impressions, rounded for display.
    pub adoption_rate: f64,
}

/// Tuning pass outcome: the proposals plus the raw signal totals.
#[derive(Debug, Clone, Serialize)]
pub struct TuneReport {
    pub applied: bool,
    pub window_queries: usize,
    pub proposals: Vec<TuneProposal>,
}

/// Compute importance-adjustment proposals from retrieval feedback over the
/// last `days` days. Pure analysis — writes nothing (the caller decides).
pub fn compute_proposals(
    repo: &MemoryRepository,
    project_id: &str,
    days: i64,
    now: i64,
) -> Result<TuneReport> {
    let since = now - days.saturating_mul(86_400);

    // Impressions: parse each logged result set and count per memory id.
    let mut impressions: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let result_sets = repo.recent_result_ids(project_id, since)?;
    let query_count = result_sets.len();
    for raw in &result_sets {
        if let Ok(ids) = serde_json::from_str::<Vec<String>>(raw) {
            for id in ids {
                *impressions.entry(id).or_default() += 1;
            }
        }
    }
    let adoptions: std::collections::HashMap<String, i64> = repo
        .adoption_counts(project_id, since)?
        .into_iter()
        .map(|(id, _ty, cnt)| (id, cnt))
        .collect();

    let mut proposals = Vec::new();
    // Active-id sets per kind, loaded lazily once and reused across candidates.
    let mut active_cache: std::collections::HashMap<MemoryKind, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    let mut candidates: Vec<(String, i64, i64)> = impressions
        .iter()
        .map(|(id, served)| {
            let adopted = adoptions.get(id).copied().unwrap_or(0);
            (id.clone(), *served, adopted)
        })
        .collect();
    candidates.sort();
    for (memory_id, served, adopted_cnt) in candidates {
        if served < MIN_IMPRESSIONS_PROMOTE {
            continue;
        }
        let rate = adopted_cnt as f64 / served as f64;

        // Resolve kind + current importance, skipping anything not active
        // (get_* does not filter archived rows).
        let Some((kind, importance)) =
            resolve_active_importance(repo, project_id, &memory_id, &mut active_cache)?
        else {
            continue;
        };

        let (action, target) = if rate >= PROMOTE_ADOPTION_RATE && importance < PROMOTE_CEILING {
            ("promote", (importance + STEP).min(PROMOTE_CEILING))
        } else if adopted_cnt == 0 && served >= MIN_IMPRESSIONS_DEMOTE && importance > DEMOTE_FLOOR
        {
            ("demote", (importance - STEP).max(DEMOTE_FLOOR))
        } else {
            continue;
        };

        proposals.push(TuneProposal {
            memory_type: kind.as_str().to_string(),
            id: memory_id,
            action,
            current_importance: importance,
            proposed_importance: target,
            impressions: served,
            adoptions: adopted_cnt,
            adoption_rate: (rate * 100.0).round() / 100.0,
        });
    }

    // Stable, reviewable order: promotions first, then by evidence volume.
    proposals.sort_by(|a, b| {
        b.adoption_rate
            .partial_cmp(&a.adoption_rate)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.impressions.cmp(&a.impressions))
            .then(a.id.cmp(&b.id))
    });

    Ok(TuneReport {
        applied: false,
        window_queries: query_count,
        proposals,
    })
}

/// Apply proposals: writes importance (severity for failures) via the same
/// update path `mark_relevance` uses. Returns the number applied; ids that
/// vanished in the meantime are skipped, not fatal.
pub fn apply_proposals(
    repo: &MemoryRepository,
    project_id: &str,
    proposals: &[TuneProposal],
    now: i64,
) -> Result<usize> {
    let mut applied = 0usize;
    for p in proposals {
        let kind = MemoryKind::from_type_str(&p.memory_type)?;
        let target = p.proposed_importance;
        macro_rules! adjust_importance {
            ($get:ident, $update:ident) => {{
                let Some(mut mem) = repo.$get(&p.id, project_id)? else {
                    continue;
                };
                mem.importance = target;
                mem.updated_at = now;
                repo.$update(&mem)?;
            }};
        }
        match kind {
            MemoryKind::Episodic => adjust_importance!(get_episodic, update_episodic),
            MemoryKind::Decision => adjust_importance!(get_decision, update_decision),
            MemoryKind::Procedural => adjust_importance!(get_procedural, update_procedural),
            // Failures have no importance field: severity is the signal, on
            // the same 0.1-importance → 1-severity mapping mark_relevance uses.
            MemoryKind::Failure => {
                let Some(mut mem) = repo.get_failure(&p.id, project_id)? else {
                    continue;
                };
                let delta = ((target - p.current_importance) * 5.0).round() as i8;
                mem.severity = (mem.severity as i8 + delta).clamp(1, 5) as u8;
                mem.updated_at = now;
                repo.update_failure(&mem)?;
            }
        }
        applied += 1;
    }
    Ok(applied)
}

/// Resolve a memory id to (kind, current ranking importance), or None when
/// the memory is gone or archived. The kind comes from probing each kind's
/// ACTIVE id set (cached across calls) — impression-only ids carry no type
/// in the logs, and a stale adoption row could name a mismatched pair.
fn resolve_active_importance(
    repo: &MemoryRepository,
    project_id: &str,
    memory_id: &str,
    active_cache: &mut std::collections::HashMap<MemoryKind, std::collections::HashSet<String>>,
) -> Result<Option<(MemoryKind, f32)>> {
    for kind in MemoryKind::all() {
        let entry = active_cache.entry(kind).or_insert_with(|| {
            match repo.list_active_candidates(kind, project_id, &[], None) {
                Ok(ids) => ids.into_iter().collect(),
                Err(e) => {
                    tracing::warn!("tune: listing active {} ids failed: {e}", kind.as_str());
                    std::collections::HashSet::new()
                }
            }
        });
        if !entry.contains(memory_id) {
            continue;
        }
        let importance = match kind {
            MemoryKind::Episodic => repo
                .get_episodic(memory_id, project_id)?
                .map(|m| m.importance),
            MemoryKind::Decision => repo
                .get_decision(memory_id, project_id)?
                .map(|m| m.importance),
            MemoryKind::Failure => repo
                .get_failure(memory_id, project_id)?
                .map(|m| m.severity as f32 / 5.0),
            MemoryKind::Procedural => repo
                .get_procedural(memory_id, project_id)?
                .map(|m| m.importance),
        };
        return Ok(importance.map(|i| (kind, i)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::*;

    const P: &str = "tune-test";

    fn repo() -> MemoryRepository {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo
    }

    fn seed(repo: &MemoryRepository, now: i64) {
        repo.create_episodic(&EpisodicMemory {
            id: "hot".into(),
            project_id: P.into(),
            session_id: "s".into(),
            summary: "adopted often".into(),
            content: "served 4 times, read 3 times".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: now,
            updated_at: now,
        })
        .unwrap();
        repo.create_episodic(&EpisodicMemory {
            id: "cold".into(),
            project_id: P.into(),
            session_id: "s".into(),
            summary: "never adopted".into(),
            content: "served 6 times, read 0 times".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: now,
            updated_at: now,
        })
        .unwrap();
        repo.create_failure(&FailureMemory {
            id: "f-hot".into(),
            project_id: P.into(),
            incident: "adopted failure".into(),
            root_cause: "rc".into(),
            fix: "fix".into(),
            prevention: "prev".into(),
            severity: 3,
            tags: vec![],
            created_at: now,
            updated_at: now,
        })
        .unwrap();
    }

    /// Serve `ids` once (one query_log row), then read `adopted` of them.
    fn log_round(repo: &MemoryRepository, ids: &[&str], adopted: &[&str], now: i64) {
        let ids: Vec<String> = ids.iter().map(|s| s.to_string()).collect();
        repo.record_query(P, "q", &ids, None, now).unwrap();
        for id in adopted {
            repo.record_adoption(P, id, type_of(id), now, 24 * 3600)
                .unwrap();
        }
    }

    fn type_of(id: &str) -> &'static str {
        if id.starts_with("f-") {
            "failure"
        } else {
            "episodic"
        }
    }

    #[test]
    fn promotes_adopted_and_demotes_ignored() {
        let repo = repo();
        let now = 1_750_000_000i64;
        seed(&repo, now);

        // hot: 4 impressions, 4 adoptions (100% ≥ 50%) → promote.
        // cold: 6 impressions, 0 adoptions → demote.
        // f-hot: 4 impressions, 3 adoptions → promote (severity 3 → 4).
        for _ in 0..3 {
            log_round(&repo, &["hot", "cold", "f-hot"], &["hot", "f-hot"], now);
        }
        log_round(&repo, &["hot", "cold", "f-hot"], &["hot"], now);
        for _ in 0..2 {
            log_round(&repo, &["cold"], &[], now);
        }

        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert_eq!(report.proposals.len(), 3, "{:?}", report.proposals);

        let by_id: std::collections::HashMap<&str, &TuneProposal> = report
            .proposals
            .iter()
            .map(|p| (p.id.as_str(), p))
            .collect();

        let hot = by_id["hot"];
        assert_eq!(hot.action, "promote");
        assert!((hot.proposed_importance - 0.6).abs() < 1e-6);
        assert_eq!(hot.impressions, 4);
        assert_eq!(hot.adoptions, 4);

        let cold = by_id["cold"];
        assert_eq!(cold.action, "demote");
        assert!((cold.proposed_importance - 0.4).abs() < 1e-6);
        assert_eq!(cold.impressions, 6);

        let f_hot = by_id["f-hot"];
        assert_eq!(f_hot.action, "promote");
        assert!((f_hot.current_importance - 0.6).abs() < 1e-6); // severity 3 / 5

        // Dry-run purity: nothing written yet.
        assert!((repo.get_episodic("hot", P).unwrap().unwrap().importance - 0.5).abs() < 1e-6);

        // Apply: importance moves; failure severity follows the 0.1→1 mapping.
        let applied = apply_proposals(&repo, P, &report.proposals, now).unwrap();
        assert_eq!(applied, 3);
        assert!((repo.get_episodic("hot", P).unwrap().unwrap().importance - 0.6).abs() < 1e-6);
        assert!((repo.get_episodic("cold", P).unwrap().unwrap().importance - 0.4).abs() < 1e-6);
        assert_eq!(repo.get_failure("f-hot", P).unwrap().unwrap().severity, 4);
    }

    #[test]
    fn below_thresholds_proposes_nothing() {
        let repo = repo();
        let now = 1_750_000_000i64;
        seed(&repo, now);

        // 2 impressions only — under every threshold.
        log_round(&repo, &["hot", "cold"], &["hot"], now);
        log_round(&repo, &["hot", "cold"], &["hot"], now);
        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert!(report.proposals.is_empty(), "{:?}", report.proposals);
    }

    #[test]
    fn respects_bounds_and_window() {
        let repo = repo();
        let now = 1_750_000_000i64;
        seed(&repo, now);

        // A high-importance memory adopted often: already at/near the ceiling
        // → no promotion available.
        let mut hi = repo.get_episodic("hot", P).unwrap().unwrap();
        hi.importance = 0.9;
        repo.update_episodic(&hi).unwrap();
        for _ in 0..4 {
            log_round(&repo, &["hot"], &["hot"], now);
        }
        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert!(report.proposals.is_empty(), "{:?}", report.proposals);

        // Old evidence outside the window is invisible to the tuner.
        let old = now - 90 * 86_400;
        repo.record_query(P, "old", &["cold".to_string()], None, old)
            .unwrap();
        repo.record_adoption(P, "cold", "episodic", old, 24 * 3600)
            .unwrap();
        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert!(report.proposals.is_empty(), "{:?}", report.proposals);
    }

    #[test]
    fn archived_memories_are_skipped() {
        let repo = repo();
        let now = 1_750_000_000i64;
        seed(&repo, now);
        for _ in 0..4 {
            log_round(&repo, &["hot"], &["hot"], now);
        }
        repo.archive(MemoryKind::Episodic, "hot", P, now).unwrap();
        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert!(report.proposals.is_empty(), "{:?}", report.proposals);
    }
}
