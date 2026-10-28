//! Feedback-driven importance tuning — closes the adoption-feedback loop that
//! v0.5 opened. `mark_relevance` and `get_memory` produce per-event adoption
//! signals; until now nothing consumed them automatically (the ±0.1 manual
//! bump was the only effect). This module turns accumulated usage evidence
//! into bounded, reviewable importance adjustments:
//!
//! - **promote**: a memory SERVED on ≥3 distinct UTC days whose daily
//!   adoption rate is ≥50% and whose importance is still below 0.9 → +0.1.
//!   It keeps proving useful but the ranking doesn't yet reflect that.
//! - **demote**: a memory served on ≥5 distinct days, never adopted,
//!   importance above 0.2, and no explicit 'useful' feedback in the window
//!   → −0.1. It keeps surfacing and being ignored.
//!
//! Both sides of the rate are deduplicated to UTC calendar days
//! (`adopted_days / served_days`). The adoption numerator was already
//! deduplicated per 24h while the impression denominator counted every
//! search, so a same-day search→read→re-search cycle could only depress the
//! rate — the most-served memories were systematically the hardest to
//! promote. Symmetric day-dedup fixes that; raw counts stay in the
//! proposals for human review.
//!
//! Failure memories are never auto-tuned: severity is an operational record
//! of incident impact, not a ranking signal — "rarely read" must not rewrite
//! it. They are counted in `failures_skipped` for manual review instead.
//!
//! Deliberately conservative: fixed ±0.1 steps (the same size as a manual
//! `mark_relevance`), clamped bounds, and — like `consolidate`/`reflect` —
//! dry-run by default. `engram tune` previews; `--apply` writes.

use crate::storage::{MemoryKind, MemoryRepository};

use anyhow::Result;
use serde::Serialize;

/// Distinct UTC days a memory must appear in result sets before it is
/// eligible for promotion. Day-granular on purpose: the adoption numerator
/// is deduplicated per day, so the denominator must match — otherwise a
/// busy day of re-searches could only hurt the rate.
const MIN_SERVED_DAYS_PROMOTE: i64 = 3;
/// Distinct UTC days before a never-adopted memory is eligible for demotion
/// — higher than the promote floor: absence of evidence is weaker evidence.
const MIN_SERVED_DAYS_DEMOTE: i64 = 5;
/// Daily adoption rate (adopted_days / served_days) that qualifies as "keeps
/// proving useful".
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
    /// Raw serve occurrences over the window — display only; eligibility
    /// and the rate use the day-deduplicated counts below.
    pub impressions: i64,
    /// Raw adoption events over the window — display only.
    pub adoptions: i64,
    /// Distinct UTC days the memory appeared in result sets (rate
    /// denominator).
    pub served_days: i64,
    /// Distinct UTC days with an adoption event (rate numerator).
    pub adopted_days: i64,
    /// adopted_days / served_days, rounded for display.
    pub adoption_rate: f64,
}

/// Tuning pass outcome: the proposals plus the raw signal totals.
#[derive(Debug, Clone, Serialize)]
pub struct TuneReport {
    pub applied: bool,
    pub window_queries: usize,
    pub proposals: Vec<TuneProposal>,
    /// Failure candidates excluded from tuning: severity is an operational
    /// record of incident impact, adjusted (if ever) by humans — not by
    /// "wasn't read often". Surfaced so the CLI can point at them.
    pub failures_skipped: usize,
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

    // Impressions: parse each logged result set, tracking both the raw serve
    // count (display) and the distinct UTC days served (the rate
    // denominator, deduplicated to match the adoption numerator).
    let mut impressions: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut served_days: std::collections::HashMap<String, std::collections::HashSet<i64>> =
        std::collections::HashMap::new();
    let result_sets = repo.recent_result_sets(project_id, since)?;
    let query_count = result_sets.len();
    for (ts, raw) in &result_sets {
        // div_euclid (floor division): correct UTC-day bucket for any sign
        // of `ts`; plain `/` truncates toward zero.
        let day = ts.div_euclid(86_400);
        if let Ok(ids) = serde_json::from_str::<Vec<String>>(raw) {
            for id in ids {
                *impressions.entry(id.clone()).or_default() += 1;
                served_days.entry(id).or_default().insert(day);
            }
        }
    }
    let mut adoptions: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut adopted_days: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    for (id, cnt, days_cnt) in repo.adoption_day_counts(project_id, since)? {
        adoptions.insert(id.clone(), cnt);
        adopted_days.insert(id, days_cnt);
    }

    let mut proposals = Vec::new();
    let mut failures_skipped = 0usize;
    // Active-id sets per kind, loaded lazily once and reused across candidates.
    let mut active_cache: std::collections::HashMap<MemoryKind, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    let mut candidates: Vec<(String, i64, i64, i64, i64)> = served_days
        .iter()
        .map(|(id, days)| {
            (
                id.clone(),
                impressions.get(id).copied().unwrap_or(0),
                adoptions.get(id).copied().unwrap_or(0),
                days.len() as i64,
                adopted_days.get(id).copied().unwrap_or(0),
            )
        })
        .collect();
    candidates.sort();
    for (memory_id, imp_cnt, adopt_cnt, served, adopted) in candidates {
        if served < MIN_SERVED_DAYS_PROMOTE {
            continue;
        }

        // Resolve kind + current importance, skipping anything not active
        // (get_* does not filter archived rows).
        let Some((kind, importance)) =
            resolve_active_importance(repo, project_id, &memory_id, &mut active_cache)?
        else {
            continue;
        };
        // Severity is an operational record of incident impact — the tuner
        // must not rewrite it because a failure "wasn't read often".
        if matches!(kind, MemoryKind::Failure) {
            failures_skipped += 1;
            continue;
        }
        // Cap at 1.0 defensively: adoption events are deduplicated per
        // memory per window (see record_adoption), but a legacy DB can still
        // carry pre-dedup rows where read days outcounted served days.
        let rate = (adopted as f64 / served as f64).min(1.0);

        let (action, target) = if rate >= PROMOTE_ADOPTION_RATE && importance < PROMOTE_CEILING {
            ("promote", (importance + STEP).min(PROMOTE_CEILING))
        } else if adopted == 0
            && served >= MIN_SERVED_DAYS_DEMOTE
            && importance > DEMOTE_FLOOR
            // Explicit 'useful' feedback in the window vetoes demotion: an
            // agent endorsement outranks "wasn't read". Checked last so the
            // query only runs for otherwise demote-eligible memories.
            && !repo.has_useful_feedback_since(project_id, &memory_id, since)?
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
            impressions: imp_cnt,
            adoptions: adopt_cnt,
            served_days: served,
            adopted_days: adopted,
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
        failures_skipped,
    })
}

/// Apply proposals: writes importance via the same update path
/// `mark_relevance` uses. Returns the number applied; ids that vanished in
/// the meantime are skipped, not fatal.
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
            // compute_proposals never emits failure proposals (severity is
            // an operational record, not a ranking knob) — defensive skip
            // for hand-crafted inputs.
            MemoryKind::Failure => continue,
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

        // hot: served 4 days, adopted 4 days (100% ≥ 50%) → promote.
        // cold: served 6 days, never adopted → demote.
        // f-hot: same evidence as hot, but a FAILURE → skipped and counted.
        // Rounds are spaced 25h apart: one adoption event per memory per 24h
        // window (anti-gaming dedup in record_adoption) also means one per
        // UTC day, so rounds on one timestamp would collapse to a single day.
        for i in 0..3 {
            log_round(
                &repo,
                &["hot", "cold", "f-hot"],
                &["hot", "f-hot"],
                now + i * 25 * 3600,
            );
        }
        log_round(
            &repo,
            &["hot", "cold", "f-hot"],
            &["hot"],
            now + 3 * 25 * 3600,
        );
        for i in 0..2 {
            log_round(&repo, &["cold"], &[], now + (4 + i) * 25 * 3600);
        }

        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert_eq!(report.proposals.len(), 2, "{:?}", report.proposals);
        assert_eq!(report.failures_skipped, 1);

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
        assert_eq!(hot.served_days, 4);
        assert_eq!(hot.adopted_days, 4);

        let cold = by_id["cold"];
        assert_eq!(cold.action, "demote");
        assert!((cold.proposed_importance - 0.4).abs() < 1e-6);
        assert_eq!(cold.impressions, 6);
        assert_eq!(cold.served_days, 6);

        // Dry-run purity: nothing written yet — including failure severity.
        assert!((repo.get_episodic("hot", P).unwrap().unwrap().importance - 0.5).abs() < 1e-6);

        // Apply: importance moves; failure severity stays an operational
        // record (never auto-adjusted).
        let applied = apply_proposals(&repo, P, &report.proposals, now).unwrap();
        assert_eq!(applied, 2);
        assert!((repo.get_episodic("hot", P).unwrap().unwrap().importance - 0.6).abs() < 1e-6);
        assert!((repo.get_episodic("cold", P).unwrap().unwrap().importance - 0.4).abs() < 1e-6);
        assert_eq!(repo.get_failure("f-hot", P).unwrap().unwrap().severity, 3);
    }

    /// Same-day re-searches must not dilute the rate: both sides of the
    /// rate are deduplicated to UTC days. Under the old per-search
    /// denominator this memory scored 3/8 = 0.375 < 0.5 and never promoted.
    #[test]
    fn same_day_serves_do_not_dilute_rate() {
        let repo = repo();
        let now = 1_750_000_000i64;
        seed(&repo, now);

        // Three distinct days, served + adopted once each...
        for i in 0..3 {
            log_round(&repo, &["hot"], &["hot"], now + i * 25 * 3600);
        }
        // ...then a same-day search→read→re-search burst (5 extra serves).
        for _ in 0..5 {
            log_round(&repo, &["hot"], &[], now + 2 * 25 * 3600);
        }

        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert_eq!(report.proposals.len(), 1, "{:?}", report.proposals);
        let hot = &report.proposals[0];
        assert_eq!(hot.id, "hot");
        assert_eq!(hot.action, "promote");
        assert_eq!(hot.impressions, 8); // raw count kept for review
        assert_eq!(hot.served_days, 3);
        assert_eq!(hot.adopted_days, 3);
        assert!((hot.adoption_rate - 1.0).abs() < 1e-6);
    }

    /// Explicit 'useful' feedback in the window vetoes auto-demotion: an
    /// agent endorsement outranks "served 5 days, never read".
    #[test]
    fn useful_feedback_vetoes_demotion() {
        let repo = repo();
        let now = 1_750_000_000i64;
        seed(&repo, now);

        for i in 0..5 {
            log_round(&repo, &["cold"], &[], now + i * 25 * 3600);
        }
        // Without feedback this demotes (the cold case above); with an
        // explicit 'useful' judgment it must not.
        repo.record_feedback(P, "cold", "episodic", "useful", now + 3 * 25 * 3600)
            .unwrap();

        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert!(report.proposals.is_empty(), "{:?}", report.proposals);
    }

    #[test]
    fn below_thresholds_proposes_nothing() {
        let repo = repo();
        let now = 1_750_000_000i64;
        seed(&repo, now);

        // Two same-day rounds: one served DAY — under every threshold
        // (day-deduplicated, so volume within a day cannot compensate).
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

        // A high-importance memory adopted every day: already at the ceiling
        // → no promotion available (needs distinct days to be eligible at
        // all, hence the 25h spacing).
        let mut hi = repo.get_episodic("hot", P).unwrap().unwrap();
        hi.importance = 0.9;
        repo.update_episodic(&hi).unwrap();
        for i in 0..4 {
            log_round(&repo, &["hot"], &["hot"], now + i * 25 * 3600);
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
        // Enough distinct days that the memory WOULD promote if active.
        for i in 0..4 {
            log_round(&repo, &["hot"], &["hot"], now + i * 25 * 3600);
        }
        repo.archive(MemoryKind::Episodic, "hot", P, now).unwrap();
        let report = compute_proposals(&repo, P, 30, now).unwrap();
        assert!(report.proposals.is_empty(), "{:?}", report.proposals);
    }

    // ─── Longitudinal feedback-loop simulation (batch-4 regression) ──
    // The batch-1 fixes each have a unit test above; this pins them as a
    // CLOSED LOOP over a 30-day sequence: memories served at rotating
    // ranks → some adopted → a few explicitly endorsed → `engram tune`
    // applies every 5 days → serving continues. Any of the batch-1
    // regressions (rate dilution by same-day re-searches, demote guard
    // loss, failure severity tuning, ceiling/floor overrun) breaks one of
    // the invariants below.

    #[test]
    fn thirty_day_feedback_loop_simulation() {
        use crate::config::Config;

        let repo = repo();
        const DAY: i64 = 86_400;
        let t0 = 1_750_000_000i64; // not day-aligned; +3600 keeps each day's
                                   // events inside ONE UTC day bucket
        let at = |day: i64| t0 + day * DAY + 3_600;

        for (id, summary) in [
            ("rising", "adopted often"),
            ("tail-a", "mid tail never read a"),
            ("tail-b", "mid tail never read b"),
            ("endorsed", "served never read but endorsed"),
        ] {
            repo.create_episodic(&EpisodicMemory {
                id: id.into(),
                project_id: P.into(),
                session_id: "sim".into(),
                summary: summary.into(),
                content: summary.into(),
                files_touched: vec![],
                related_commits: vec![],
                importance: 0.5,
                tags: vec![],
                created_at: t0,
                updated_at: t0,
            })
            .unwrap();
        }
        repo.create_failure(&FailureMemory {
            id: "f-sev".into(),
            project_id: P.into(),
            incident: "severity must never move".into(),
            root_cause: "rc".into(),
            fix: "fix".into(),
            prevention: "prev".into(),
            severity: 4,
            tags: vec![],
            created_at: t0,
            updated_at: t0,
        })
        .unwrap();

        let mut rising_history = vec![0.5f32];
        let mut tail_history = vec![0.5f32];
        let mut endorsed_history = vec![0.5f32];

        for day in 0..30i64 {
            // Serve a 5-result page; rank order rotates day over day so the
            // loop sees the memories at different positions.
            let base = ["rising", "tail-a", "tail-b", "endorsed", "f-sev"];
            let page: Vec<String> = (0..5)
                .map(|i| base[(i + day as usize) % 5].to_string())
                .collect();
            repo.record_query(P, "sim", &page, None, at(day)).unwrap();
            // rising is adopted every other day (rate ≥ 0.5 forever);
            // f-sev mirrors it — eligible evidence the failure skip must
            // ignore. Explicit 'useful' feedback every third day guards
            // endorsed (mark_relevance-equivalent judgment).
            if day % 2 == 0 {
                repo.record_adoption(P, "rising", "episodic", at(day), DAY)
                    .unwrap();
            }
            repo.record_adoption(P, "f-sev", "failure", at(day), DAY)
                .unwrap();
            if day % 3 == 0 {
                repo.record_feedback(P, "endorsed", "episodic", "useful", at(day))
                    .unwrap();
            }
            // Day 10: a same-day re-search burst (5 extra pages, no
            // adoptions) — under the pre-batch-1 per-search denominator
            // this diluted rising's rate below the promote threshold.
            if day == 10 {
                for _ in 0..5 {
                    repo.record_query(P, "sim", &page, None, at(day)).unwrap();
                }
            }

            // `engram tune --apply` every 5 days (after the day's events).
            if (day + 1) % 5 == 0 {
                let report = compute_proposals(&repo, P, 30, at(day)).unwrap();
                if day == 14 {
                    // The tune right after the burst: rising must STILL
                    // promote, with the rate computed over distinct DAYS
                    // (8 adopted days / 15 served days; impressions carry
                    // the raw 20 serves for human review).
                    let rising = report
                        .proposals
                        .iter()
                        .find(|p| p.id == "rising")
                        .expect("same-day burst must not block promotion");
                    assert_eq!(rising.action, "promote");
                    assert_eq!(rising.served_days, 15);
                    assert_eq!(rising.impressions, 20);
                    assert!((rising.adoption_rate - 8.0 / 15.0).abs() < 0.005);
                }
                apply_proposals(&repo, P, &report.proposals, at(day)).unwrap();

                let rising_now = repo.get_episodic("rising", P).unwrap().unwrap().importance;
                assert!(
                    rising_now >= *rising_history.last().unwrap() - 1e-6,
                    "adopted memory importance must be monotone non-decreasing: \
                     {} -> {rising_now}",
                    rising_history.last().unwrap()
                );
                assert!(
                    rising_now <= 0.9 + 1e-6,
                    "promote ceiling overrun: {rising_now}"
                );
                rising_history.push(rising_now);

                for tail in ["tail-a", "tail-b"] {
                    let now_imp = repo.get_episodic(tail, P).unwrap().unwrap().importance;
                    assert!(
                        now_imp >= 0.2 - 1e-6,
                        "{tail} sank below the demote floor: {now_imp}"
                    );
                    tail_history.push(now_imp);
                }
                let endorsed_now = repo
                    .get_episodic("endorsed", P)
                    .unwrap()
                    .unwrap()
                    .importance;
                assert!(
                    endorsed_now >= *endorsed_history.last().unwrap() - 1e-6,
                    "useful-feedback veto failed: {} -> {endorsed_now}",
                    endorsed_history.last().unwrap()
                );
                endorsed_history.push(endorsed_now);

                assert!(
                    report.failures_skipped >= 1,
                    "served+adopted failure must land in failures_skipped"
                );
                assert_eq!(
                    repo.get_failure("f-sev", P).unwrap().unwrap().severity,
                    4,
                    "failure severity must never be auto-tuned"
                );
            }
        }

        // The loop actually closed: rising promoted to the ceiling in ≤4
        // tunes (0.5 → 0.9), tails parked AT the floor without going under,
        // endorsed never moved.
        assert!((rising_history.last().unwrap() - 0.9).abs() < 1e-6);
        assert!((tail_history.last().unwrap() - 0.2).abs() < 1e-6);
        assert!(endorsed_history.iter().all(|v| (*v - 0.5).abs() < 1e-6));

        // And retrieval still works over the tuned store (the "search
        // again" half of the loop) — ranking consumes the tuned importances
        // without breaking.
        let provider = crate::mcp::server::DefaultMemoryProvider::new(repo, Config::default());
        let results = provider
            .search_core("adopted often", P, None, &[], None, 5)
            .unwrap();
        assert!(
            results.iter().any(|r| r.id == "rising"),
            "post-tune search must still surface the promoted memory"
        );
    }
}
