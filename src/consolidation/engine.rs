use crate::storage::{MemoryKind, MemoryRepository};
use anyhow::Result;
use rusqlite::params;
use std::collections::{HashMap, HashSet};

pub const DEFAULT_JACCARD_THRESHOLD: f64 = 0.85;

// ─── Session-import template scaffold ─────────────────────────────────
//
// `engram session-import` stitches every imported episodic memory's content
// from these fixed phrases (see `cli::commands::session_import_content`,
// which builds from THESE constants). They are scaffold, not session
// content: every import carries them verbatim, so near-dup comparison must
// exclude them — two templated imports from DIFFERENT sessions share ~15
// scaffold tokens, and with a single real word of difference the Jaccard
// cleared the default 0.85 threshold (s / (s + 2r) >= 0.85 holds at r <= 1),
// silently archiving one of them. One source of truth: the builder and the
// exclusion list below must not drift apart.
pub const SESSION_IMPORT_TPL_PROMPTS: &str = "User prompts:";
pub const SESSION_IMPORT_TPL_CONCLUSIONS: &str = "Assistant conclusions (latest last):";
pub const SESSION_IMPORT_TPL_FILES: &str = "Files touched";
pub const SESSION_IMPORT_TPL_ERRORS: &str = "Errors encountered";
pub const SESSION_IMPORT_TPL_FAILURE_CANDIDATES: &str = "failure-memory candidates";
pub const SESSION_IMPORT_TEMPLATE_PHRASES: &[&str] = &[
    SESSION_IMPORT_TPL_PROMPTS,
    SESSION_IMPORT_TPL_CONCLUSIONS,
    SESSION_IMPORT_TPL_FILES,
    SESSION_IMPORT_TPL_ERRORS,
    SESSION_IMPORT_TPL_FAILURE_CANDIDATES,
];

/// Punctuation-trimmed, lowercased forms of the template phrases above.
/// `scaffold_words_cover_template_phrases` pins the derivation.
pub(crate) const SESSION_IMPORT_SCAFFOLD_WORDS: &[&str] = &[
    "user",
    "prompts",
    "assistant",
    "conclusions",
    "latest",
    "last",
    "files",
    "touched",
    "errors",
    "encountered",
    "failure-memory",
    "candidates",
];

/// Whether a whitespace token is template scaffold rather than content:
/// its trimmed form is one of [`SESSION_IMPORT_SCAFFOLD_WORDS`], or it is
/// pure punctuation / digits (the "-" list bullets, the "—" dash, the
/// "(3):" section counts — numbers of prompts/files, not session content).
pub(crate) fn is_session_import_scaffold(token: &str) -> bool {
    let trimmed = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
    if !trimmed.chars().any(|c| c.is_alphabetic()) {
        return true;
    }
    let norm = trimmed.to_lowercase();
    SESSION_IMPORT_SCAFFOLD_WORDS.contains(&norm.as_str())
}

/// One group of duplicate memories: keeper kept active, others to be archived.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsolidationGroup {
    pub keeper_id: String,
    pub duplicate_ids: Vec<String>,
    pub reason: String,
}

/// Consolidation result for a single memory kind.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsolidationPlan {
    pub memory_type: String,
    pub groups: Vec<ConsolidationGroup>,
    pub archived: usize,
}

/// Consolidation engine: exact content-hash dedup + near-duplicate detection
/// (word-set Jaccard; step-sequence similarity for procedures).
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

    /// Compute a deterministic content hash for deduplication.
    /// Uses FNV-1a hash which is stable across Rust versions,
    /// unlike `DefaultHasher` which provides no stability guarantees.
    pub fn content_hash(summary: &str, content: &str) -> String {
        let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
        for byte in summary.bytes().chain(content.bytes()) {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3); // FNV prime
        }
        format!("{:016x}", hash)
    }

    /// Build a consolidation plan for one kind: groups of exact (and optionally
    /// near-) duplicate active memories. keeper = earliest created. Does NOT mutate.
    pub fn plan_for_kind(
        &self,
        repo: &MemoryRepository,
        project_id: &str,
        kind: MemoryKind,
        include_near_dup: bool,
        threshold: f64,
    ) -> Result<ConsolidationPlan> {
        // (id, canonical dedup text) for active memories, oldest first. The
        // dedup text is assembled in Rust with the same per-kind builders the
        // content_hash write path uses (storage::repository::dedup_text_*) —
        // the old SQL `dedup_text_expr` concatenation is gone, so the hash
        // column and this scan share ONE formula.
        use crate::storage::repository::{
            dedup_text_decision, dedup_text_episodic, dedup_text_failure, dedup_text_procedural,
        };
        let (sql, text_of): (String, fn(&rusqlite::Row<'_>) -> rusqlite::Result<String>) =
            match kind {
                MemoryKind::Episodic => (
                    format!(
                        "SELECT id, summary, content FROM {} \
                         WHERE project_id = ?1 AND archived_at IS NULL ORDER BY created_at ASC",
                        kind.table()
                    ),
                    |row| {
                        Ok(dedup_text_episodic(
                            &row.get::<_, String>(1)?,
                            &row.get::<_, String>(2)?,
                        ))
                    },
                ),
                MemoryKind::Decision => (
                    format!(
                        "SELECT id, title, context, rationale, tradeoffs FROM {} \
                         WHERE project_id = ?1 AND archived_at IS NULL ORDER BY created_at ASC",
                        kind.table()
                    ),
                    |row| {
                        Ok(dedup_text_decision(
                            &row.get::<_, String>(1)?,
                            &row.get::<_, String>(2)?,
                            &row.get::<_, String>(3)?,
                            &row.get::<_, String>(4)?,
                        ))
                    },
                ),
                MemoryKind::Failure => (
                    format!(
                        "SELECT id, incident, root_cause, fix, prevention FROM {} \
                         WHERE project_id = ?1 AND archived_at IS NULL ORDER BY created_at ASC",
                        kind.table()
                    ),
                    |row| {
                        Ok(dedup_text_failure(
                            &row.get::<_, String>(1)?,
                            &row.get::<_, String>(2)?,
                            &row.get::<_, String>(3)?,
                            &row.get::<_, String>(4)?,
                        ))
                    },
                ),
                MemoryKind::Procedural => (
                    format!(
                        "SELECT id, workflow_name, steps FROM {} \
                         WHERE project_id = ?1 AND archived_at IS NULL ORDER BY created_at ASC",
                        kind.table()
                    ),
                    |row| {
                        Ok(dedup_text_procedural(
                            &row.get::<_, String>(1)?,
                            &row.get::<_, String>(2)?,
                        ))
                    },
                ),
            };
        let conn = repo.connection()?;
        let mut stmt = conn.prepare(&sql)?;
        let recs: Vec<(String, String)> = stmt
            .query_map(params![project_id], |row| {
                Ok((row.get::<_, String>(0)?, text_of(row)?))
            })?
            .collect::<std::result::Result<_, _>>()?;
        drop(stmt);

        // Near-duplicate comparison key, materialized ONCE per record. The
        // previous implementation rebuilt both records' word HashSets for
        // every pair — with N same-scale memories that is N²/2 set builds
        // (the O(N²) pairing is inherent; the per-pair allocation was not).
        //
        // Procedural memories compare by STEP SEQUENCE, not word set: steps
        // are order-sensitive (`test → build → deploy` is not `deploy → test`),
        // but Jaccard is order-blind — two procedures with identical steps in
        // different order scored 1.0 and one got archived. Sequence
        // similarity (LCS-based) preserves the order signal.
        enum NearKey {
            Words(HashSet<String>, usize),
            Steps(Vec<String>),
        }
        let near_key = |kind: MemoryKind, text: &str| -> NearKey {
            if kind == MemoryKind::Procedural {
                // dedup text = "workflow_name\n[step json]"; recover the steps.
                let steps: Vec<String> = text
                    .split_once('\n')
                    .and_then(|(_, json)| serde_json::from_str(json).ok())
                    .unwrap_or_default();
                NearKey::Steps(steps)
            } else {
                // Episodic only: drop the session-import template scaffold
                // (see the constants above) so templated imports are compared
                // on session content alone. Decision/failure texts never
                // carry the template, and excluding plain English words
                // ("user", "files") there would change their near-dup
                // semantics for no benefit.
                let exclude_scaffold = kind == MemoryKind::Episodic;
                let mut kept = 0usize;
                let set: HashSet<String> = text
                    .split_whitespace()
                    .filter(|w| {
                        let keep = !(exclude_scaffold && is_session_import_scaffold(w));
                        if keep {
                            kept += 1;
                        }
                        keep
                    })
                    .map(str::to_string)
                    .collect();
                NearKey::Words(set, kept.max(1))
            }
        };
        let keys: Vec<NearKey> = recs.iter().map(|(_, t)| near_key(kind, t)).collect();

        let mut grouped: HashSet<usize> = HashSet::new();
        let mut groups: Vec<ConsolidationGroup> = Vec::new();

        // Exact duplicates by content hash.
        let mut by_hash: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, (_, text)) in recs.iter().enumerate() {
            by_hash
                .entry(Self::content_hash(text, ""))
                .or_default()
                .push(i);
        }
        for idxs in by_hash.values() {
            if idxs.len() > 1 {
                let keeper = idxs[0]; // earliest (recs sorted asc)
                let dups: Vec<String> = idxs[1..].iter().map(|&j| recs[j].0.clone()).collect();
                for &j in idxs {
                    grouped.insert(j);
                }
                groups.push(ConsolidationGroup {
                    keeper_id: recs[keeper].0.clone(),
                    duplicate_ids: dups,
                    reason: "exact".into(),
                });
            }
        }

        // Optional near-duplicates among the still-ungrouped, by Jaccard
        // (word set) or step-sequence similarity for procedures.
        if include_near_dup {
            let remaining: Vec<usize> = (0..recs.len()).filter(|i| !grouped.contains(i)).collect();
            for (pos, &a) in remaining.iter().enumerate() {
                if grouped.contains(&a) {
                    continue;
                }
                let mut dups = Vec::new();
                for &b in &remaining[pos + 1..] {
                    if grouped.contains(&b) {
                        continue;
                    }
                    let similar = match (&keys[a], &keys[b]) {
                        // Cheap length pre-filter: Jaccard of word sets
                        // cannot reach a >=0.6 threshold when one text has
                        // >2.5x the words of the other.
                        (NearKey::Words(set_a, words_a), NearKey::Words(set_b, words_b)) => {
                            if *words_a.max(words_b) as f64 > 2.5 * *words_a.min(words_b) as f64 {
                                false
                            } else {
                                set_similar(set_a, set_b, threshold)
                            }
                        }
                        // Step lists: same length pre-filter on step counts.
                        (NearKey::Steps(sa), NearKey::Steps(sb)) => {
                            if sa.len().max(sb.len()) as f64
                                > 2.5 * sa.len().min(sb.len()).max(1) as f64
                            {
                                false
                            } else {
                                sequence_similar(sa, sb, threshold)
                            }
                        }
                        // Mixed keys cannot happen (per-kind comparison), but
                        // fall back to false rather than panic if it ever did.
                        _ => false,
                    };
                    if similar {
                        dups.push(recs[b].0.clone());
                        grouped.insert(b);
                    }
                }
                if !dups.is_empty() {
                    grouped.insert(a);
                    groups.push(ConsolidationGroup {
                        keeper_id: recs[a].0.clone(),
                        duplicate_ids: dups,
                        reason: if kind == MemoryKind::Procedural {
                            format!("near(sequence>={threshold})")
                        } else {
                            format!("near(jaccard>={threshold})")
                        },
                    });
                }
            }
        }

        Ok(ConsolidationPlan {
            memory_type: kind.as_str().into(),
            groups,
            archived: 0,
        })
    }

    /// Plan (and optionally apply) consolidation across the given kinds.
    /// `apply=false` → dry-run (no mutation). `apply=true` → archive duplicates.
    #[allow(clippy::too_many_arguments)]
    pub fn consolidate(
        &self,
        repo: &MemoryRepository,
        project_id: &str,
        kinds: &[MemoryKind],
        include_near_dup: bool,
        threshold: f64,
        apply: bool,
        now: i64,
    ) -> Result<Vec<ConsolidationPlan>> {
        let mut plans = Vec::new();
        for &kind in kinds {
            let mut plan =
                self.plan_for_kind(repo, project_id, kind, include_near_dup, threshold)?;
            if apply {
                let mut archived = 0;
                for group in &plan.groups {
                    for dup_id in &group.duplicate_ids {
                        // Best-effort: a single archive failure (e.g. SQLITE_BUSY)
                        // must not abort the whole pass and leave a half-applied
                        // intermediate state. Log and continue; `archived` stays
                        // an accurate count of what was actually archived.
                        match repo.archive(kind, dup_id, project_id, now) {
                            Ok(true) => archived += 1,
                            Ok(false) => {}
                            Err(e) => tracing::warn!(
                                kind = kind.as_str(),
                                id = %dup_id,
                                error = %e,
                                "consolidate: failed to archive duplicate, skipping"
                            ),
                        }
                    }
                }
                plan.archived = archived;
            }
            plans.push(plan);
        }
        Ok(plans)
    }
}

/// Jaccard over prebuilt word sets: |A ∩ B| / |A ∪ B| >= threshold, with
/// empty-set equality. Module-level so `plan_for_kind` and the tests below
/// exercise the SAME comparison (the old string-based `jaccard_similarity`
/// existed only for tests and read as dead code once made private).
fn set_similar(a: &HashSet<String>, b: &HashSet<String>, threshold: f64) -> bool {
    if a.is_empty() && b.is_empty() {
        return true;
    }
    let intersection = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    union > 0.0 && (intersection / union) >= threshold
}

/// Order-aware sequence similarity: 2·LCS / (len_a + len_b).
/// Classic DP over short step lists (< 50 steps typical).
fn sequence_similar(a: &[String], b: &[String], threshold: f64) -> bool {
    if a.is_empty() && b.is_empty() {
        return true;
    }
    if a.is_empty() || b.is_empty() {
        return false;
    }
    let n = a.len();
    let m = b.len();
    let mut dp = vec![0u32; m + 1];
    for i in 1..=n {
        let mut prev_diag = 0;
        for j in 1..=m {
            let tmp = dp[j];
            dp[j] = if a[i - 1] == b[j - 1] {
                prev_diag + 1
            } else {
                dp[j].max(dp[j - 1])
            };
            prev_diag = tmp;
        }
    }
    let lcs = dp[m] as f64;
    (2.0 * lcs / (n + m) as f64) >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    // MemoryKind/MemoryRepository come via `use super::*` once Step 3b adds the
    // top-level import; only models need an explicit use here.
    use crate::models::{DecisionMemory, EpisodicMemory};

    fn repo_with_dupes() -> MemoryRepository {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk = |summary: &str, content: &str, ts: i64| EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: summary.into(),
            content: content.into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: ts,
            updated_at: ts,
        };
        // 两条完全相同（精确重复），keeper 应为更早的 ts=100。
        repo.create_episodic(&mk("same", "body", 100)).unwrap();
        repo.create_episodic(&mk("same", "body", 200)).unwrap();
        // 一条独立。
        repo.create_episodic(&mk("unique", "other", 300)).unwrap();
        repo
    }

    #[test]
    fn test_consolidate_dry_run_reports_but_does_not_archive() {
        let repo = repo_with_dupes();
        let engine = ConsolidationEngine::new();
        let plans = engine
            .consolidate(&repo, "p", &[MemoryKind::Episodic], false, 0.85, false, 999)
            .unwrap();
        // 一组精确重复，duplicate_ids 长度 1。
        let groups: Vec<_> = plans.iter().flat_map(|p| &p.groups).collect();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].duplicate_ids.len(), 1);
        // dry-run：数据未变，search 仍能搜到（两条 "same" 都在）。
        assert_eq!(repo.search_episodic("same", "p", 10).unwrap().len(), 2);
    }

    #[test]
    fn test_consolidate_apply_archives_duplicates_keeps_earliest() {
        let repo = repo_with_dupes();
        let engine = ConsolidationEngine::new();
        let plans = engine
            .consolidate(&repo, "p", &[MemoryKind::Episodic], false, 0.85, true, 999)
            .unwrap();
        let group = &plans[0].groups[0];
        // keeper 仍活跃，且必须是最早创建的那条（ts=100，而非 ts=200）。
        let keeper = repo.get_episodic(&group.keeper_id, "p").unwrap().unwrap();
        assert_eq!(
            keeper.created_at, 100,
            "keeper must be the earliest (ts=100)"
        );
        assert_eq!(repo.search_episodic("same", "p", 10).unwrap().len(), 1);
        assert_eq!(
            repo.list_archived(MemoryKind::Episodic, "p", 10)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_content_hash_deterministic() {
        let h1 = ConsolidationEngine::content_hash("summary", "content");
        let h2 = ConsolidationEngine::content_hash("summary", "content");
        assert_eq!(h1, h2);

        let h3 = ConsolidationEngine::content_hash("other", "content");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_set_similar_over_word_sets() {
        let words = |s: &str| s.split_whitespace().map(str::to_string).collect();
        assert!(set_similar(
            &words("hello world foo"),
            &words("hello world bar"),
            0.4,
        ));
        assert!(!set_similar(
            &words("completely different text"),
            &words("totally unrelated words"),
            0.5,
        ));
    }

    #[test]
    fn test_set_similar_identical() {
        let words = |s: &str| s.split_whitespace().map(str::to_string).collect();
        assert!(set_similar(&words("same text"), &words("same text"), 0.99));
    }

    #[test]
    fn scaffold_words_cover_template_phrases() {
        // Every whitespace token of every template phrase (as the session-
        // import builder actually emits them, punctuation included) must be
        // classified as scaffold — otherwise the builder and the exclusion
        // list have drifted apart and templated imports regain phantom
        // shared words.
        for phrase in SESSION_IMPORT_TEMPLATE_PHRASES {
            for token in phrase.split_whitespace() {
                assert!(
                    is_session_import_scaffold(token),
                    "template token {token:?} (from {phrase:?}) is not excluded from \
                     near-dup word sets"
                );
            }
        }
        // Real content words must NOT be excluded.
        for token in ["revert", "src/auth.rs", "panic:", "timeout"] {
            assert!(!is_session_import_scaffold(token), "{token:?} is content");
        }
    }
    // 构造一条 episodic 记录（可指定 project_id），便于跨项目/近义场景复用。
    fn mk_episodic(summary: &str, content: &str, ts: i64, project_id: &str) -> EpisodicMemory {
        EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: project_id.into(),
            session_id: "s".into(),
            summary: summary.into(),
            content: content.into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: ts,
            updated_at: ts,
        }
    }

    #[test]
    fn test_set_similar_threshold_boundary_ge_semantics() {
        // 两个集合：交集 2 词、并集 4 词 → Jaccard 恰好 0.5。
        // 验证 `>=` 闭区间语义：threshold=0.5 判相似，threshold=0.6 判不相似。
        let words = |s: &str| s.split_whitespace().map(str::to_string).collect();
        let a = words("alpha beta gamma delta");
        let b = words("alpha beta");
        assert!(
            set_similar(&a, &b, 0.5),
            "Jaccard 恰好等于 threshold 应判为 near-dup（闭区间）"
        );
        assert!(
            !set_similar(&a, &b, 0.6),
            "Jaccard 低于 threshold 应判为非 near-dup"
        );
    }

    #[test]
    fn test_plan_near_dup_end_to_end() {
        // 近义（非精确）重复走通 plan_for_kind 全链路：SQL→内存→grouped。
        // A={deploy,service}，B={deploy,service,config}，Jaccard=2/3≈0.667。
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo.create_episodic(&mk_episodic("deploy service", "", 100, "p"))
            .unwrap();
        repo.create_episodic(&mk_episodic("deploy service config", "", 200, "p"))
            .unwrap();

        let engine = ConsolidationEngine::new();
        let plan = engine
            .plan_for_kind(&repo, "p", MemoryKind::Episodic, true, 0.6)
            .unwrap();

        assert_eq!(plan.groups.len(), 1, "应识别出一组近义重复");
        let g = &plan.groups[0];
        assert!(
            g.reason.starts_with("near(jaccard>="),
            "近义组 reason 应形如 near(jaccard>=...)，实际：{}",
            g.reason
        );
        assert_eq!(g.duplicate_ids.len(), 1);
        // keeper 必须是更早创建的 A（ts=100）。
        let keeper = repo.get_episodic(&g.keeper_id, "p").unwrap().unwrap();
        assert_eq!(keeper.created_at, 100, "near-dup keeper 仍应为最早创建者");
        assert_eq!(plan.archived, 0, "plan_for_kind 不应归档");
    }

    #[test]
    fn procedural_near_dup_respects_step_order() {
        // Same steps, different order = different procedure. Word-set
        // Jaccard scored these 1.0 (order-blind) and archived the newer one;
        // sequence similarity (LCS) must NOT group them.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk = |steps: Vec<&str>, ts: i64| crate::models::ProceduralMemory {
            id: format!("proc-{ts}"),
            project_id: "p".into(),
            workflow_name: "release".into(),
            steps: steps.iter().map(|s| s.to_string()).collect(),
            related_tools: vec![],
            tags: vec![],
            importance: 0.5,
            created_at: ts,
            updated_at: ts,
        };
        repo.create_procedural(&mk(vec!["test", "build", "deploy"], 100))
            .unwrap();
        repo.create_procedural(&mk(vec!["deploy", "build", "test"], 200))
            .unwrap();
        // Genuine near-dup: last step reworded → LCS 2·2/(3+3) ≈ 0.67 ≥ 0.6.
        repo.create_procedural(&mk(vec!["test", "build", "publish"], 300))
            .unwrap();

        let engine = ConsolidationEngine::new();
        let plan = engine
            .plan_for_kind(&repo, "p", MemoryKind::Procedural, true, 0.6)
            .unwrap();

        assert_eq!(plan.groups.len(), 1, "{:?}", plan.groups);
        let g = &plan.groups[0];
        assert_eq!(g.keeper_id, "proc-100");
        assert_eq!(g.duplicate_ids, vec!["proc-300".to_string()]);
        assert!(
            g.reason.starts_with("near(sequence>="),
            "procedural groups use sequence similarity, got: {}",
            g.reason
        );
    }

    #[test]
    fn test_near_dup_single_group_does_not_reopen() {
        // A 与 B、C 均近义：应只产生一组（A 为 keeper，[B,C] 为 dups）。
        // 验证 grouped HashSet 的「已并入则跳过」逻辑——B/C 不会再作为 keeper 开新组。
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo.create_episodic(&mk_episodic("deploy service", "", 100, "p"))
            .unwrap(); // A
        repo.create_episodic(&mk_episodic("deploy service config", "", 200, "p"))
            .unwrap(); // B
        repo.create_episodic(&mk_episodic("deploy service runtime", "", 300, "p"))
            .unwrap(); // C

        let engine = ConsolidationEngine::new();
        let plan = engine
            .plan_for_kind(&repo, "p", MemoryKind::Episodic, true, 0.6)
            .unwrap();

        assert_eq!(plan.groups.len(), 1, "三条近义记录应只归为一组");
        let g = &plan.groups[0];
        assert_eq!(g.duplicate_ids.len(), 2, "B、C 都应并入 A 这一组");
        let keeper = repo.get_episodic(&g.keeper_id, "p").unwrap().unwrap();
        assert_eq!(keeper.created_at, 100);
    }

    #[test]
    fn test_consolidate_multiple_kinds_are_isolated() {
        // 跨类型 consolidate：同时处理 episodic 与 decision，
        // 断言返回 plan 数 == kinds 数、各 plan 的 memory_type 正确、互不串数据。
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        // episodic 一组精确重复。
        repo.create_episodic(&mk_episodic("same", "body", 100, "p"))
            .unwrap();
        repo.create_episodic(&mk_episodic("same", "body", 200, "p"))
            .unwrap();
        // decision 一组精确重复（dedup_text = title||ctx||rationale||tradeoffs）。
        let mk_dec = |ts: i64| DecisionMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            title: "t".into(),
            context: "c".into(),
            rationale: "r".into(),
            tradeoffs: "to".into(),
            related_files: vec![],
            tags: vec![],
            importance: 0.5,
            superseded_by: None,
            created_at: ts,
            updated_at: ts,
        };
        repo.create_decision(&mk_dec(100)).unwrap();
        repo.create_decision(&mk_dec(200)).unwrap();

        let engine = ConsolidationEngine::new();
        let plans = engine
            .consolidate(
                &repo,
                "p",
                &[MemoryKind::Episodic, MemoryKind::Decision],
                false,
                0.85,
                false,
                999,
            )
            .unwrap();

        assert_eq!(plans.len(), 2, "kinds 数 == plan 数");
        assert_eq!(plans[0].memory_type, "episodic");
        assert_eq!(plans[1].memory_type, "decision");
        // 各自恰好一组、互不影响。
        assert_eq!(plans[0].groups.len(), 1);
        assert_eq!(plans[1].groups.len(), 1);
    }

    #[test]
    fn test_consolidate_apply_archived_count_across_groups() {
        // apply=true 下多组重复归档：archived 计数应 == 实际归档行数（两组各 1 条 dup）。
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo.create_episodic(&mk_episodic("a", "a", 100, "p"))
            .unwrap();
        repo.create_episodic(&mk_episodic("a", "a", 110, "p"))
            .unwrap();
        repo.create_episodic(&mk_episodic("b", "b", 200, "p"))
            .unwrap();
        repo.create_episodic(&mk_episodic("b", "b", 210, "p"))
            .unwrap();

        let engine = ConsolidationEngine::new();
        let plans = engine
            .consolidate(&repo, "p", &[MemoryKind::Episodic], false, 0.85, true, 999)
            .unwrap();

        let plan = &plans[0];
        assert_eq!(plan.groups.len(), 2, "两组独立精确重复");
        assert_eq!(plan.archived, 2, "archived 应等于实际归档行数");
        assert_eq!(
            repo.list_archived(MemoryKind::Episodic, "p", 10)
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn test_plan_isolates_by_project_id() {
        // 跨项目隔离：两个 project 各有相同的精确重复，plan 只归集本项目内记录。
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo.create_episodic(&mk_episodic("same", "body", 100, "p1"))
            .unwrap();
        repo.create_episodic(&mk_episodic("same", "body", 200, "p1"))
            .unwrap();
        repo.create_episodic(&mk_episodic("same", "body", 100, "p2"))
            .unwrap();
        repo.create_episodic(&mk_episodic("same", "body", 200, "p2"))
            .unwrap();

        let engine = ConsolidationEngine::new();
        let plan_p1 = engine
            .plan_for_kind(&repo, "p1", MemoryKind::Episodic, false, 0.85)
            .unwrap();
        let plan_p2 = engine
            .plan_for_kind(&repo, "p2", MemoryKind::Episodic, false, 0.85)
            .unwrap();

        for plan in [&plan_p1, &plan_p2] {
            assert_eq!(plan.groups.len(), 1, "每个项目各自一组");
            assert_eq!(plan.groups[0].duplicate_ids.len(), 1);
        }
        // 两个项目的 keeper 不应是同一条。
        assert_ne!(plan_p1.groups[0].keeper_id, plan_p2.groups[0].keeper_id);
    }
}
