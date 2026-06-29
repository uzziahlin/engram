use crate::storage::{MemoryKind, MemoryRepository};
use anyhow::Result;
use rusqlite::params;
use std::collections::{HashMap, HashSet};

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
        // (id, created_at, canonical_text) for active memories, oldest first.
        let sql = format!(
            "SELECT id, created_at, {} AS dedup_text FROM {} \
             WHERE project_id = ?1 AND archived_at IS NULL ORDER BY created_at ASC",
            kind.dedup_text_expr(),
            kind.table()
        );
        let conn = repo.connection()?;
        let mut stmt = conn.prepare(&sql)?;
        let recs: Vec<(String, String)> = stmt
            .query_map(params![project_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(2)?))
            })?
            .collect::<std::result::Result<_, _>>()?;
        drop(stmt);

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

        // Optional near-duplicates among the still-ungrouped, by Jaccard.
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
                    if Self::jaccard_similarity(&recs[a].1, &recs[b].1, threshold) {
                        dups.push(recs[b].0.clone());
                        grouped.insert(b);
                    }
                }
                if !dups.is_empty() {
                    grouped.insert(a);
                    groups.push(ConsolidationGroup {
                        keeper_id: recs[a].0.clone(),
                        duplicate_ids: dups,
                        reason: format!("near(jaccard>={threshold})"),
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
                        if repo.archive(kind, dup_id, project_id, now)? {
                            archived += 1;
                        }
                    }
                }
                plan.archived = archived;
            }
            plans.push(plan);
        }
        Ok(plans)
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
        let keeper = repo.get_episodic(&group.keeper_id).unwrap().unwrap();
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
        assert!(ConsolidationEngine::jaccard_similarity(
            "same text",
            "same text",
            0.99
        ));
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
    fn test_jaccard_threshold_boundary_ge_semantics() {
        // 两个文本：交集 2 词、并集 4 词 → Jaccard 恰好 0.5。
        // 验证 `>=` 闭区间语义：threshold=0.5 判相似，threshold=0.6 判不相似。
        let a = "alpha beta gamma delta";
        let b = "alpha beta";
        assert!(
            ConsolidationEngine::jaccard_similarity(a, b, 0.5),
            "Jaccard 恰好等于 threshold 应判为 near-dup（闭区间）"
        );
        assert!(
            !ConsolidationEngine::jaccard_similarity(a, b, 0.6),
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
        let keeper = repo.get_episodic(&g.keeper_id).unwrap().unwrap();
        assert_eq!(keeper.created_at, 100, "near-dup keeper 仍应为最早创建者");
        assert_eq!(plan.archived, 0, "plan_for_kind 不应归档");
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
        let keeper = repo.get_episodic(&g.keeper_id).unwrap().unwrap();
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
