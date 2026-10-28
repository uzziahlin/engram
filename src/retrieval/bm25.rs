use crate::models::{DecisionMemory, EpisodicMemory, FailureMemory, ProceduralMemory};
use crate::retrieval::planner::MemorySource;
use crate::storage::{ActiveMemory, MemoryKind, MemoryRepository, ScoredMemory};
use anyhow::Result;
use std::collections::HashSet;

/// A single search result with relevance score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: String,
    /// Owning project. Global search (`project_id = "*"`) mixes hits from
    /// many projects; callers need the exact project for get_memory /
    /// mark_relevance / forget_memory, which all require it.
    pub project_id: String,
    pub memory_type: String,
    pub summary: String,
    pub relevance_score: f32,
    pub importance: f32,
    pub created_at: i64,
    /// The memory's tags — lets callers (e.g. the MCP search filter) match
    /// without a second DB round-trip.
    pub tags: Vec<String>,
    /// Type-specific full fields (root_cause/fix/prevention, context/tradeoffs,
    /// steps, content…). Search is the *only* read path agents have, so results
    /// must carry the full payload, not just a teaser summary.
    pub detail: serde_json::Value,
}

/// Trait for extracting searchable fields from memory types.
trait Searchable {
    fn search_id(&self) -> &str;
    fn search_project_id(&self) -> &str;
    fn search_summary(&self) -> &str;
    fn search_created_at(&self) -> i64;
    fn search_importance(&self) -> f32;
    fn search_tags(&self) -> Vec<String>;
    fn search_detail(&self) -> serde_json::Value;
}

impl Searchable for EpisodicMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_project_id(&self) -> &str {
        &self.project_id
    }
    fn search_summary(&self) -> &str {
        &self.summary
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        self.importance.clamp(0.0, 1.0)
    }
    fn search_tags(&self) -> Vec<String> {
        self.tags.clone()
    }
    fn search_detail(&self) -> serde_json::Value {
        serde_json::json!({
            "content": self.content,
            "files_touched": self.files_touched,
            "related_commits": self.related_commits,
        })
    }
}

impl Searchable for DecisionMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_project_id(&self) -> &str {
        &self.project_id
    }
    fn search_summary(&self) -> &str {
        &self.title
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        self.importance.clamp(0.0, 1.0)
    }
    fn search_tags(&self) -> Vec<String> {
        self.tags.clone()
    }
    fn search_detail(&self) -> serde_json::Value {
        let mut detail = serde_json::json!({
            "context": self.context,
            "rationale": self.rationale,
            "tradeoffs": self.tradeoffs,
            "related_files": self.related_files,
        });
        if let Some(sb) = &self.superseded_by {
            detail["superseded_by"] = serde_json::json!(sb);
        }
        detail
    }
}

impl Searchable for FailureMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_project_id(&self) -> &str {
        &self.project_id
    }
    fn search_summary(&self) -> &str {
        &self.incident
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        (self.severity as f32 / 5.0).clamp(0.0, 1.0)
    }
    fn search_tags(&self) -> Vec<String> {
        self.tags.clone()
    }
    fn search_detail(&self) -> serde_json::Value {
        serde_json::json!({
            "root_cause": self.root_cause,
            "fix": self.fix,
            "prevention": self.prevention,
            "severity": self.severity,
        })
    }
}

impl Searchable for ProceduralMemory {
    fn search_id(&self) -> &str {
        &self.id
    }
    fn search_project_id(&self) -> &str {
        &self.project_id
    }
    fn search_summary(&self) -> &str {
        &self.workflow_name
    }
    fn search_created_at(&self) -> i64 {
        self.created_at
    }
    fn search_importance(&self) -> f32 {
        self.importance.clamp(0.0, 1.0)
    }
    fn search_tags(&self) -> Vec<String> {
        self.tags.clone()
    }
    fn search_detail(&self) -> serde_json::Value {
        serde_json::json!({
            "steps": self.steps,
            "related_tools": self.related_tools,
        })
    }
}

/// Type-erased search fields with the RAW bm25 score, before any
/// normalization — the calibration pass needs the raw scale. `tokens` is the
/// doc's searchable token set (same preprocessing as the FTS index), used by
/// the query-term coverage signal.
struct ScoredRaw {
    raw: f64,
    tokens: HashSet<String>,
    id: String,
    project_id: String,
    summary: String,
    importance: f32,
    created_at: i64,
    tags: Vec<String>,
    detail: serde_json::Value,
}

/// Collect every string leaf of a JSON detail value into one text blob
/// (values only, not keys) — the same text the FTS index covers.
fn collect_text_leaves(v: &serde_json::Value, out: &mut String) {
    match v {
        serde_json::Value::String(s) => {
            out.push(' ');
            out.push_str(s);
        }
        serde_json::Value::Array(a) => {
            for x in a {
                collect_text_leaves(x, out);
            }
        }
        serde_json::Value::Object(o) => {
            for x in o.values() {
                collect_text_leaves(x, out);
            }
        }
        _ => {}
    }
}

fn doc_tokens(summary: &str, detail: &serde_json::Value) -> HashSet<String> {
    let mut text = String::with_capacity(summary.len() + 128);
    text.push_str(summary);
    collect_text_leaves(detail, &mut text);
    MemoryRepository::preprocess_cjk(&text)
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

fn raw_group<T: Searchable>(scored: Vec<ScoredMemory<T>>) -> Vec<ScoredRaw> {
    scored
        .into_iter()
        .map(|s| {
            let m = s.memory;
            let summary = m.search_summary().to_string();
            let detail = m.search_detail();
            let tokens = doc_tokens(&summary, &detail);
            ScoredRaw {
                raw: (-s.bm25_score).max(0.0),
                tokens,
                id: m.search_id().to_string(),
                project_id: m.search_project_id().to_string(),
                summary,
                importance: m.search_importance(),
                created_at: m.search_created_at(),
                tags: m.search_tags(),
                detail,
            }
        })
        .collect()
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

    /// Ids that count as search impressions for the feedback loop: every
    /// result EXCEPT graph-expansion neighbors. Neighbors never matched the
    /// query — counting them as impressions made `engram tune` systematically
    /// demote them (served ≥5, adopted 0, by construction).
    pub fn impression_ids(results: &[SearchResult]) -> Vec<String> {
        results
            .iter()
            .filter(|r| {
                !r.detail
                    .get("graph_neighbor")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            })
            .map(|r| r.id.clone())
            .collect()
    }

    /// Floor of the query-term coverage factor. A doc matching none of the
    /// query tokens exactly (only via porter-stemmed variants the Rust-side
    /// comparison cannot see) keeps this share of its score instead of being
    /// zeroed out.
    const COVERAGE_FLOOR: f32 = 0.35;

    /// Whether a query token is present in a doc's token set. Exact match
    /// first; longer tokens also match on prefix overlap in either direction,
    /// approximating FTS5's porter stemming ("crash" ~ "crashed"). Short
    /// tokens (<4 chars) skip the prefix heuristic — "cat"/"catalog" style
    /// false positives cost more than the stems they'd catch.
    fn token_matches(q: &str, doc_tokens: &HashSet<String>) -> bool {
        if doc_tokens.contains(q) {
            return true;
        }
        if q.chars().count() < 4 {
            return false;
        }
        doc_tokens
            .iter()
            .any(|d| d.starts_with(q) || (d.chars().count() >= 4 && q.starts_with(d.as_str())))
    }

    /// Fraction of the query's tokens the doc actually contains. Single-token
    /// queries are always 1.0 (nothing to discriminate).
    fn coverage(raw: &ScoredRaw, query_tokens: &[String]) -> f32 {
        if query_tokens.len() <= 1 {
            return 1.0;
        }
        let matched = query_tokens
            .iter()
            .filter(|q| Self::token_matches(q, &raw.tokens))
            .count();
        matched as f32 / query_tokens.len() as f32
    }

    /// Convert an [`ActiveMemory`] into a SearchResult with `relevance_score
    /// = 0.0` — these are not BM25 hits; the reranker recomputes the final
    /// score. Importance is normalized the same way as the BM25 path (via
    /// `Searchable`).
    fn active_to_result(m: ActiveMemory) -> SearchResult {
        let generic = |memory_type: &str,
                       id: &str,
                       project_id: &str,
                       summary: &str,
                       importance: f32,
                       created_at: i64,
                       tags: Vec<String>,
                       detail: serde_json::Value| SearchResult {
            id: id.to_string(),
            project_id: project_id.to_string(),
            memory_type: memory_type.into(),
            summary: summary.to_string(),
            relevance_score: 0.0,
            importance,
            created_at,
            tags,
            detail,
        };
        match m {
            ActiveMemory::Episodic(m) => generic(
                "episodic",
                &m.id,
                &m.project_id,
                &m.summary,
                m.importance.clamp(0.0, 1.0),
                m.created_at,
                m.tags.clone(),
                serde_json::json!({
                    "content": m.content,
                    "files_touched": m.files_touched,
                    "related_commits": m.related_commits,
                }),
            ),
            ActiveMemory::Decision(m) => {
                let mut detail = serde_json::json!({
                    "context": m.context,
                    "rationale": m.rationale,
                    "tradeoffs": m.tradeoffs,
                    "related_files": m.related_files,
                });
                if let Some(sb) = &m.superseded_by {
                    detail["superseded_by"] = serde_json::json!(sb);
                }
                generic(
                    "decision",
                    &m.id,
                    &m.project_id,
                    &m.title,
                    m.importance.clamp(0.0, 1.0),
                    m.created_at,
                    m.tags.clone(),
                    detail,
                )
            }
            ActiveMemory::Failure(m) => {
                let severity = m.severity;
                generic(
                    "failure",
                    &m.id,
                    &m.project_id,
                    &m.incident,
                    (severity as f32 / 5.0).clamp(0.0, 1.0),
                    m.created_at,
                    m.tags.clone(),
                    serde_json::json!({
                        "root_cause": m.root_cause,
                        "fix": m.fix,
                        "prevention": m.prevention,
                        "severity": severity,
                    }),
                )
            }
            ActiveMemory::Procedural(m) => generic(
                "procedural",
                &m.id,
                &m.project_id,
                &m.workflow_name,
                m.importance.clamp(0.0, 1.0),
                m.created_at,
                m.tags.clone(),
                serde_json::json!({
                    "steps": m.steps,
                    "related_tools": m.related_tools,
                }),
            ),
        }
    }

    /// Search a subset of memory types through the ONE unified FTS table.
    ///
    /// The unified table makes raw bm25() scores comparable across kinds
    /// (one IDF scale over the whole corpus) — the whole cross-type
    /// calibration layer of the per-type-table era is gone.
    ///
    /// relevance is a per-query MONOTONE normalization of the raw score:
    /// `raw / query_best`, where query_best is the highest raw score of THIS
    /// query's result set (all-zero → all 0). Ordering by relevance is
    /// therefore always identical to ordering by raw score, values are
    /// bounded [0,1], and no fixed-scale constant (the old SIGMOID_SCALE was
    /// tuned against a 22-row corpus and saturated as corpora grew) is
    /// involved. Absolute scores are NOT comparable across queries — that is
    /// inherent to bm25 (a rare-term query and a common-term query land on
    /// different scales); ranking and within-result relative scores are what
    /// downstream consumers (reranker, agent) actually use.
    ///
    /// Query-term coverage is kept: on OR-fallback (and any multi-token)
    /// query, a doc's score scales with how many of the query's tokens it
    /// actually contains — bm25 raw score is IDF-dominated, so a short doc
    /// matching ONE rare token could outrank a doc matching all four query
    /// tokens. AND-path docs match every token by construction, so their
    /// factor is ~1.0 and ranking there is unchanged.
    pub fn search_by_types(
        repo: &MemoryRepository,
        query: &str,
        project_id: &str,
        sources: &[MemorySource],
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        if sources.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let to_kind = |src: &MemorySource| match src {
            MemorySource::Episodic => MemoryKind::Episodic,
            MemorySource::Decision => MemoryKind::Decision,
            MemorySource::Failure => MemoryKind::Failure,
            MemorySource::Procedural => MemoryKind::Procedural,
        };
        // Kind routing: a single source filters inside the index
        // (`kind:"…"` MATCH column); a strict subset of 2-3 sources cannot
        // (the filter column takes one value), so it runs unfiltered and
        // drops foreign kinds after decode, over-fetching ×4 to compensate.
        let all = sources.len() == MemorySource::all().len();
        let kind_filter = match sources {
            [only] => Some(to_kind(only)),
            _ => None,
        };
        let subset_postfilter = !all && kind_filter.is_none();

        let hits = repo.search_unified(
            query,
            project_id,
            kind_filter,
            if subset_postfilter {
                (limit * 4).min(400)
            } else {
                limit
            },
        )?;

        let mut groups: Vec<(Vec<ScoredRaw>, MemoryKind)> = Vec::new();
        if !hits.episodic.is_empty() {
            groups.push((raw_group(hits.episodic), MemoryKind::Episodic));
        }
        if !hits.decisions.is_empty() {
            groups.push((raw_group(hits.decisions), MemoryKind::Decision));
        }
        if !hits.failures.is_empty() {
            groups.push((raw_group(hits.failures), MemoryKind::Failure));
        }
        if !hits.procedures.is_empty() {
            groups.push((raw_group(hits.procedures), MemoryKind::Procedural));
        }
        if subset_postfilter {
            let allowed: Vec<MemoryKind> = sources.iter().map(to_kind).collect();
            groups.retain(|(_, kind)| allowed.contains(kind));
        }

        // Stopwords drop out of the coverage denominator too (see
        // is_stopword): content-word coverage is the signal.
        let query_tokens: Vec<String> = MemoryRepository::preprocess_cjk(query)
            .split_whitespace()
            .filter(|t| !MemoryRepository::is_stopword(t))
            .map(str::to_string)
            .collect();

        // Per-query monotone normalization (see method doc).
        let query_best = groups
            .iter()
            .flat_map(|(raws, _)| raws.iter().map(|r| r.raw))
            .fold(0.0f64, f64::max);

        let mut results = Vec::new();
        for (raws, kind) in groups {
            for raw in raws {
                let relevance = if query_best > 0.0 {
                    let normalized = (raw.raw / query_best) as f32;
                    let cov = Self::coverage(&raw, &query_tokens);
                    normalized * (Self::COVERAGE_FLOOR + (1.0 - Self::COVERAGE_FLOOR) * cov)
                } else {
                    0.0
                };
                results.push(SearchResult {
                    id: raw.id,
                    project_id: raw.project_id,
                    memory_type: kind.as_str().to_string(),
                    summary: raw.summary,
                    relevance_score: relevance,
                    importance: raw.importance,
                    created_at: raw.created_at,
                    tags: raw.tags,
                    detail: raw.detail,
                });
            }
        }

        // Sort by relevance score descending
        results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results)
    }

    /// Search all memory types via FTS5 BM25 for a given query.
    /// Returns combined results sorted by relevance.
    pub fn search_all(
        repo: &MemoryRepository,
        query: &str,
        project_id: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        Self::search_by_types(repo, query, project_id, MemorySource::all(), limit)
    }

    /// Search a specific memory type. Delegates to [`Self::search_by_types`]
    /// with a single source so filtered and unfiltered searches share ONE
    /// query path.
    pub fn search_by_type(
        repo: &MemoryRepository,
        query: &str,
        project_id: &str,
        memory_type: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // A typo'd type must fail loudly, not silently return "no
        // memories" — the caller then knows to fix the filter.
        let source = MemorySource::from_type_str(memory_type)?;
        Self::search_by_types(
            repo,
            query,
            project_id,
            std::slice::from_ref(&source),
            limit,
        )
    }

    /// Materialize `SearchResult`s for explicit `(memory_type, id)` pairs by
    /// loading them from the repo — ONE batched query per kind (this used to
    /// be one `get_active_*` round-trip per id). Used to bring vector-only
    /// hits and graph neighbors into the fused candidate set.
    /// `relevance_score` is 0.0 (these were not BM25 hits); the reranker
    /// recomputes the final score.
    ///
    /// Uses wildcard-capable, archived-excluding getters: `project_id = '*'`
    /// (global search) matches any project, and a forgotten memory can never
    /// resurface through a stale embedding or graph edge.
    pub fn fetch_by_ids(
        repo: &MemoryRepository,
        ids: &[(String, String)],
        project_id: &str,
    ) -> Result<Vec<SearchResult>> {
        Ok(repo
            .get_active_any_batch(ids, project_id)?
            .into_iter()
            .map(Self::active_to_result)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{DecisionMemory, EpisodicMemory, FailureMemory, ProceduralMemory};

    fn episodic(importance: f32) -> EpisodicMemory {
        EpisodicMemory {
            id: "e1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "sum".into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance,
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        }
    }
    fn failure(severity: u8) -> FailureMemory {
        FailureMemory {
            id: "f1".into(),
            project_id: "p".into(),
            incident: "boom".into(),
            root_cause: "rc".into(),
            fix: "fx".into(),
            prevention: "pv".into(),
            severity,
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        }
    }
    fn decision() -> DecisionMemory {
        DecisionMemory {
            id: "d1".into(),
            project_id: "p".into(),
            title: "t".into(),
            context: "ctx".into(),
            rationale: "r".into(),
            tradeoffs: "to".into(),
            related_files: vec![],
            tags: vec![],
            importance: 0.5,
            superseded_by: None,
            created_at: 0,
            updated_at: 0,
        }
    }
    fn procedural() -> ProceduralMemory {
        ProceduralMemory {
            id: "pr1".into(),
            project_id: "p".into(),
            workflow_name: "wf".into(),
            steps: vec![],
            related_tools: vec![],
            tags: vec![],
            importance: 0.5,
            created_at: 0,
            updated_at: 0,
        }
    }

    #[test]
    fn importance_maps_per_type() {
        // episodic: passthrough
        let r = BM25Retriever::active_to_result(ActiveMemory::Episodic(episodic(0.8)));
        assert!((r.importance - 0.8).abs() < 1e-6);
        assert_eq!(
            r.relevance_score, 0.0,
            "materialized ids carry no BM25 score"
        );
        // failure: severity / 5
        let r = BM25Retriever::active_to_result(ActiveMemory::Failure(failure(5)));
        assert!((r.importance - 1.0).abs() < 1e-6);
        let r = BM25Retriever::active_to_result(ActiveMemory::Failure(failure(1)));
        assert!((r.importance - 0.2).abs() < 1e-6);
        // decision / procedural: neutral 0.5
        let r = BM25Retriever::active_to_result(ActiveMemory::Decision(decision()));
        assert!((r.importance - 0.5).abs() < 1e-6);
        let r = BM25Retriever::active_to_result(ActiveMemory::Procedural(procedural()));
        assert!((r.importance - 0.5).abs() < 1e-6);
    }

    #[test]
    fn search_by_types_returns_only_requested_types() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();

        let ep = EpisodicMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "alpha episodic".into(),
            content: "alpha".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        };
        let fail = FailureMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            incident: "alpha failure".into(),
            root_cause: "rc".into(),
            fix: "fx".into(),
            prevention: "pv".into(),
            severity: 3,
            tags: vec![],
            created_at: 2,
            updated_at: 2,
        };
        let dec = DecisionMemory {
            id: uuid::Uuid::new_v4().to_string(),
            project_id: "p".into(),
            title: "alpha decision".into(),
            context: "ctx".into(),
            rationale: "r".into(),
            tradeoffs: "to".into(),
            related_files: vec![],
            tags: vec![],
            importance: 0.5,
            superseded_by: None,
            created_at: 3,
            updated_at: 3,
        };
        repo.create_episodic(&ep).unwrap();
        repo.create_failure(&fail).unwrap();
        repo.create_decision(&dec).unwrap();

        // Request only Failure + Episodic → Decision must be filtered out.
        let r = BM25Retriever::search_by_types(
            &repo,
            "alpha",
            "p",
            &[MemorySource::Failure, MemorySource::Episodic],
            10,
        )
        .unwrap();
        let types: std::collections::HashSet<&str> =
            r.iter().map(|x| x.memory_type.as_str()).collect();
        assert!(types.contains("failure"));
        assert!(types.contains("episodic"));
        assert!(
            !types.contains("decision"),
            "decision must be filtered out by source routing"
        );

        // search_all (all four) surfaces decision too.
        let r_all = BM25Retriever::search_all(&repo, "alpha", "p", 10).unwrap();
        let types_all: std::collections::HashSet<&str> =
            r_all.iter().map(|x| x.memory_type.as_str()).collect();
        assert!(types_all.contains("decision"));
    }

    #[test]
    fn bm25_search_ranks_relevant_higher() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk = |id: &str, summary: &str| EpisodicMemory {
            id: id.into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: summary.into(),
            content: summary.into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mk("e1", "learning rust memory system"))
            .unwrap();
        repo.create_episodic(&mk("e2", "python web server setup"))
            .unwrap();
        let results = repo.search_episodic("rust", "p", 10).unwrap();
        assert!(!results.is_empty());
        assert_eq!(
            results[0].memory.id, "e1",
            "relevant memory should rank first"
        );
    }

    #[test]
    fn bm25_search_cjk_finds_chinese() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mem = EpisodicMemory {
            id: "cjk1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "修复认证模块".into(),
            content: "认证 token 刷新逻辑".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mem).unwrap();
        let results = repo.search_episodic("认证", "p", 10).unwrap();
        assert!(!results.is_empty(), "CJK search should find the memory");
    }

    #[test]
    fn coverage_demotes_rare_single_term_hit() {
        // The known OR-fallback weakness: a short doc containing ONE rare
        // query token (high IDF → high raw bm25) must not outrank a doc that
        // contains ALL the query tokens.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk = |id: &str, summary: &str, content: &str| EpisodicMemory {
            id: id.into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: summary.into(),
            content: content.into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        };
        // "xylophone" is absent everywhere else → very rare token.
        repo.create_episodic(&mk("rare", "xylophone", "unrelated filler text"))
            .unwrap();
        repo.create_episodic(&mk(
            "full",
            "crash report analysis",
            "crash analysis with stack trace",
        ))
        .unwrap();
        let r = BM25Retriever::search_all(&repo, "crash analysis xylophone", "p", 10).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(
            r[0].id,
            "full",
            "doc matching 2/3 query terms must outrank the single-rare-term doc (got {:?})",
            r.iter()
                .map(|x| (x.id.clone(), x.relevance_score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn fetch_by_ids_supports_wildcard_and_excludes_archived() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mem = EpisodicMemory {
            id: "wx1".into(),
            project_id: "other-project".into(),
            session_id: "s".into(),
            summary: "global knowledge".into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mem).unwrap();
        // Wildcard: a cross-project id materializes under '*'.
        let r = BM25Retriever::fetch_by_ids(
            &repo,
            &[(("episodic".to_string()), "wx1".to_string())],
            "*",
        )
        .unwrap();
        assert_eq!(
            r.len(),
            1,
            "wildcard project must materialize cross-project ids"
        );
        assert_eq!(r[0].id, "wx1");
        // Strict project still returns nothing for another project's id.
        let r = BM25Retriever::fetch_by_ids(
            &repo,
            &[("episodic".to_string(), "wx1".to_string())],
            "mine",
        )
        .unwrap();
        assert!(r.is_empty());
        // Archived rows never materialize (stale embeddings/graph edges).
        repo.archive(
            crate::storage::repository::MemoryKind::Episodic,
            "wx1",
            "other-project",
            99,
        )
        .unwrap();
        let r =
            BM25Retriever::fetch_by_ids(&repo, &[("episodic".to_string(), "wx1".to_string())], "*")
                .unwrap();
        assert!(r.is_empty(), "archived memory must not resurface");
    }

    #[test]
    fn search_by_type_shares_calibration_with_search_all() {
        // Same doc, same query: the score must not depend on whether a
        // memory_type filter was present (the old separate normalization
        // gave filtered searches a different scale).
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let mk = |id: &str| EpisodicMemory {
            id: id.into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "alpha beta gamma".into(),
            content: "alpha beta gamma".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1,
            updated_at: 1,
        };
        repo.create_episodic(&mk("e1")).unwrap();
        let filtered =
            BM25Retriever::search_by_type(&repo, "alpha beta", "p", "episodic", 10).unwrap();
        let all = BM25Retriever::search_all(&repo, "alpha beta", "p", 10).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(all.len(), 1);
        assert!(
            (filtered[0].relevance_score - all[0].relevance_score).abs() < 1e-6,
            "filtered {} vs unfiltered {}",
            filtered[0].relevance_score,
            all[0].relevance_score
        );
    }
}
