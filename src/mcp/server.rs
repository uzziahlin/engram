use crate::collectors;
use crate::config::Config;
use crate::consolidation::ConsolidationEngine;
use crate::context::composer::{ContextBudget, ContextComposer};
use crate::git_integration::GitIntegration;
use crate::mcp::embedding_service::EmbeddingService;
use crate::models::*;
use crate::retrieval::bm25::BM25Retriever;
use crate::retrieval::intent_classifier::IntentClassifier;
use crate::retrieval::planner::RetrievalPlanner;
use crate::retrieval::reranker::Reranker;
use crate::storage::MemoryKind;
use crate::storage::MemoryRepository;
use crate::storage::ScoredMemory;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::mcp::transport::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, RequestHandler};

/// Jaccard similarity threshold for near-duplicate consolidation.
const CONSOLIDATE_JACCARD_THRESHOLD: f64 = crate::consolidation::engine::DEFAULT_JACCARD_THRESHOLD;

/// Dispatch a tool call: deserialize arguments and invoke the provider method.
/// Returns `(Some(result), None)` on successful parse, or `(None, Some(error))` on parse failure.
macro_rules! dispatch_tool {
    ($args:expr, $input_ty:ty, $provider:expr, $method:ident) => {
        match serde_json::from_value::<$input_ty>($args) {
            Ok(i) => (Some($provider.$method(i)), None),
            Err(e) => (
                None,
                Some(JsonRpcError {
                    code: -32602,
                    message: format!("Invalid params: {e}"),
                    data: None,
                }),
            ),
        }
    };
}

/// Clamp a caller-supplied `limit` into 1..=1000. Unclamped values flowed
/// straight into SQL LIMIT and result materialization (limit=1_000_000 kept
/// every row in memory; `limit * 3` overflowed in debug builds).
fn clamp_limit(v: Option<usize>, default: usize) -> usize {
    v.unwrap_or(default).clamp(1, 1000)
}

/// Clamp a `days` window into 0..=3650 (10 years). `now - days*86400`
/// previously overflowed i64 on extreme values (saturating_mul saturates, but
/// the subtraction then wrapped).
fn clamp_days(days: i64) -> i64 {
    days.clamp(0, 3650)
}

/// mark_relevance tool input: agent feedback on a search result's usefulness.
#[derive(Debug, Deserialize)]
pub struct MarkRelevanceInput {
    pub project_id: String,
    pub memory_type: String,
    pub id: String,
    /// "useful" bumps importance (+0.1), "irrelevant" drops it (-0.1).
    pub feedback: String,
}

/// search_memory tool input.
#[derive(Debug, Deserialize)]
pub struct SearchMemoryInput {
    project_id: String,
    query: String,
    #[serde(default)]
    memory_type: Option<String>,
    /// None → `[retrieval].default_limit` from config.
    #[serde(default)]
    limit: Option<usize>,
    /// If non-empty, only memories carrying at least one of these tags.
    #[serde(default)]
    tags: Vec<String>,
    /// If set, only memories created before this unix timestamp.
    #[serde(default)]
    before: Option<i64>,
}

/// get_memory tool input — fetch one memory's full record by id.
#[derive(Debug, Deserialize)]
pub struct GetMemoryInput {
    project_id: String,
    memory_type: String,
    id: String,
}

/// related_files tool input.
#[derive(Debug, Deserialize)]
pub struct RelatedFilesInput {
    project_id: String,
    file: String,
}

/// timeline tool input.
#[derive(Debug, Deserialize)]
pub struct TimelineInput {
    project_id: String,
    #[serde(default = "default_days")]
    days: i64,
    /// Optional filter: count only this memory type. The other per-type
    /// counts stay 0 so the response shape never changes with filtering.
    #[serde(default)]
    memory_type: Option<String>,
    /// Optional window end as an absolute unix timestamp (exclusive).
    #[serde(default)]
    until: Option<i64>,
}

fn default_days() -> i64 {
    7
}

/// recent_failures tool input.
#[derive(Debug, Deserialize)]
pub struct RecentFailuresInput {
    project_id: String,
    #[serde(default)]
    service: Option<String>,
    /// None → `[retrieval].default_limit` from config.
    #[serde(default)]
    limit: Option<usize>,
}

/// architectural_decisions tool input.
#[derive(Debug, Deserialize)]
pub struct ArchitecturalDecisionsInput {
    pub project_id: String,
    #[serde(default)]
    topic: Option<String>,
    /// None → `[retrieval].default_limit` from config.
    #[serde(default)]
    limit: Option<usize>,
    /// Superseded decisions are history: excluded unless explicitly requested.
    #[serde(default)]
    include_superseded: bool,
}

/// query_stats tool input — retrieval feedback aggregated by query string.
#[derive(Debug, Deserialize)]
pub struct QueryStatsInput {
    project_id: String,
    #[serde(default = "default_days")]
    days: i64,
    /// None → `[retrieval].default_limit` from config.
    #[serde(default)]
    limit: Option<usize>,
}

/// session_brief tool input — one-call context assembly for session start.
#[derive(Debug, Deserialize)]
pub struct SessionBriefInput {
    pub project_id: String,
    /// Optional focus topic: also runs a search and includes the top hits.
    #[serde(default)]
    pub focus: Option<String>,
}

/// reflect tool input — scan a project's active failures for recurring tags and
/// propose preventive rules.
#[derive(Debug, Deserialize)]
pub struct ReflectInput {
    project_id: String,
    /// Dry-run by default; `true` persists proposals as pending suggestions.
    #[serde(default)]
    apply: bool,
    /// Override `[reflection].min_occurrences`; `None` uses the configured default.
    #[serde(default)]
    min_occurrences: Option<usize>,
}

/// list_suggestions tool input.
#[derive(Debug, Deserialize)]
pub struct ListSuggestionsInput {
    project_id: String,
}

/// confirm_suggestion / reject_suggestion tool input (shared shape).
#[derive(Debug, Deserialize)]
pub struct SuggestionIdInput {
    project_id: String,
    id: String,
}

/// create_episodic tool input.
#[derive(Debug, Deserialize)]
pub struct CreateEpisodicInput {
    pub project_id: String,
    pub session_id: String,
    pub summary: String,
    pub content: String,
    #[serde(default)]
    pub files_touched: Vec<String>,
    #[serde(default)]
    pub related_commits: Vec<String>,
    #[serde(default = "default_importance")]
    pub importance: f32,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_importance() -> f32 {
    0.5
}

/// create_decision tool input.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateDecisionInput {
    pub project_id: String,
    pub title: String,
    pub context: String,
    pub rationale: String,
    pub tradeoffs: String,
    #[serde(default)]
    pub related_files: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Ranking signal (0..=1, default 0.5).
    #[serde(default)]
    pub importance: Option<f32>,
    /// Id of the decision this one replaces (supersede chain): the old
    /// decision stays searchable as history but is flagged and downranked.
    #[serde(default)]
    pub supersedes: Option<String>,
}

/// create_failure tool input.
#[derive(Debug, Deserialize)]
pub struct CreateFailureInput {
    pub project_id: String,
    pub incident: String,
    pub root_cause: String,
    pub fix: String,
    pub prevention: String,
    #[serde(default = "default_severity")]
    pub severity: u8,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_severity() -> u8 {
    3
}

/// create_procedural tool input.
#[derive(Debug, Deserialize)]
pub struct CreateProceduralInput {
    pub project_id: String,
    pub workflow_name: String,
    pub steps: Vec<String>,
    #[serde(default)]
    pub related_tools: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Ranking signal (0..=1, default 0.5).
    #[serde(default)]
    pub importance: Option<f32>,
}

/// ingest_commits tool input — auto-generate memories from git history.
#[derive(Debug, Deserialize)]
pub struct IngestCommitsInput {
    pub project_id: String,
    pub repo_path: String,
    #[serde(default = "default_ingest_count")]
    pub count: usize,
    #[serde(default)]
    pub session_id: Option<String>,
}

fn default_ingest_count() -> usize {
    20
}

/// collect_sources tool input — gather bootstrap evidence from an existing project.
#[derive(Debug, Deserialize)]
pub struct CollectSourcesInput {
    pub project_id: String,
    pub repo_path: String,
    /// Comma-separated dimension list ("git,decisions"). None/empty → all.
    #[serde(default)]
    pub dimensions: Option<String>,
    /// How many recent commits to walk for the git dimension.
    #[serde(default = "default_collect_commits")]
    pub max_commits: usize,
}

fn default_collect_commits() -> usize {
    200
}

/// forget_memory tool input — soft-delete (archive) one memory.
#[derive(Debug, Deserialize)]
pub struct ForgetMemoryInput {
    pub project_id: String,
    pub memory_type: String,
    pub id: String,
}

/// restore_memory tool input — un-archive one memory.
#[derive(Debug, Deserialize)]
pub struct RestoreMemoryInput {
    pub project_id: String,
    pub memory_type: String,
    pub id: String,
}

/// update_memory tool input — patch selected fields of one memory.
/// All keys other than the three below are captured into `patch`.
#[derive(Debug, Deserialize)]
pub struct UpdateMemoryInput {
    pub project_id: String,
    pub memory_type: String,
    pub id: String,
    #[serde(flatten)]
    pub patch: serde_json::Map<String, serde_json::Value>,
}

/// forget_batch tool input — archive memories matching tags/before. Dry-run by default.
#[derive(Debug, Deserialize)]
pub struct ForgetBatchInput {
    pub project_id: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub before: Option<i64>,
    #[serde(default)]
    pub apply: bool,
}

/// list_archived tool input.
#[derive(Debug, Deserialize)]
pub struct ListArchivedInput {
    pub project_id: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    /// None → `[retrieval].default_limit` from config.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// consolidate_memories tool input — dedup. Dry-run by default.
#[derive(Debug, Deserialize)]
pub struct ConsolidateInput {
    pub project_id: String,
    #[serde(default)]
    pub memory_type: Option<String>,
    #[serde(default)]
    pub include_near_dup: bool,
    #[serde(default)]
    pub apply: bool,
}

/// distill_backlog tool input — the distill loop's material list: raw
/// session-import memories awaiting distillation, oldest first.
#[derive(Debug, Deserialize)]
pub struct DistillBacklogInput {
    pub project_id: String,
    /// None → 10; clamped into 1..=50.
    #[serde(default)]
    pub limit: Option<usize>,
}

/// MCP Tool definition.
#[derive(Debug, Serialize)]
pub struct ToolDefinition {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: serde_json::Value,
}

/// MCP Server (stdio transport).
///
/// Implements a simple JSON-RPC based MCP server over stdio.
/// The MemoryToolProvider trait decouples business logic from
/// the transport layer.
pub trait MemoryToolProvider: Send + Sync {
    // Read tools
    fn search_memory(&self, input: SearchMemoryInput) -> Result<serde_json::Value>;
    fn get_memory(&self, input: GetMemoryInput) -> Result<serde_json::Value>;
    fn related_files(&self, input: RelatedFilesInput) -> Result<serde_json::Value>;
    fn timeline(&self, input: TimelineInput) -> Result<serde_json::Value>;
    fn recent_failures(&self, input: RecentFailuresInput) -> Result<serde_json::Value>;
    fn query_stats(&self, input: QueryStatsInput) -> Result<serde_json::Value>;
    fn session_brief(&self, input: SessionBriefInput) -> Result<serde_json::Value>;
    fn architectural_decisions(
        &self,
        input: ArchitecturalDecisionsInput,
    ) -> Result<serde_json::Value>;
    fn distill_backlog(&self, input: DistillBacklogInput) -> Result<serde_json::Value>;

    // Write tools
    fn create_episodic(&self, input: CreateEpisodicInput) -> Result<serde_json::Value>;
    fn create_decision(&self, input: CreateDecisionInput) -> Result<serde_json::Value>;
    fn create_failure(&self, input: CreateFailureInput) -> Result<serde_json::Value>;
    fn create_procedural(&self, input: CreateProceduralInput) -> Result<serde_json::Value>;
    fn ingest_commits(&self, input: IngestCommitsInput) -> Result<serde_json::Value>;
    fn collect_sources(&self, input: CollectSourcesInput) -> Result<serde_json::Value>;
    fn forget_memory(&self, input: ForgetMemoryInput) -> Result<serde_json::Value>;
    fn restore_memory(&self, input: RestoreMemoryInput) -> Result<serde_json::Value>;
    fn update_memory(&self, input: UpdateMemoryInput) -> Result<serde_json::Value>;
    fn mark_relevance(&self, input: MarkRelevanceInput) -> Result<serde_json::Value>;
    fn forget_batch(&self, input: ForgetBatchInput) -> Result<serde_json::Value>;
    fn list_archived(&self, input: ListArchivedInput) -> Result<serde_json::Value>;
    fn consolidate_memories(&self, input: ConsolidateInput) -> Result<serde_json::Value>;

    // Reflection tools — propose preventive rules from recurring failures.
    fn reflect(&self, input: ReflectInput) -> Result<serde_json::Value>;
    fn list_suggestions(&self, input: ListSuggestionsInput) -> Result<serde_json::Value>;
    fn confirm_suggestion(&self, input: SuggestionIdInput) -> Result<serde_json::Value>;
    fn reject_suggestion(&self, input: SuggestionIdInput) -> Result<serde_json::Value>;
}

/// Default implementation of MemoryToolProvider backed by SQLite.
///
/// `MemoryRepository` holds an r2d2 connection pool (Send+Sync), so `repo`
/// needs no Mutex: read concurrency comes from WAL multi-reader, write
/// concurrency from `busy_timeout` + WAL's single-writer rule. All state lives
/// in SQLite; the provider holds no in-memory caches.
pub struct DefaultMemoryProvider {
    repo: MemoryRepository,
    config: Config,
    classifier: IntentClassifier,
    planner: RetrievalPlanner,
    reranker: Reranker,
    composer: ContextComposer,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    embedding: EmbeddingService,
}

impl DefaultMemoryProvider {
    pub fn new(repo: MemoryRepository, config: Config) -> Self {
        // Build the embedding service (semantic, if enabled) before `config` moves.
        // Read the ranking weights before `config` is moved into the struct.
        let plan_weights = crate::retrieval::planner::PlanWeights {
            relevance: config.retrieval.weight_relevance,
            recency: config.retrieval.weight_recency,
            importance: config.retrieval.weight_importance,
            type_weight: config.retrieval.weight_type,
        };
        Self {
            repo,
            embedding: EmbeddingService::new(&config),
            config,
            classifier: IntentClassifier::new(),
            planner: RetrievalPlanner::new(plan_weights),
            reranker: Reranker::new(),
            composer: ContextComposer::new(),
        }
    }

    /// Borrow the repository. No lock needed: MemoryRepository holds an r2d2
    /// pool (Send+Sync); each method borrows its own pooled connection.
    fn lock_repo(&self) -> &MemoryRepository {
        &self.repo
    }

    /// Record a search in the query log (retrieval-feedback signal for
    /// `query_stats` / `engram queries`). Shared by the CLI `search` command;
    /// the MCP `search_memory` tool records inline with its own rendering.
    pub fn record_query(
        &self,
        project_id: &str,
        query: &str,
        result_ids: &[String],
        memory_type: Option<&str>,
    ) -> Result<()> {
        self.repo.record_query(
            project_id,
            query,
            result_ids,
            memory_type,
            chrono::Utc::now().timestamp(),
        )
    }

    /// Full retrieval pipeline: validate → intent → plan → BM25 (+filters)
    /// → semantic fuse → rerank → truncate. Returns ranked results; callers
    /// decide how to render them.
    pub fn search_core(
        &self,
        query: &str,
        project_id: &str,
        memory_type: Option<&str>,
        tags: &[String],
        before: Option<i64>,
        limit: usize,
    ) -> Result<Vec<crate::retrieval::bm25::SearchResult>> {
        if query.trim().is_empty() {
            anyhow::bail!("query must be a non-empty string");
        }
        if let Some(mt) = memory_type {
            // Fail loudly on a typo'd type instead of silently returning
            // "no memories" — matches get/update/forget behavior.
            MemoryKind::from_type_str(mt)?;
        }

        let repo = self.lock_repo();

        // Intent routing is SOFT: it adjusts ranking weights (relevance/type/
        // recency boosts in the plan below) but never narrows the memory
        // types searched. Hard source-narrowing cost recall — e.g. a Workflow
        // query could no longer surface a decision record about test design.
        let intents = if self.config.retrieval.intent_routing {
            self.classifier.classify(query)
        } else {
            vec![MemoryIntent::General]
        };
        let plan = self.planner.plan(&intents);
        let now_ts = chrono::Utc::now().timestamp();

        // Filters shrink the candidate set post-retrieval, so over-fetch when
        // any is present to keep result counts useful, then truncate to limit.
        let has_filters = !tags.is_empty() || before.is_some();
        let fetch_limit = if has_filters {
            (limit * 3).min(100)
        } else {
            limit
        };

        let mut results = if let Some(mt) = memory_type {
            BM25Retriever::search_by_type(repo, query, project_id, mt, fetch_limit)?
        } else {
            BM25Retriever::search_all(repo, query, project_id, fetch_limit)?
        };

        if has_filters {
            results.retain(|r| {
                let tags_ok = tags.is_empty() || r.tags.iter().any(|t| tags.contains(t));
                let before_ok = before.is_none_or(|b| r.created_at < b);
                tags_ok && before_ok
            });
        }

        // Semantic fusion: when an embedder is present, blend vector top-K with
        // the BM25 candidates (two-arm RRF — see EmbeddingService::fuse).
        // Vector-only hits are materialized so they can surface even when BM25
        // missed them.
        #[cfg(feature = "semantic")]
        {
            if self.embedding.is_active() {
                results = self.embedding.fuse(
                    repo,
                    query,
                    project_id,
                    results,
                    self.config.semantic.top_k,
                    memory_type,
                    tags,
                    before,
                )?;
            }
        }

        self.reranker.deduplicate(&mut results);
        let half_life_seconds = (self.config.retrieval.recency_half_life_days as f32) * 86400.0;
        self.reranker
            .rerank(&mut results, &plan, now_ts, half_life_seconds);

        // Supersede chain: a replaced decision stays searchable as history,
        // but must not outrank the decision that replaced it — halve its
        // score, flag it, and re-sort. This runs BEFORE the truncate so a
        // superseded decision cannot occupy a slot it is about to be demoted
        // out of (it previously displaced a valid result and only then sank).
        let mut supersede_adjusted = false;
        for r in results.iter_mut().filter(|r| r.memory_type == "decision") {
            let superseded = r
                .detail
                .get("superseded_by")
                .and_then(|v| v.as_str())
                .is_some_and(|s| !s.is_empty());
            if superseded {
                r.relevance_score *= 0.5;
                supersede_adjusted = true;
                if let Some(obj) = r.detail.as_object_mut() {
                    obj.insert("superseded".to_string(), serde_json::Value::Bool(true));
                }
            }
        }
        if supersede_adjusted {
            results.sort_by(|a, b| {
                b.relevance_score
                    .partial_cmp(&a.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Graph-based second tier: memories that share an entity (file/tool)
        // with the top hits are contextually related even when their text
        // didn't match the query. Appended below the reranked order, so this
        // can only ADD context, never displace a match. If neighbors exist,
        // the lowest-ranked true hits yield their slots so the response still
        // respects `limit` (previously limit+2 came back); with no neighbors
        // the full limit of true hits is kept. At least one true hit always
        // survives — a lone query match is never displaced by a graph
        // neighbor. Neighbors respect the caller's explicit filters
        // (memory_type/tags/before) with the same semantics as the vector
        // path — a typed or time-bounded search must not leak foreign
        // neighbors through the graph.
        results.truncate(limit);
        if self.config.retrieval.graph_expansion && !results.is_empty() {
            const GRAPH_NEIGHBOR_CAP: usize = 2;
            let seeds: Vec<String> = results.iter().take(3).map(|r| r.id.clone()).collect();
            let existing: std::collections::HashSet<&str> =
                results.iter().map(|r| r.id.as_str()).collect();
            let neighbors_raw = repo.neighbor_memories(project_id, &seeds, GRAPH_NEIGHBOR_CAP)?;
            let fresh: Vec<(String, String)> = neighbors_raw
                .into_iter()
                .filter(|(_, id)| !existing.contains(id.as_str()))
                .collect();
            let mut neighbors: Vec<_> = Vec::new();
            for mut sr in BM25Retriever::fetch_by_ids(repo, &fresh, project_id)? {
                // Explicit filters apply to graph neighbors too, mirroring
                // the vector path in EmbeddingService::fuse: memory_type must
                // match, tags must intersect, created_at must precede
                // `before`.
                let type_ok = memory_type.is_none_or(|mt| sr.memory_type == mt);
                let tags_ok = tags.is_empty() || sr.tags.iter().any(|t| tags.contains(t));
                let before_ok = before.is_none_or(|b| sr.created_at < b);
                if !(type_ok && tags_ok && before_ok) {
                    continue;
                }
                // Fixed low relevance: these are "related by graph", not
                // text matches — they must rank below every true hit.
                sr.relevance_score = 0.30;
                if let Some(obj) = sr.detail.as_object_mut() {
                    obj.insert("graph_neighbor".to_string(), serde_json::Value::Bool(true));
                }
                neighbors.push(sr);
            }
            // At most limit-1 neighbors: keep room for one true hit.
            neighbors.truncate(limit.saturating_sub(1).min(GRAPH_NEIGHBOR_CAP));
            if !neighbors.is_empty() {
                results.truncate(limit - neighbors.len());
                results.extend(neighbors);
            }
        }
        Ok(results)
    }

    /// Backfill embeddings for active memories (semantic feature).
    /// Forwards to the embedded `EmbeddingService`; see its `reindex` for
    /// the full `project`/`force`/`dry_run` semantics.
    #[cfg(feature = "semantic")]
    pub fn reindex_embeddings(
        &self,
        project: Option<&str>,
        force: bool,
        dry_run: bool,
    ) -> Result<crate::mcp::embedding_service::ReindexReport> {
        self.embedding
            .reindex(self.lock_repo(), project, force, dry_run)
    }
}

impl MemoryToolProvider for DefaultMemoryProvider {
    fn search_memory(&self, input: SearchMemoryInput) -> Result<serde_json::Value> {
        let limit = clamp_limit(input.limit, self.config.retrieval.default_limit);
        let now_ts = chrono::Utc::now().timestamp();

        let results = self.search_core(
            &input.query,
            &input.project_id,
            input.memory_type.as_deref(),
            &input.tags,
            input.before,
            limit,
        )?;

        let budget = ContextBudget::new(
            self.config.context.context_window_tokens,
            self.config.context.memory_budget_percent,
        );
        let context = self.composer.compose_context(&results, &budget);

        // Results carry each memory's full type-specific payload (root_cause/
        // fix/prevention, context/tradeoffs, steps, content…): search is the
        // agents' only read path, so a teaser summary is not enough.
        let mut result_items: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                let mut item = serde_json::json!({
                    "id": r.id,
                    // Global search (`project_id = "*"`) mixes projects; the
                    // exact owner must travel with the hit so agents can call
                    // get_memory / mark_relevance / forget_memory on it.
                    "project_id": r.project_id,
                    "memory_type": r.memory_type,
                    "summary": r.summary,
                    "tags": r.tags,
                    "relevance_score": (r.relevance_score.clamp(0.0, 1.0) * 100.0).round() / 100.0,
                    "importance": (r.importance * 100.0).round() / 100.0,
                    "created_at": r.created_at,
                });
                if let (Some(obj), Some(detail)) = (item.as_object_mut(), r.detail.as_object()) {
                    for (k, v) in detail {
                        obj.insert(k.clone(), v.clone());
                    }
                }
                item
            })
            .collect();

        // The token budget now covers the RESULTS array too — it previously
        // only constrained the `context` field while `results` carried every
        // full payload regardless of size. Over budget: strip the detail
        // fields from the LOWEST-ranked results first (agents can re-fetch
        // the full record via get_memory).
        let mut used = crate::context::composer::ContextComposer::estimate_tokens(
            &serde_json::to_string(&result_items).unwrap_or_default(),
        );
        let mut idx = result_items.len();
        while used > budget.reserved_for_memory && idx > 0 {
            idx -= 1;
            if let Some(obj) = result_items[idx].as_object_mut() {
                let keep = [
                    "id",
                    "project_id",
                    "memory_type",
                    "summary",
                    "relevance_score",
                    "importance",
                    "tags",
                    "created_at",
                ];
                obj.retain(|k, _| keep.contains(&k.as_str()));
                obj.insert("detail_omitted".to_string(), serde_json::Value::Bool(true));
            }
            used = crate::context::composer::ContextComposer::estimate_tokens(
                &serde_json::to_string(&result_items).unwrap_or_default(),
            );
        }

        // Best-effort retrieval feedback: log this query + its hits so
        // `query_stats` / `engram queries` can surface hit-rate signal, and
        // `engram tune` learns from impressions/adoptions. Graph neighbors
        // are excluded — they never matched the query, so counting them as
        // impressions poisoned the tuning signal. A logging failure must
        // never break search.
        let result_ids = BM25Retriever::impression_ids(&results);
        if let Err(e) = self.repo.record_query(
            &input.project_id,
            &input.query,
            &result_ids,
            input.memory_type.as_deref(),
            now_ts,
        ) {
            tracing::warn!("query log failed: {e}");
        }

        Ok(serde_json::json!({
            "results": result_items,
            "total": result_items.len(),
            "context": context,
        }))
    }

    fn get_memory(&self, input: GetMemoryInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let kind = MemoryKind::from_type_str(&input.memory_type)?;
        macro_rules! fetch {
            ($get:ident) => {{
                // project scoping is enforced in the SQL itself
                let mem = repo.$get(&input.id, &input.project_id)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "memory not found in project {}: {}",
                        input.project_id,
                        input.id
                    )
                })?;
                serde_json::to_value(&mem)?
            }};
        }
        let memory = match kind {
            MemoryKind::Episodic => fetch!(get_episodic),
            MemoryKind::Decision => fetch!(get_decision),
            MemoryKind::Failure => fetch!(get_failure),
            MemoryKind::Procedural => fetch!(get_procedural),
        };
        // Adoption signal: this memory was returned by a recent search and
        // now actually read. Feeds query_stats adoption rates. Best-effort.
        let fetched_id = memory.get("id").and_then(|v| v.as_str()).unwrap_or("");
        if !fetched_id.is_empty() {
            if let Err(e) = repo.record_adoption(
                &input.project_id,
                fetched_id,
                kind.as_str(),
                chrono::Utc::now().timestamp(),
                24 * 3600,
            ) {
                tracing::warn!("adoption tracking failed: {e}");
            }
        }
        Ok(serde_json::json!({
            "memory_type": kind.as_str(),
            "memory": memory,
        }))
    }

    fn related_files(&self, input: RelatedFilesInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        // Resolve the file's graph neighborhood directly from the indexed
        // graph_relations table — no full-project graph load into memory.
        let (entity_id, edges) = repo.related_files_for(&input.file, &input.project_id)?;
        let found = entity_id.is_some();

        // The primary payload: ACTIVE memories touching this file, each with
        // its memory_type and label so agents can act on them (the old output
        // listed neighbor entity names — for memories that is a bare UUID,
        // unusable without a type for get_memory).
        let memories: Vec<serde_json::Value> = repo
            .related_memories_for_file(&input.file, &input.project_id, 20)?
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "memory_type": m.memory_type,
                    "id": m.id,
                    "summary": m.label,
                    "relation": m.relation,
                })
            })
            .collect();

        // Non-memory neighbors (other files/tools in the graph), kept for
        // completeness.
        let related_entities: Vec<_> = edges
            .iter()
            .map(|e| {
                serde_json::json!({
                    "name": e.other_name,
                    "relation": e.relation_type,
                    "direction": e.direction,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "file": input.file,
            "found": found,
            "memories": memories,
            "related_entities": related_entities,
            "detail": if found { serde_json::Value::Null } else {
                serde_json::json!("no File entity for this name in the project — no memory has touched it (or the file is new)")
            },
        }))
    }

    fn timeline(&self, input: TimelineInput) -> Result<serde_json::Value> {
        // Same validation convention as resolve_kinds: a bad memory_type is a
        // clear error, not a silently empty timeline.
        let kind = match &input.memory_type {
            Some(s) => Some(MemoryKind::from_type_str(s)?),
            None => None,
        };
        let repo = self.lock_repo();
        let since = chrono::Utc::now().timestamp() - clamp_days(input.days).saturating_mul(86400);
        let rows = repo.timeline(&input.project_id, since, input.until, kind)?;
        let events: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "day": r.day,
                    "counts": {
                        "episodic": r.episodic,
                        "decision": r.decision,
                        "failure": r.failure,
                        "procedural": r.procedural,
                        "total": r.episodic + r.decision + r.failure + r.procedural,
                    },
                })
            })
            .collect();
        // Echo the window actually used (days is clamped 0..=3650); the
        // optional filters appear only when the caller supplied them.
        let mut out = serde_json::json!({
            "project_id": input.project_id,
            "days": clamp_days(input.days),
            "events": events,
        });
        if let Some(mt) = &input.memory_type {
            out["memory_type"] = serde_json::json!(mt);
        }
        if let Some(until) = input.until {
            out["until"] = serde_json::json!(until);
        }
        Ok(out)
    }

    fn query_stats(&self, input: QueryStatsInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let limit = clamp_limit(input.limit, self.config.retrieval.default_limit);
        let since = chrono::Utc::now().timestamp() - clamp_days(input.days).saturating_mul(86400);
        let rows = repo.query_stats(&input.project_id, since, limit)?;
        let queries: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "query": r.query,
                    "count": r.count,
                    "result_count_avg": (r.result_count_avg * 100.0).round() / 100.0,
                    "last_at": r.last_at,
                })
            })
            .collect();
        Ok(serde_json::json!({ "queries": queries }))
    }

    fn recent_failures(&self, input: RecentFailuresInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let limit = clamp_limit(input.limit, self.config.retrieval.default_limit);
        let query = input.service.as_deref().unwrap_or("");
        let results = if query.is_empty() {
            repo.list_recent_failures(&input.project_id, limit)?
                .into_iter()
                .map(|m| ScoredMemory {
                    memory: m,
                    bm25_score: 0.0,
                })
                .collect()
        } else {
            repo.search_failures(query, &input.project_id, limit)?
        };
        // Full failure record: the root-cause analysis is the whole point of
        // a failure memory — hiding it behind a follow-up call that doesn't
        // exist made stored knowledge unreadable.
        let failures: Vec<serde_json::Value> = results
            .iter()
            .map(|f| {
                serde_json::json!({
                    "id": f.memory.id,
                    "incident": f.memory.incident,
                    "root_cause": f.memory.root_cause,
                    "fix": f.memory.fix,
                    "prevention": f.memory.prevention,
                    "severity": f.memory.severity,
                    "tags": f.memory.tags,
                    "created_at": f.memory.created_at,
                })
            })
            .collect();

        Ok(serde_json::json!({ "failures": failures }))
    }

    fn session_brief(&self, input: SessionBriefInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();

        let decisions: Vec<_> = repo
            .list_recent_decisions(&input.project_id, 8)?
            .into_iter()
            // Superseded decisions are history — the brief carries CURRENT
            // constraints only.
            .filter(|d| d.superseded_by.is_none())
            .take(5)
            .collect();
        let failures = repo.list_recent_failures(&input.project_id, 5)?;
        let procedures = repo.list_top_procedural(&input.project_id, 5)?;
        let pending = repo.list_pending_suggestions(&input.project_id)?;
        // Raw session imports awaiting distillation — surfacing the count
        // here is the distill loop's trigger: the agent sees the backlog in
        // every session brief and can fetch the `engram.distill` prompt.
        // Distilled typed memories (source tag kept, `distilled` added) no
        // longer count; the oldest raw import's age hints at how overdue
        // the loop is.
        let (raw_session_imports, oldest_raw_import) =
            repo.count_session_imports(&input.project_id)?;

        // Optional focused context: run the query through the same pipeline
        // search_memory uses (intent → BM25 → rerank), capped at 3 hits.
        let focused = match input.focus.as_deref() {
            Some(q) if !q.trim().is_empty() => Some(
                self.search_core(q, &input.project_id, None, &[], None, 3)
                    .unwrap_or_default(),
            ),
            _ => None,
        };

        // ── Text brief: the section order is the priority order; over
        // budget, lines drop from the END (suggestions first, decisions last).
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!(
            "=== Engram Session Brief: {} ===",
            input.project_id
        ));
        if raw_session_imports > 0 {
            let oldest = oldest_raw_import
                .map(|ts| format!("oldest {} days ago; ", (now - ts).max(0) / 86_400))
                .unwrap_or_default();
            lines.push(format!(
                "-- {raw_session_imports} raw session-import memories awaiting distillation \
                 ({oldest}fetch prompt engram.distill to refine them into typed memories) --"
            ));
        }
        if !decisions.is_empty() {
            lines.push("-- Decisions (constraints to respect) --".into());
            for d in &decisions {
                lines.push(format!(
                    "[decision] {} — {}",
                    d.title,
                    first_sentence(&d.rationale)
                ));
            }
        }
        if !failures.is_empty() {
            lines.push("-- Recent failures (do not repeat) --".into());
            for f in &failures {
                lines.push(format!(
                    "[failure s{}/5] {} → prevention: {}",
                    f.severity,
                    f.incident,
                    first_sentence(&f.prevention)
                ));
            }
        }
        if !procedures.is_empty() {
            lines.push("-- Procedures (how this project works) --".into());
            for p in &procedures {
                lines.push(format!(
                    "[procedure] {}: {}",
                    p.workflow_name,
                    p.steps.join(" → ")
                ));
            }
        }
        if !pending.is_empty() {
            lines.push(format!(
                "-- Pending reflection suggestions ({}) --",
                pending.len()
            ));
            for s in &pending {
                if self.config.reflection.agent_confirm {
                    lines.push(format!(
                        "[suggestion x{}] {} — confirm or reject via list_suggestions",
                        s.occurrence_count, s.suggested_workflow_name
                    ));
                } else {
                    lines.push(format!(
                        "[suggestion x{}] {} — awaiting human confirmation \
                         (engram confirm-suggestion / reject-suggestion)",
                        s.occurrence_count, s.suggested_workflow_name
                    ));
                }
            }
        }
        if let Some(hits) = &focused {
            lines.push(format!(
                "-- Focused context: {:?} --",
                input.focus.as_deref().unwrap_or("")
            ));
            for r in hits {
                lines.push(format!(
                    "[{}] {} (score {:.2})",
                    r.memory_type,
                    r.summary,
                    r.relevance_score.clamp(0.0, 1.0)
                ));
            }
        }
        lines.push("=== End Brief ===".into());

        let budget = ContextBudget::new(
            self.config.context.context_window_tokens,
            self.config.context.memory_budget_percent,
        );
        while lines.len() > 2
            && ContextComposer::estimate_tokens(&lines.join("\n")) > budget.reserved_for_memory
        {
            // Drop the lowest-priority line (never the framing header/footer).
            lines.pop();
            // Keep section headers from dangling empty.
            if lines.last().is_some_and(|l| l.starts_with("--")) {
                lines.pop();
            }
        }

        // ── Structured sections: full payloads for agents that want fields.
        let decisions_out: Vec<_> = decisions
            .iter()
            .map(|d| {
                serde_json::json!({
                    "id": d.id, "title": d.title, "rationale": d.rationale,
                    "importance": d.importance, "tags": d.tags, "created_at": d.created_at,
                })
            })
            .collect();
        let failures_out: Vec<_> = failures
            .iter()
            .map(|f| {
                serde_json::json!({
                    "id": f.id, "incident": f.incident, "root_cause": f.root_cause,
                    "fix": f.fix, "prevention": f.prevention, "severity": f.severity,
                    "tags": f.tags, "created_at": f.created_at,
                })
            })
            .collect();
        let procedures_out: Vec<_> = procedures
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": p.id, "workflow_name": p.workflow_name, "steps": p.steps,
                    "related_tools": p.related_tools, "importance": p.importance,
                })
            })
            .collect();
        let suggestions_out: Vec<_> = pending
            .iter()
            .map(|s| {
                serde_json::json!({
                    "id": s.id, "workflow_name": s.suggested_workflow_name,
                    "occurrence_count": s.occurrence_count, "pattern_tag": s.pattern_tag,
                })
            })
            .collect();
        let focused_out: Option<Vec<_>> = focused.as_ref().map(|hits| {
            hits.iter()
                .map(|r| {
                    serde_json::json!({
                        "id": r.id, "memory_type": r.memory_type, "summary": r.summary,
                        "relevance_score": r.relevance_score.clamp(0.0, 1.0),
                    })
                })
                .collect()
        });

        Ok(serde_json::json!({
            "project_id": input.project_id,
            "brief": lines.join("\n"),
            "decisions": decisions_out,
            "failures": failures_out,
            "procedures": procedures_out,
            "pending_suggestions": suggestions_out,
            "raw_session_imports": raw_session_imports,
            "focused": focused_out,
            "generated_at": now,
        }))
    }

    fn distill_backlog(&self, input: DistillBacklogInput) -> Result<serde_json::Value> {
        use crate::consolidation::engine::{SESSION_IMPORT_TPL_ERRORS, SESSION_IMPORT_TPL_FILES};

        let repo = self.lock_repo();
        // Backlog pages are read top-down by the distill agent — a bound an
        // order of magnitude below search_memory's keeps the response small.
        let limit = input.limit.unwrap_or(10).clamp(1, 50);
        let now = chrono::Utc::now().timestamp();
        let (items, total) = repo.list_distill_backlog(&input.project_id, limit)?;
        let items: Vec<_> = items
            .iter()
            .map(|m| {
                // Signal overview parsed from the session-import template
                // headers (built by `cli::commands::session_import_content`
                // from THESE constants). A section that is absent (nothing of
                // that kind was recorded) parses to null, not 0.
                let error_count = parse_session_import_count(&m.content, SESSION_IMPORT_TPL_ERRORS);
                let file_count = parse_session_import_count(&m.content, SESSION_IMPORT_TPL_FILES);
                let age_days = (now - m.created_at).max(0) / 86_400;
                // The content starts with template scaffold on its own lines;
                // flatten whitespace so the preview is one readable line.
                let flat: String = m.content.split_whitespace().collect::<Vec<_>>().join(" ");
                serde_json::json!({
                    "id": m.id,
                    "memory_type": "episodic",
                    "summary": m.summary,
                    "created_at": m.created_at,
                    "age_days": age_days,
                    "error_count": error_count,
                    "file_count": file_count,
                    "preview": first_sentence(&flat),
                })
            })
            .collect();
        Ok(serde_json::json!({ "total": total, "items": items }))
    }

    fn architectural_decisions(
        &self,
        input: ArchitecturalDecisionsInput,
    ) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let limit = clamp_limit(input.limit, self.config.retrieval.default_limit);
        let query = input.topic.as_deref().unwrap_or("");
        let mut results = if query.is_empty() {
            repo.list_recent_decisions(&input.project_id, limit * 2)?
                .into_iter()
                .map(|m| ScoredMemory {
                    memory: m,
                    bm25_score: 0.0,
                })
                .collect::<Vec<_>>()
        } else {
            repo.search_decisions(query, &input.project_id, limit * 2)?
        };
        // Superseded decisions are excluded by default (they are history, not
        // current constraints); over-fetch above keeps the page full.
        if !input.include_superseded {
            results.retain(|d| d.memory.superseded_by.is_none());
        }
        results.truncate(limit);
        let decisions: Vec<serde_json::Value> = results
            .iter()
            .map(|d| {
                let mut item = serde_json::json!({
                    "id": d.memory.id,
                    "title": d.memory.title,
                    "context": d.memory.context,
                    "rationale": d.memory.rationale,
                    "tradeoffs": d.memory.tradeoffs,
                    "related_files": d.memory.related_files,
                    "tags": d.memory.tags,
                    "created_at": d.memory.created_at,
                });
                if let Some(sb) = &d.memory.superseded_by {
                    item["superseded_by"] = serde_json::json!(sb);
                }
                item
            })
            .collect();

        Ok(serde_json::json!({ "decisions": decisions }))
    }

    // ─── Write Tools ───────────────────────────────────────────────

    fn create_episodic(&self, input: CreateEpisodicInput) -> Result<serde_json::Value> {
        if !(0.0..=1.0).contains(&input.importance) {
            anyhow::bail!(
                "importance must be between 0.0 and 1.0, got {}",
                input.importance
            );
        }
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();
        let id = uuid::Uuid::new_v4().to_string();

        let memory = EpisodicMemory {
            id: id.clone(),
            project_id: input.project_id,
            session_id: input.session_id,
            summary: input.summary,
            content: input.content,
            files_touched: input.files_touched,
            related_commits: input.related_commits,
            importance: input.importance,
            tags: input.tags,
            created_at: now,
            updated_at: now,
        };

        // Write-time exact-dup guard: with write_dedup on, the row's
        // content_hash is written and the partial unique index enforces dedup
        // ATOMICALLY inside the INSERT — the old two-step
        // find_exact_duplicate → create raced (both callers passed the check
        // and double-wrote). On collision the existing ACTIVE row's id comes
        // back and is answered as "duplicate", not an error (agents retry
        // writes; near-dups stay consolidate's job). With write_dedup off the
        // row carries no hash and duplicates are created freely.
        let duplicate = if self.config.retrieval.write_dedup {
            repo.create_episodic_dedup(&memory)?
        } else {
            repo.create_episodic(&memory)?;
            None
        };
        if let Some(existing) = duplicate {
            return Ok(serde_json::json!({
                "id": existing,
                "status": "duplicate",
                "detail": "identical active memory already exists; nothing written",
            }));
        }
        #[cfg(feature = "semantic")]
        self.embedding.index(
            repo,
            "episodic",
            &memory.id,
            &memory.project_id,
            &memory.embedding_text(),
        );

        Ok(serde_json::json!({
            "id": id,
            "status": "created",
            "created_at": now,
        }))
    }

    fn create_decision(&self, input: CreateDecisionInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();

        // Fail fast BEFORE any write: a bad `supersedes` id must not leave a
        // committed decision behind with the chain broken (the old two-step
        // path could do exactly that, and the retry then hit the dedup guard
        // which returned early without ever performing the supersede).
        if let Some(old_id) = &input.supersedes {
            if repo.get_decision(old_id, &input.project_id)?.is_none() {
                anyhow::bail!(
                    "cannot supersede: no decision {old_id} in project {}",
                    input.project_id
                );
            }
        }

        // Write-time exact-dup guard (see create_episodic): the partial
        // unique index on content_hash enforces dedup inside the INSERT.
        let id = uuid::Uuid::new_v4().to_string();

        let memory = DecisionMemory {
            id: id.clone(),
            project_id: input.project_id.clone(),
            title: input.title,
            context: input.context,
            rationale: input.rationale,
            tradeoffs: input.tradeoffs,
            related_files: input.related_files,
            tags: input.tags,
            importance: input.importance.unwrap_or(0.5).clamp(0.0, 1.0),
            superseded_by: None,
            created_at: now,
            updated_at: now,
        };

        // Create + supersede in ONE transaction (validated above); without
        // the supersede there is nothing to couple and the plain create runs.
        // The dedup flag rides along so the collision (an identical ACTIVE
        // decision already exists) surfaces as the existing row's id instead
        // of a double write.
        let duplicate = match &input.supersedes {
            Some(old_id) => repo.create_decision_with_supersede(
                &memory,
                old_id,
                self.config.retrieval.write_dedup,
            )?,
            None => {
                if self.config.retrieval.write_dedup {
                    repo.create_decision_dedup(&memory)?
                } else {
                    repo.create_decision(&memory)?;
                    None
                }
            }
        };
        if let Some(existing) = duplicate {
            // Retry of a supersede that failed around an earlier create: the
            // decision exists but the chain may be missing — apply the
            // supersede to the EXISTING record instead of silently returning
            // "duplicate" with the chain still broken.
            let mut supersede_applied = serde_json::Value::Null;
            if let Some(old_id) = &input.supersedes {
                supersede_applied = serde_json::json!(old_id);
                if let Err(e) = repo.supersede_decision(&input.project_id, old_id, &existing, now) {
                    // Race (old deleted between the pre-check and here):
                    // report, don't fail — the memory itself is intact.
                    supersede_applied = serde_json::json!({
                        "error": e.to_string(),
                    });
                }
            }
            return Ok(serde_json::json!({
                "id": existing,
                "status": "duplicate",
                "superseded": supersede_applied,
                "detail": "identical active decision already exists; nothing written \
                           (pass `supersedes` when intentionally replacing a decision)",
            }));
        }
        #[cfg(feature = "semantic")]
        self.embedding.index(
            repo,
            "decision",
            &memory.id,
            &memory.project_id,
            &memory.embedding_text(),
        );

        let superseded = match &input.supersedes {
            Some(old_id) => serde_json::json!(old_id),
            None => serde_json::Value::Null,
        };
        Ok(serde_json::json!({
            "id": id,
            "status": "created",
            "superseded": superseded,
            "created_at": now,
        }))
    }

    fn create_failure(&self, input: CreateFailureInput) -> Result<serde_json::Value> {
        if !(1..=5).contains(&input.severity) {
            anyhow::bail!("severity must be between 1 and 5, got {}", input.severity);
        }
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();
        let id = uuid::Uuid::new_v4().to_string();

        let memory = FailureMemory {
            id: id.clone(),
            project_id: input.project_id,
            incident: input.incident,
            root_cause: input.root_cause,
            fix: input.fix,
            prevention: input.prevention,
            severity: input.severity,
            tags: input.tags,
            created_at: now,
            updated_at: now,
        };

        // Write-time exact-dup guard (see create_episodic).
        let duplicate = if self.config.retrieval.write_dedup {
            repo.create_failure_dedup(&memory)?
        } else {
            repo.create_failure(&memory)?;
            None
        };
        if let Some(existing) = duplicate {
            return Ok(serde_json::json!({
                "id": existing,
                "status": "duplicate",
                "detail": "identical active failure already exists; nothing written",
            }));
        }
        #[cfg(feature = "semantic")]
        self.embedding.index(
            repo,
            "failure",
            &memory.id,
            &memory.project_id,
            &memory.embedding_text(),
        );

        Ok(serde_json::json!({
            "id": id,
            "status": "created",
            "severity": input.severity,
            "created_at": now,
        }))
    }

    fn create_procedural(&self, input: CreateProceduralInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();
        let id = uuid::Uuid::new_v4().to_string();

        let memory = ProceduralMemory {
            id: id.clone(),
            project_id: input.project_id,
            workflow_name: input.workflow_name,
            steps: input.steps,
            related_tools: input.related_tools,
            tags: input.tags,
            importance: input.importance.unwrap_or(0.5).clamp(0.0, 1.0),
            created_at: now,
            updated_at: now,
        };

        // Write-time exact-dup guard (see create_episodic). The steps column
        // stores the JSON array string, so the hashed dedup text matches it.
        let duplicate = if self.config.retrieval.write_dedup {
            repo.create_procedural_dedup(&memory)?
        } else {
            repo.create_procedural(&memory)?;
            None
        };
        if let Some(existing) = duplicate {
            return Ok(serde_json::json!({
                "id": existing,
                "status": "duplicate",
                "detail": "identical active procedure already exists; nothing written",
            }));
        }
        #[cfg(feature = "semantic")]
        self.embedding.index(
            repo,
            "procedural",
            &memory.id,
            &memory.project_id,
            &memory.embedding_text(),
        );

        Ok(serde_json::json!({
            "id": id,
            "status": "created",
            "created_at": now,
        }))
    }

    fn reflect(&self, input: ReflectInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let min = input
            .min_occurrences
            .unwrap_or(self.config.reflection.min_occurrences);
        let engine = crate::reflection::ReflectionEngine::with_min_occurrences(min);
        let plan = engine.reflect(
            repo,
            &input.project_id,
            input.apply,
            chrono::Utc::now().timestamp(),
        )?;
        Ok(serde_json::json!({
            "applied": input.apply,
            "min_occurrences": min,
            "proposed": plan.suggestions.len(),
            "created": plan.created,
            "suggestions": serde_json::to_value(&plan.suggestions)?,
        }))
    }

    fn list_suggestions(&self, input: ListSuggestionsInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let pending = repo.list_pending_suggestions(&input.project_id)?;
        Ok(serde_json::json!({
            "count": pending.len(),
            "suggestions": serde_json::to_value(&pending)?,
        }))
    }

    fn confirm_suggestion(&self, input: SuggestionIdInput) -> Result<serde_json::Value> {
        // Optional human gate: `reflection.agent_confirm = false` reserves
        // confirmation for the CLI, making the documented "awaiting human
        // confirmation" model enforceable instead of aspirational.
        if !self.config.reflection.agent_confirm {
            anyhow::bail!(
                "confirm_suggestion is disabled by config \
                 ([reflection] agent_confirm = false): suggestions are confirmed \
                 by a human via `engram confirm-suggestion {}`",
                input.id
            );
        }
        let repo = self.lock_repo();
        match repo.confirm_suggestion(
            &input.id,
            &input.project_id,
            chrono::Utc::now().timestamp(),
        )? {
            Some(proc_id) => Ok(serde_json::json!({
                "id": input.id,
                "status": "confirmed",
                "procedural_id": proc_id,
            })),
            None => anyhow::bail!(
                "no pending suggestion '{}' in project '{}'",
                input.id,
                input.project_id
            ),
        }
    }

    fn reject_suggestion(&self, input: SuggestionIdInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let rejected =
            repo.reject_suggestion(&input.id, &input.project_id, chrono::Utc::now().timestamp())?;
        if !rejected {
            anyhow::bail!(
                "no pending suggestion '{}' in project '{}'",
                input.id,
                input.project_id
            );
        }
        Ok(serde_json::json!({ "id": input.id, "status": "rejected" }))
    }

    fn ingest_commits(&self, input: IngestCommitsInput) -> Result<serde_json::Value> {
        let count = input.count.clamp(1, 1000);
        // repo_path is agent-controlled: validate before touching the disk
        // (home dir / FS root / outside-allowlist rejection).
        let repo_path = crate::path_guard::validate_repo_path(
            std::path::Path::new(&input.repo_path),
            &self.config.security.allowed_roots,
        )?;
        let git = GitIntegration::new(&repo_path)?;
        let session_id = input.session_id.unwrap_or_else(|| "auto-ingest".into());

        let events = git.get_recent_commits(count)?;
        // Count against what was actually available: a shallow repo with 5
        // commits and count=20 has 0 duplicates, not 15 (the old `count -
        // fresh` arithmetic reported missing history as "skipped").
        let scanned = events.len();
        let repo = self.lock_repo();

        // Dedup BEFORE generating: the old path built a full memory (with
        // tree-diff-derived files) for every commit and threw most away.
        let ingested_hashes = repo.get_ingested_commits(&input.project_id)?;
        let fresh: Vec<crate::git_integration::CommitEvent> = events
            .into_iter()
            .filter(|e| !ingested_hashes.contains(&e.commit_hash))
            .collect();
        let skipped = scanned.saturating_sub(fresh.len());

        // One memory per (type, scope) milestone instead of one per commit —
        // the distillation git_collector's bootstrap path always intended.
        let memories =
            crate::git_integration::milestone_memories(&input.project_id, &session_id, &fresh);

        let mut ingested = Vec::new();
        for mem in &memories {
            repo.create_episodic(mem)?;
            // Embedding parity with the manual create path — without this,
            // git-ingested memories were invisible to semantic search until a
            // manual `engram reindex`.
            #[cfg(feature = "semantic")]
            self.embedding.index(
                repo,
                "episodic",
                &mem.id,
                &mem.project_id,
                &mem.embedding_text(),
            );
            ingested.push(serde_json::json!({
                "id": mem.id,
                "summary": mem.summary,
                "commits": mem.related_commits.len(),
                "files_touched": mem.files_touched,
            }));
        }

        Ok(serde_json::json!({
            "ingested": ingested.len(),
            "total_commits_scanned": scanned,
            "skipped_duplicates": skipped,
            "memories": ingested,
        }))
    }

    fn collect_sources(&self, input: CollectSourcesInput) -> Result<serde_json::Value> {
        let dimensions = collectors::Dimension::parse_list(input.dimensions.as_deref());
        if dimensions.is_empty() {
            anyhow::bail!(
                "no valid dimensions parsed from {:?}; valid: git, decisions, failures, workflow",
                input.dimensions
            );
        }

        // Reuse the commit-hash dedup so re-collection stays idempotent: commits
        // already stored as episodic memories are skipped by the git collector.
        let ingested_hashes = {
            let repo = self.lock_repo();
            repo.get_ingested_commits(&input.project_id)?
        };

        let opts = collectors::CollectOptions {
            max_commits: input.max_commits.clamp(1, 1000),
            ingested_commit_hashes: ingested_hashes,
            allowed_roots: self.config.security.allowed_roots.clone(),
            ..Default::default()
        };

        let sources = collectors::collect(
            &input.project_id,
            std::path::Path::new(&input.repo_path),
            &dimensions,
            &opts,
        )?;

        // Serialize directly so the agent receives the full structured bundle.
        Ok(serde_json::to_value(&sources)?)
    }

    fn forget_memory(&self, input: ForgetMemoryInput) -> Result<serde_json::Value> {
        let kind = MemoryKind::from_type_str(&input.memory_type)?;
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();
        let archived = repo.archive(kind, &input.id, &input.project_id, now)?;
        #[cfg(feature = "semantic")]
        if archived {
            // The vector cache still holds the now-archived memory.
            self.embedding.invalidate_vectors(Some(&input.project_id));
        }
        Ok(serde_json::json!({
            "id": input.id,
            "archived": archived,
            "status": if archived { "archived" } else { "not_found_or_already_archived" },
        }))
    }

    fn restore_memory(&self, input: RestoreMemoryInput) -> Result<serde_json::Value> {
        let kind = MemoryKind::from_type_str(&input.memory_type)?;
        let repo = self.lock_repo();
        let restored = repo.restore(kind, &input.id, &input.project_id)?;
        #[cfg(feature = "semantic")]
        if restored {
            self.embedding.invalidate_vectors(Some(&input.project_id));
        }
        Ok(serde_json::json!({
            "id": input.id,
            "restored": restored,
            "status": if restored { "restored" } else { "not_found_or_active" },
        }))
    }

    fn update_memory(&self, input: UpdateMemoryInput) -> Result<serde_json::Value> {
        let kind = MemoryKind::from_type_str(&input.memory_type)?;
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();

        // A patch with no fields is a caller bug — most often a strict MCP
        // client stripping keys the inputSchema did not declare (the old
        // schema declared only the three fixed params). That used to land
        // here as a silent "updated" that changed nothing but updated_at:
        // the agent read the memory back unchanged, retried with more
        // parameters, and looped. Fail loudly with guidance instead.
        if input.patch.is_empty() {
            anyhow::bail!(
                "empty patch: no fields to update. Pass the fields to change \
                 alongside project_id/memory_type/id — see this tool's schema \
                 for the fields each memory_type accepts"
            );
        }

        macro_rules! guarded_update {
            ($get:ident, $update:ident) => {{
                // project scoping is enforced in the SQL itself
                let existing = repo.$get(&input.id, &input.project_id)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "memory not found in project {}: {}",
                        input.project_id,
                        input.id
                    )
                })?;
                let existing_value = serde_json::to_value(&existing)?;
                let (applied, ignored) =
                    partition_patch(&input.memory_type, &existing_value, &input.patch)?;
                if applied.is_empty() {
                    // Two distinct caller mistakes, two distinct messages so
                    // the retry can be a fix instead of another loop: only
                    // protected keys vs fields of a different memory_type.
                    if input
                        .patch
                        .keys()
                        .all(|k| PROTECTED_PATCH_KEYS.contains(&k.as_str()))
                    {
                        let keys: Vec<&str> = input.patch.keys().map(String::as_str).collect();
                        anyhow::bail!(
                            "no fields to update: patch contained only protected keys \
                             ({keys:?}) — those are never changed; pass the fields to \
                             change instead"
                        );
                    }
                    let wrong: Vec<&str> = input
                        .patch
                        .keys()
                        .filter(|k| !PROTECTED_PATCH_KEYS.contains(&k.as_str()))
                        .map(String::as_str)
                        .collect();
                    anyhow::bail!(
                        "no fields to update: {wrong:?} do not apply to memory_type '{}'; \
                         applicable fields: {}",
                        input.memory_type,
                        applicable_field_list(&existing_value)
                    );
                }
                let updated = merge_patch(&existing, &applied, now)?;
                repo.$update(&updated)?;
                (applied, ignored)
            }};
        }

        let (applied, ignored) = match kind {
            MemoryKind::Episodic => guarded_update!(get_episodic, update_episodic),
            MemoryKind::Decision => guarded_update!(get_decision, update_decision),
            MemoryKind::Failure => guarded_update!(get_failure, update_failure),
            MemoryKind::Procedural => guarded_update!(get_procedural, update_procedural),
        };

        // Semantic freshness: re-embed the updated text so vector search
        // matches what the memory now says. The create path embeds; without
        // this, update left the OLD text's vector in place indefinitely.
        #[cfg(feature = "semantic")]
        {
            if self.embedding.is_active() {
                macro_rules! reembed {
                    ($get:ident) => {{
                        if let Some(m) = self.repo.$get(&input.id, &input.project_id)? {
                            self.embedding.index(
                                &self.repo,
                                kind.as_str(),
                                &input.id,
                                &input.project_id,
                                &m.embedding_text(),
                            );
                        }
                    }};
                }
                match kind {
                    MemoryKind::Episodic => reembed!(get_episodic),
                    MemoryKind::Decision => reembed!(get_decision),
                    MemoryKind::Failure => reembed!(get_failure),
                    MemoryKind::Procedural => reembed!(get_procedural),
                }
            }
        }

        // Echo what actually changed: `applied` maps each written field to
        // its new value (no follow-up get_memory needed to verify), and
        // `ignored` explains every key that was not applied. Omitted when
        // empty to keep the common all-valid case terse.
        let mut response = serde_json::json!({
            "id": input.id,
            "status": "updated",
            "applied": serde_json::Value::Object(applied),
            "updated_at": now,
        });
        if !ignored.is_empty() {
            response
                .as_object_mut()
                .expect("response is an object")
                .insert("ignored".into(), serde_json::Value::Object(ignored));
        }
        Ok(response)
    }

    fn mark_relevance(&self, input: MarkRelevanceInput) -> Result<serde_json::Value> {
        let kind = MemoryKind::from_type_str(&input.memory_type)?;
        let step = match input.feedback.as_str() {
            "useful" => 0.1,
            "irrelevant" => -0.1,
            other => anyhow::bail!("feedback must be 'useful' or 'irrelevant', got {other:?}"),
        };
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();

        // Feedback cooldown (24h, one judgment per memory per day). Keyed on
        // the memory_feedback table, NOT adoptions: get_memory writes an
        // adoption row, and keying the cooldown there meant the recommended
        // search → get_memory → mark_relevance flow ALWAYS hit cooldown —
        // importance never moved. Direction-agnostic: the symmetric window
        // keeps a client loop from swinging importance to the bounds
        // (10 × useful = +1.0) or grinding it to the floor with 'irrelevant'.
        const FEEDBACK_WINDOW_SECS: i64 = 24 * 3600;
        let cooled =
            repo.has_recent_feedback(&input.project_id, &input.id, now, FEEDBACK_WINDOW_SECS)?;
        if cooled {
            return Ok(serde_json::json!({
                "id": input.id,
                "feedback": input.feedback,
                "status": "cooldown",
                "detail": "feedback already recorded for this memory in the last 24h; \
                           importance unchanged",
            }));
        }

        macro_rules! adjust_importance {
            ($get:ident, $update:ident) => {{
                let mut mem = repo.$get(&input.id, &input.project_id)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "memory not found in project {}: {}",
                        input.project_id,
                        input.id
                    )
                })?;
                mem.importance = (mem.importance + step).clamp(0.0, 1.0);
                mem.updated_at = now;
                let importance = mem.importance;
                repo.$update(&mem)?;
                importance
            }};
        }
        // Failure memories have no importance field — severity IS their
        // importance signal (severity/5 in the reranker), so feedback adjusts
        // severity by half a point, clamped to the 1..=5 scale.
        macro_rules! adjust_severity {
            ($get:ident, $update:ident) => {{
                let mut mem = repo.$get(&input.id, &input.project_id)?.ok_or_else(|| {
                    anyhow::anyhow!(
                        "memory not found in project {}: {}",
                        input.project_id,
                        input.id
                    )
                })?;
                let delta = (step * 5.0).round() as i8;
                mem.severity = (mem.severity as i8 + delta).clamp(1, 5) as u8;
                mem.updated_at = now;
                let importance = mem.severity as f32 / 5.0;
                repo.$update(&mem)?;
                importance
            }};
        }
        let importance = match kind {
            MemoryKind::Episodic => adjust_importance!(get_episodic, update_episodic),
            MemoryKind::Decision => adjust_importance!(get_decision, update_decision),
            MemoryKind::Failure => adjust_severity!(get_failure, update_failure),
            MemoryKind::Procedural => adjust_importance!(get_procedural, update_procedural),
        };

        // The judgment itself: drives the cooldown above (both directions)
        // and the tuner's demote guard (has_useful_feedback_since).
        repo.record_feedback(
            &input.project_id,
            &input.id,
            kind.as_str(),
            &input.feedback,
            now,
        )?;

        // Explicit "useful" is also an adoption event for retrieval feedback
        // — a stronger signal than a passive read. Dedup stays inside
        // record_adoption. "irrelevant" is a ranking judgment, not evidence
        // about adoption: it writes no adoption row.
        if step > 0.0 {
            let _ =
                repo.record_adoption(&input.project_id, &input.id, kind.as_str(), now, 24 * 3600);
        }

        Ok(serde_json::json!({
            "id": input.id,
            "feedback": input.feedback,
            "importance": importance,
        }))
    }

    fn forget_batch(&self, input: ForgetBatchInput) -> Result<serde_json::Value> {
        let kinds = resolve_kinds(&input.memory_type)?;
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();
        let mut matched: Vec<serde_json::Value> = Vec::new();

        for kind in kinds {
            if input.apply {
                let ids =
                    repo.archive_batch(kind, &input.project_id, &input.tags, input.before, now)?;
                for id in ids {
                    matched.push(serde_json::json!({ "id": id, "memory_type": kind.as_str() }));
                }
            } else {
                // dry-run: list candidates without mutating.
                for row in
                    repo.list_active_candidates(kind, &input.project_id, &input.tags, input.before)?
                {
                    matched.push(serde_json::json!({ "id": row, "memory_type": kind.as_str() }));
                }
            }
        }

        Ok(serde_json::json!({
            "applied": input.apply,
            "matched": matched,
            "count": matched.len(),
        }))
    }

    fn list_archived(&self, input: ListArchivedInput) -> Result<serde_json::Value> {
        let kinds = resolve_kinds(&input.memory_type)?;
        let repo = self.lock_repo();
        let limit = clamp_limit(input.limit, self.config.retrieval.default_limit);
        let mut archived: Vec<serde_json::Value> = Vec::new();
        for kind in kinds {
            for row in repo.list_archived(kind, &input.project_id, limit)? {
                archived.push(serde_json::to_value(&row)?);
            }
        }
        Ok(serde_json::json!({ "archived": archived, "count": archived.len() }))
    }

    fn consolidate_memories(&self, input: ConsolidateInput) -> Result<serde_json::Value> {
        let kinds = resolve_kinds(&input.memory_type)?;
        let repo = self.lock_repo();
        let now = chrono::Utc::now().timestamp();
        let engine = ConsolidationEngine::new();
        let plans = engine.consolidate(
            repo,
            &input.project_id,
            &kinds,
            input.include_near_dup,
            CONSOLIDATE_JACCARD_THRESHOLD,
            input.apply,
            now,
        )?;
        let total_archived: usize = plans.iter().map(|p| p.archived).sum();
        #[cfg(feature = "semantic")]
        if input.apply && total_archived > 0 {
            self.embedding.invalidate_vectors(Some(&input.project_id));
        }
        Ok(serde_json::json!({
            "applied": input.apply,
            "plans": plans,
            "total_archived": total_archived,
        }))
    }
}

/// Keys update_memory never writes, whether or not a patch carries them:
/// identity (id/project_id), routing (memory_type), and bookkeeping
/// (created_at/updated_at — updated_at is always stamped server-side, so a
/// patch value for it never took effect; it is reported instead of being
/// silently overwritten).
const PROTECTED_PATCH_KEYS: [&str; 5] = [
    "id",
    "project_id",
    "created_at",
    "memory_type",
    "updated_at",
];

/// Split a caller's patch against one memory's serialized field set: keys the
/// model actually carries (minus protected keys) go into `applied`; every
/// other key lands in `ignored` with a human-readable reason. Derived from
/// the row itself, so a new model field is patchable the moment it exists —
/// the schema anti-drift test keeps tools/list honest about it.
fn partition_patch(
    memory_type: &str,
    existing: &serde_json::Value,
    patch: &serde_json::Map<String, serde_json::Value>,
) -> Result<(
    serde_json::Map<String, serde_json::Value>,
    serde_json::Map<String, serde_json::Value>,
)> {
    let empty = serde_json::Map::new();
    let model = existing.as_object().unwrap_or(&empty);
    let mut applied = serde_json::Map::new();
    let mut ignored = serde_json::Map::new();
    for (k, v) in patch {
        if PROTECTED_PATCH_KEYS.contains(&k.as_str()) {
            ignored.insert(
                k.clone(),
                serde_json::json!("protected field, never changed"),
            );
        } else if model.contains_key(k) {
            applied.insert(k.clone(), v.clone());
        } else {
            ignored.insert(
                k.clone(),
                serde_json::json!(format!("not a field of memory_type '{memory_type}'")),
            );
        }
    }
    Ok((applied, ignored))
}

/// Comma-joined, sorted list of the fields a patch may set on one memory
/// (serialized keys minus protected ones) — used in the "no applicable
/// fields" error so the message tracks the model instead of drifting.
fn applicable_field_list(existing: &serde_json::Value) -> String {
    let mut fields: Vec<&str> = existing
        .as_object()
        .map(|o| o.keys().map(String::as_str).collect())
        .unwrap_or_default();
    fields.retain(|k| !PROTECTED_PATCH_KEYS.contains(k));
    fields.sort_unstable();
    fields.join(", ")
}

/// Merge a patch object into an existing memory: overwrite only keys that already
/// exist on the model (ignores unknown keys), never touch the protected keys,
/// and stamp updated_at. Returns the rebuilt typed value. `update_memory`
/// pre-filters its patch through `partition_patch`, so the protections here
/// are defense in depth.
fn merge_patch<T>(
    existing: &T,
    patch: &serde_json::Map<String, serde_json::Value>,
    now: i64,
) -> Result<T>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let mut obj = match serde_json::to_value(existing)? {
        serde_json::Value::Object(m) => m,
        _ => anyhow::bail!("memory did not serialize to an object"),
    };
    for (k, v) in patch {
        if PROTECTED_PATCH_KEYS.contains(&k.as_str()) {
            continue;
        }
        if obj.contains_key(k) {
            obj.insert(k.clone(), v.clone());
        }
    }
    obj.insert("updated_at".into(), serde_json::json!(now));
    Ok(serde_json::from_value(serde_json::Value::Object(obj))?)
}

/// First sentence (or the first ~200 chars) of a longer text — brief entries
/// stay one line each; full payloads live in the structured sections.
fn first_sentence(text: &str) -> String {
    let t = text.trim();
    if let Some(i) = t.find(". ").filter(|i| *i <= 200) {
        return t[..i + 1].trim_end().to_string();
    }
    if t.chars().count() <= 200 {
        return t.to_string();
    }
    let cut = t.char_indices().nth(197).map_or(200, |(i, _)| i);
    format!("{}…", &t[..cut])
}

/// Parse the section count out of a session-import content template header.
/// The builder writes `Files touched (N):` and `Errors encountered (N —
/// failure-memory candidates):` (see `cli::commands::session_import_content`,
/// which stitches from the same `consolidation::engine` constants this is
/// called with) — the digits right after the opening paren are the count.
/// `None` = section absent (nothing of that kind was recorded) or malformed.
fn parse_session_import_count(content: &str, header: &str) -> Option<i64> {
    let rest = content.split_once(header)?.1.trim_start();
    let after_paren = rest.strip_prefix('(')?;
    let digits: String = after_paren
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Resolve an optional memory_type into the kinds to operate on.
fn resolve_kinds(memory_type: &Option<String>) -> Result<Vec<MemoryKind>> {
    match memory_type {
        Some(s) => Ok(vec![MemoryKind::from_type_str(s)?]),
        None => Ok(MemoryKind::all().to_vec()),
    }
}

/// The bootstrap prompt template, kept in sync with `docs/bootstrap.md` so
/// humans and agents read the same guidance (single source of truth).
const BOOTSTRAP_PROMPT_TEMPLATE: &str = include_str!("../../docs/bootstrap.md");
const DISTILL_PROMPT_NAME: &str = "engram.distill";
const DISTILL_PROMPT_TEMPLATE: &str = include_str!("../../docs/distill.md");

/// Prompt templates this server exposes via `prompts/list` + `prompts/get`.
/// Keeping the registry here lets `initialize` advertise the capability and
/// `prompts/list` enumerate from one place.
const BOOTSTRAP_PROMPT_NAME: &str = "engram.bootstrap";

/// Render the bootstrap prompt with the caller's arguments substituted in.
/// Unknown arguments fall back to placeholders so the agent still sees guidance.
fn render_bootstrap_prompt(project_id: &str, repo_path: &str, dimensions: &str) -> String {
    BOOTSTRAP_PROMPT_TEMPLATE
        .replace("{{PROJECT_ID}}", project_id)
        .replace("{{REPO_PATH}}", repo_path)
        .replace("{{DIMENSIONS}}", dimensions)
}

/// MCP Server running on stdio transport.
pub struct McpServer {
    provider: Arc<dyn MemoryToolProvider>,
    worker_threads: usize,
}

impl Default for McpServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate `NoopProvider` stub methods returning fixed placeholder JSON.
///
/// Each entry is `name, InputType, { json object }`; entries are `;`-separated
/// because the JSON object literal contains commas. The `{...}` is captured as
/// a single token tree and forwarded verbatim to `serde_json::json!`.
macro_rules! noop_stubs {
    ($($name:ident, $input:ty, $json:tt);* $(;)?) => {
        $(
            fn $name(&self, _: $input) -> Result<serde_json::Value> {
                Ok(serde_json::json!($json))
            }
        )*
    };
}
impl McpServer {
    pub fn new() -> Self {
        struct NoopProvider;
        impl MemoryToolProvider for NoopProvider {
            noop_stubs! {
                search_memory, SearchMemoryInput, {"results": [], "total": 0};
                get_memory, GetMemoryInput, {"memory": null};
                related_files, RelatedFilesInput, {"entities": []};
                timeline, TimelineInput, {"project_id": "", "days": 7, "events": []};
                recent_failures, RecentFailuresInput, {"failures": []};
                query_stats, QueryStatsInput, {"queries": []};
                session_brief, SessionBriefInput, {"brief": "", "decisions": [], "failures": [], "procedures": [], "pending_suggestions": []};
                architectural_decisions, ArchitecturalDecisionsInput, {"decisions": []};
                distill_backlog, DistillBacklogInput, {"total": 0, "items": []};
                create_episodic, CreateEpisodicInput, {"id": "", "status": "noop"};
                create_decision, CreateDecisionInput, {"id": "", "status": "noop"};
                create_failure, CreateFailureInput, {"id": "", "status": "noop"};
                create_procedural, CreateProceduralInput, {"id": "", "status": "noop"};
                ingest_commits, IngestCommitsInput, {"ingested": 0};
                collect_sources, CollectSourcesInput, {"summary": {"total_items": 0}};
                forget_memory, ForgetMemoryInput, {"archived": false, "status": "noop"};
                restore_memory, RestoreMemoryInput, {"restored": false, "status": "noop"};
                update_memory, UpdateMemoryInput, {"id": "", "status": "noop"};
                mark_relevance, MarkRelevanceInput, {"id": "", "status": "noop"};
                forget_batch, ForgetBatchInput, {"applied": false, "matched": [], "count": 0};
                list_archived, ListArchivedInput, {"archived": [], "count": 0};
                consolidate_memories, ConsolidateInput, {"applied": false, "plans": [], "total_archived": 0};
                reflect, ReflectInput, {"applied": false, "proposed": 0, "created": 0, "suggestions": []};
                list_suggestions, ListSuggestionsInput, {"count": 0, "suggestions": []};
                confirm_suggestion, SuggestionIdInput, {"id": "", "status": "noop"};
                reject_suggestion, SuggestionIdInput, {"id": "", "status": "noop"};
            }
        }
        Self {
            provider: Arc::new(NoopProvider),
            worker_threads: crate::config::McpConfig::default().worker_threads,
        }
    }

    pub fn with_provider(provider: Arc<dyn MemoryToolProvider>) -> Self {
        Self::with_provider_and_workers(
            provider,
            crate::config::McpConfig::default().worker_threads,
        )
    }

    /// Build a server with a specific worker-thread count for the request loop.
    pub fn with_provider_and_workers(
        provider: Arc<dyn MemoryToolProvider>,
        worker_threads: usize,
    ) -> Self {
        Self {
            provider,
            worker_threads: worker_threads.max(1),
        }
    }

    /// Get the list of available MCP tools.
    pub fn list_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "search_memory".into(),
                description: "Search all memory types via BM25 full-text search. Results carry each memory's full payload (e.g. a failure's root_cause/fix/prevention) — no follow-up fetch needed. Ranking blends relevance, recency, importance, and type priors; intent keywords adjust weights but never exclude types.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier; '*' searches across all projects (global knowledge)" },
                        "query": { "type": "string", "description": "Search query" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"], "description": "Filter by memory type" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Only memories carrying any of these tags" },
                        "before": { "type": "integer", "description": "Only memories created before this unix timestamp" },
                        "limit": { "type": "integer", "description": "Max results (default: retrieval.default_limit)" },
                    },
                    "required": ["project_id", "query"],
                }),
            },
            ToolDefinition {
                name: "get_memory".into(),
                description: "Fetch one memory's complete record by id and type — every stored field, including archived-relevant metadata and full structured content.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"] },
                        "id": { "type": "string", "description": "Memory id (from search_memory results)" },
                    },
                    "required": ["project_id", "memory_type", "id"],
                }),
            },
            ToolDefinition {
                name: "related_files".into(),
                description: "Find memories that touch a specific file (via the relationship graph). Each result carries memory_type, id, and a summary — usable directly with get_memory. Run before modifying a file to surface prior work, decisions, and failures involving it.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "file": { "type": "string", "description": "File name/path to find related memories for" },
                    },
                    "required": ["project_id", "file"],
                }),
            },
            ToolDefinition {
                name: "timeline".into(),
                description: "Get a timeline of memory activity for a project: per-UTC-day counts across all four memory types (episodic, decision, failure, procedural) over the past N days. Archived memories are excluded. Optional memory_type counts a single type (the other per-type counts stay 0); optional until bounds the window's end with an absolute unix timestamp.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "days": { "type": "integer", "description": "Number of days to look back", "default": 7 },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"], "description": "Only count this memory type; the other per-type counts stay 0" },
                        "until": { "type": "integer", "description": "Window end as an absolute unix timestamp (exclusive); omitted means now" },
                    },
                    "required": ["project_id"],
                    "additionalProperties": false,
                }),
            },
            ToolDefinition {
                name: "recent_failures".into(),
                description: "Get recent failure memories for a project".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "service": { "type": "string", "description": "Optional full-text filter matched against failure text (incident, root_cause, fix, prevention) — NOT a structured 'service' field" },
                        "limit": { "type": "integer", "description": "Max results", "default": 10 },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "query_stats".into(),
                description: "Aggregate past search queries by frequency and average hit count — surfaces which queries are common and which return few results (retrieval feedback)".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "days": { "type": "integer", "description": "Number of days to look back", "default": 7 },
                        "limit": { "type": "integer", "description": "Max distinct queries", "default": 10 },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "session_brief".into(),
                description: "One-call context assembly for session start: the project's key decisions, recent failures, active procedures, and pending reflection suggestions — rendered as a compact text brief plus structured sections, within the memory token budget. Call this at the start of a session instead of guessing which searches to run; pass `focus` to also include top search hits for the task at hand.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "focus": { "type": "string", "description": "Optional task/topic: also returns the top search hits for it" },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "architectural_decisions".into(),
                description: "Get architectural decision memories for a project. Current decisions by default — superseded (replaced) decisions only appear with include_superseded=true, marked with superseded_by.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "topic": { "type": "string", "description": "Filter by topic keyword" },
                        "limit": { "type": "integer", "description": "Max results", "default": 10 },
                        "include_superseded": { "type": "boolean", "description": "Also include decisions replaced by a newer one (default false)", "default": false },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "distill_backlog".into(),
                description: "List the raw session-import memories awaiting distillation, ordered oldest first with a per-item signal overview (age, error/file counts, content preview). This is the material-picking entry point of the engram.distill prompt: take items from the top (longest rotting), get_memory each for the full transcript dump, distill into typed memories, then forget the raw ones. Distilled memories (tagged 'distilled') never appear here — the total reaching 0 means the loop is closed.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "limit": { "type": "integer", "description": "Max entries returned (default 10, clamped 1..=50)" },
                    },
                    "required": ["project_id"],
                    "additionalProperties": false,
                }),
            },
            // ─── Write Tools ─────────────────────────────────────────
            ToolDefinition {
                name: "create_episodic".into(),
                description: "Create an episodic memory recording a task, debugging session, refactor, migration, deployment, or feature implementation".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "session_id": { "type": "string", "description": "Session identifier" },
                        "summary": { "type": "string", "description": "Brief summary of what happened" },
                        "content": { "type": "string", "description": "Detailed description" },
                        "files_touched": { "type": "array", "items": { "type": "string" }, "description": "Files affected" },
                        "related_commits": { "type": "array", "items": { "type": "string" }, "description": "Related commit hashes" },
                        "importance": { "type": "number", "description": "Importance score 0-1 (default 0.5)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags for categorization" },
                    },
                    "required": ["project_id", "session_id", "summary", "content"],
                }),
            },
            ToolDefinition {
                name: "create_decision".into(),
                description: "Record an architectural or design decision with rationale and tradeoffs. When this decision REPLACES an earlier one, pass that decision's id as `supersedes` — the old record stays searchable as history but is flagged and downranked.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "title": { "type": "string", "description": "Decision title" },
                        "context": { "type": "string", "description": "Context and background" },
                        "rationale": { "type": "string", "description": "Why this decision was made" },
                        "tradeoffs": { "type": "string", "description": "Tradeoffs considered" },
                        "related_files": { "type": "array", "items": { "type": "string" }, "description": "Related files" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags for categorization" },
                        "importance": { "type": "number", "description": "Ranking signal 0-1 (default 0.5)" },
                        "supersedes": { "type": "string", "description": "Id of the decision this one replaces (starts a supersede chain)" },
                    },
                    "required": ["project_id", "title", "context", "rationale", "tradeoffs"],
                }),
            },
            ToolDefinition {
                name: "create_failure".into(),
                description: "Record a failure, incident, or bug with root cause analysis and prevention measures".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "incident": { "type": "string", "description": "What happened" },
                        "root_cause": { "type": "string", "description": "Root cause analysis" },
                        "fix": { "type": "string", "description": "How it was fixed" },
                        "prevention": { "type": "string", "description": "How to prevent recurrence" },
                        "severity": { "type": "integer", "description": "Severity 1-5 (default 3)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags for categorization" },
                    },
                    "required": ["project_id", "incident", "root_cause", "fix", "prevention"],
                }),
            },
            ToolDefinition {
                name: "create_procedural".into(),
                description: "Record a workflow, process, or coding convention with step-by-step instructions".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "workflow_name": { "type": "string", "description": "Name of the workflow" },
                        "steps": { "type": "array", "items": { "type": "string" }, "description": "Step-by-step instructions" },
                        "related_tools": { "type": "array", "items": { "type": "string" }, "description": "Tools used in this workflow" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags for categorization" },
                        "importance": { "type": "number", "minimum": 0.0, "maximum": 1.0, "description": "Ranking signal (default 0.5; core conventions ≥0.7)" },
                    },
                    "required": ["project_id", "workflow_name", "steps"],
                }),
            },
            ToolDefinition {
                name: "ingest_commits".into(),
                description: "Auto-generate episodic memories from recent git commits in a repository".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "repo_path": { "type": "string", "description": "Path to the git repository" },
                        "count": { "type": "integer", "description": "Number of recent commits to ingest (default 20)" },
                        "session_id": { "type": "string", "description": "Optional session identifier" },
                    },
                    "required": ["project_id", "repo_path"],
                }),
            },
            ToolDefinition {
                name: "collect_sources".into(),
                description: "Gather structured evidence (git themes, docs, decisions, failures, workflows) from an existing project for memory bootstrap. Returns raw material only — does not write memories. Pair with the `engram.bootstrap` prompt.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "repo_path": { "type": "string", "description": "Path to the project root (git optional)" },
                        "dimensions": { "type": "string", "description": "Comma-separated: git,decisions,failures,workflow (default: all)" },
                        "max_commits": { "type": "integer", "description": "Max commits to walk for the git dimension (default 200)" },
                    },
                    "required": ["project_id", "repo_path"],
                }),
            },
            ToolDefinition {
                name: "forget_memory".into(),
                description: "Soft-delete (archive) a memory by id. Reversible via restore_memory.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"] },
                        "id": { "type": "string", "description": "Memory id to archive" },
                    },
                    "required": ["project_id", "memory_type", "id"],
                }),
            },
            ToolDefinition {
                name: "restore_memory".into(),
                description: "Un-archive a previously forgotten memory by id.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"] },
                        "id": { "type": "string", "description": "Memory id to restore" },
                    },
                    "required": ["project_id", "memory_type", "id"],
                }),
            },
            ToolDefinition {
                name: "update_memory".into(),
                description: "Patch fields of an existing memory (correction). Pass ONLY the fields to change (plus project_id/memory_type/id) — omitted fields keep their current value. Fields that do not apply to the target memory_type are ignored and reported in the response's `ignored` map with a reason; `applied` echoes every field that was actually written, so no follow-up get_memory is needed to verify. id/project_id/created_at/updated_at are server-managed and never changed. An empty patch — or one whose keys are all protected or belong to another memory_type — is an error, never a silent success. Patchable fields by type — episodic: session_id, summary, content, files_touched, related_commits; decision: title, context, rationale, tradeoffs, related_files, superseded_by; failure: incident, root_cause, fix, prevention, severity; procedural: workflow_name, steps, related_tools; importance (0-1) applies to episodic/decision/procedural (failure has no importance — severity is its ranking signal); tags applies to all types.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "description": "Only the fields to change need to be passed. Fields that do not apply to the target memory_type are ignored and reported in the response's `ignored` map; unknown keys are rejected by this schema.",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"] },
                        "id": { "type": "string", "description": "Memory id to update" },
                        "session_id": { "type": "string", "description": "episodic only" },
                        "summary": { "type": "string", "description": "episodic only" },
                        "content": { "type": "string", "description": "episodic only" },
                        "files_touched": { "type": "array", "items": { "type": "string" }, "description": "episodic only" },
                        "related_commits": { "type": "array", "items": { "type": "string" }, "description": "episodic only" },
                        "title": { "type": "string", "description": "decision only" },
                        "context": { "type": "string", "description": "decision only" },
                        "rationale": { "type": "string", "description": "decision only" },
                        "tradeoffs": { "type": "string", "description": "decision only" },
                        "related_files": { "type": "array", "items": { "type": "string" }, "description": "decision only" },
                        "superseded_by": { "type": ["string", "null"], "description": "decision only — id of the decision that replaced this one; normally set via create_decision's `supersedes`, patch only to repair a broken chain" },
                        "incident": { "type": "string", "description": "failure only" },
                        "root_cause": { "type": "string", "description": "failure only" },
                        "fix": { "type": "string", "description": "failure only" },
                        "prevention": { "type": "string", "description": "failure only" },
                        "severity": { "type": "integer", "minimum": 1, "maximum": 5, "description": "failure only" },
                        "workflow_name": { "type": "string", "description": "procedural only" },
                        "steps": { "type": "array", "items": { "type": "string" }, "description": "procedural only" },
                        "related_tools": { "type": "array", "items": { "type": "string" }, "description": "procedural only" },
                        "importance": { "type": "number", "minimum": 0, "maximum": 1, "description": "episodic/decision/procedural only — failure has no importance (severity is its ranking signal)" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "all types" },
                    },
                    "required": ["project_id", "memory_type", "id"],
                    "additionalProperties": false,
                }),
            },
            ToolDefinition {
                name: "mark_relevance".into(),
                description: "Give feedback on a search result: 'useful' raises its ranking importance (+0.1, also counted as adopted), 'irrelevant' lowers it (-0.1). The normal flow IS search → get_memory → mark_relevance — reading a result does NOT cool down feedback. One feedback per memory per 24h (either direction): a repeat inside the window returns status \"cooldown\" and changes nothing. Call this after reading a result that helped (or didn't) — it tunes future rankings.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"] },
                        "id": { "type": "string", "description": "Memory id the feedback applies to" },
                        "feedback": { "type": "string", "enum": ["useful", "irrelevant"], "description": "useful = +0.1 importance, irrelevant = -0.1" },
                    },
                    "required": ["project_id", "memory_type", "id", "feedback"],
                }),
            },
            ToolDefinition {
                name: "forget_batch".into(),
                description: "Archive memories matching tags and/or a before-timestamp. Dry-run by default (apply=true to archive).".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"], "description": "Omit for all types" },
                        "tags": { "type": "array", "items": { "type": "string" }, "description": "Archive memories carrying ANY of these tags" },
                        "before": { "type": "integer", "description": "Archive memories created before this unix timestamp" },
                        "apply": { "type": "boolean", "description": "false (default) = dry-run; true = archive", "default": false },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "list_archived".into(),
                description: "List archived (soft-deleted) memories for audit or restore.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"], "description": "Omit for all types" },
                        "limit": { "type": "integer", "description": "Max results per type", "default": 10 },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "consolidate_memories".into(),
                description: "Find duplicate memories (exact by default; include_near_dup for fuzzy). Dry-run by default; apply=true archives duplicates keeping the earliest.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"], "description": "Omit for all types" },
                        "include_near_dup": { "type": "boolean", "description": "Also group near-duplicates (Jaccard)", "default": false },
                        "apply": { "type": "boolean", "description": "false (default) = report only; true = archive duplicates", "default": false },
                    },
                    "required": ["project_id"],
                }),
            },
            // ─── Reflection Tools ───────────────────────────────────
            ToolDefinition {
                name: "reflect".into(),
                description: "Scan active failures for recurring tags and propose preventive procedural rules. Dry-run by default (apply=false); persisted proposals stay invisible to search until confirmed.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "apply": { "type": "boolean", "description": "false (default) = preview only; true = persist proposals as pending", "default": false },
                        "min_occurrences": { "type": "integer", "description": "Override [reflection].min_occurrences threshold" },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "list_suggestions".into(),
                description: "List pending reflection proposals (draft preventive rules distilled from recurring failures) awaiting confirmation.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "confirm_suggestion".into(),
                description: "Confirm a pending reflection proposal: promote it into a searchable procedural memory.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "id": { "type": "string", "description": "Suggestion id (from list_suggestions)" },
                    },
                    "required": ["project_id", "id"],
                }),
            },
            ToolDefinition {
                name: "reject_suggestion".into(),
                description: "Reject a pending reflection proposal: discard it without creating a procedural memory.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "id": { "type": "string", "description": "Suggestion id (from list_suggestions)" },
                    },
                    "required": ["project_id", "id"],
                }),
            },
        ]
    }

    /// Run the MCP server over the stdio JSON-RPC transport.
    pub fn run(self: Arc<Self>) -> Result<()> {
        crate::mcp::transport::run_stdio(Arc::clone(&self), self.worker_threads)
    }

    fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => {
                // Protocol version negotiation (MCP spec): echo the client's
                // requested version when we support it; otherwise reply with
                // our latest supported version. The old unconditional
                // hardcoded reply never honored the client's request.
                let requested = request
                    .params
                    .as_ref()
                    .and_then(|p| p.get("protocolVersion"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                const SUPPORTED_PROTOCOL_VERSIONS: &[&str] = &["2024-11-05"];
                let negotiated = if SUPPORTED_PROTOCOL_VERSIONS.contains(&requested) {
                    requested.to_string()
                } else {
                    SUPPORTED_PROTOCOL_VERSIONS[0].to_string()
                };
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: request.id,
                    result: Some(serde_json::json!({
                        "protocolVersion": negotiated,
                        "capabilities": { "tools": {}, "prompts": {} },
                        "serverInfo": {
                            "name": "engram",
                            // Keep in lockstep with Cargo.toml — a stale hardcoded
                            // version here made clients see 0.1.0 in 0.2.0 builds.
                            "version": env!("CARGO_PKG_VERSION"),
                        },
                    })),
                    error: None,
                }
            }
            // MCP liveness probe: clients may ping to check connection health;
            // answering with an empty result is required by the spec.
            "ping" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result: Some(serde_json::json!({})),
                error: None,
            },
            "tools/list" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result: Some(serde_json::json!({
                    "tools": Self::list_tools(),
                })),
                error: None,
            },
            "prompts/list" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result: Some(serde_json::json!({
                    "prompts": [{
                        "name": BOOTSTRAP_PROMPT_NAME,
                        "description": "Initialize engram memory for an existing project: collect evidence via collect_sources, then distill it into structured memories with quality bars.",
                        "arguments": [
                            { "name": "project_id", "description": "Project identifier (required)", "required": true },
                            { "name": "repo_path", "description": "Path to the project root (required)", "required": true },
                            { "name": "dimensions", "description": "Comma-separated: git,decisions,failures,workflow (default: all)", "required": false },
                        ],
                    }, {
                        "name": DISTILL_PROMPT_NAME,
                        "description": "Refine raw session-import memories (auto-written by `engram hook`) into clean typed memories: decisions, failures (with root cause), procedures. Run periodically or after several sessions.",
                        "arguments": [
                            { "name": "project_id", "description": "Project identifier (required)", "required": true },
                        ],
                    }],
                })),
                error: None,
            },
            "prompts/get" => {
                let params = request.params.unwrap_or_default();
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if name.is_empty() {
                    return JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: request.id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32602,
                            message: "Invalid params: prompts/get requires a string 'name'".into(),
                            data: None,
                        }),
                    };
                }
                if name == DISTILL_PROMPT_NAME {
                    let args = params.get("arguments").cloned().unwrap_or_default();
                    let Some(project_id) = args.get("project_id").and_then(|v| v.as_str()) else {
                        return JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: request.id,
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32602,
                                message: "Invalid params: engram.distill requires a string \
                                          argument 'project_id'"
                                    .into(),
                                data: None,
                            }),
                        };
                    };
                    let rendered = DISTILL_PROMPT_TEMPLATE.replace("{{PROJECT_ID}}", project_id);
                    JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: request.id,
                        result: Some(serde_json::json!({
                            "description": "Refine raw session-import memories into typed memories",
                            "messages": [{
                                "role": "user",
                                "content": { "type": "text", "text": rendered }
                            }],
                        })),
                        error: None,
                    }
                } else if name != BOOTSTRAP_PROMPT_NAME {
                    JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: request.id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32601,
                            message: format!("Unknown prompt: {name}"),
                            data: None,
                        }),
                    }
                } else {
                    let args = params.get("arguments").cloned().unwrap_or_default();
                    // prompts/list declares project_id/repo_path as required —
                    // enforce it here instead of silently rendering "<project_id>"
                    // placeholders into the prompt the agent will follow.
                    let (Some(project_id), Some(repo_path)) = (
                        args.get("project_id").and_then(|v| v.as_str()),
                        args.get("repo_path").and_then(|v| v.as_str()),
                    ) else {
                        return JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: request.id,
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32602,
                                message: "Invalid params: engram.bootstrap requires string \
                                          arguments 'project_id' and 'repo_path'"
                                    .into(),
                                data: None,
                            }),
                        };
                    };
                    let dimensions = args
                        .get("dimensions")
                        .and_then(|v| v.as_str())
                        .unwrap_or("git, decisions, failures, workflow");
                    let rendered = render_bootstrap_prompt(project_id, repo_path, dimensions);
                    JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: request.id,
                        result: Some(serde_json::json!({
                            "description": "Initialize engram memory for an existing project",
                            "messages": [{
                                "role": "user",
                                "content": { "type": "text", "text": rendered }
                            }],
                        })),
                        error: None,
                    }
                }
            }
            "tools/call" => {
                let params = request.params.unwrap_or_default();
                // A missing/non-string `name` is a malformed tools/call —
                // report it as invalid params instead of "unknown tool ''".
                let tool_name = match params.get("name").and_then(|v| v.as_str()) {
                    Some(n) if !n.is_empty() => n,
                    _ => {
                        return JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: request.id,
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32602,
                                message: "Invalid params: tools/call requires a string 'name'"
                                    .into(),
                                data: None,
                            }),
                        };
                    }
                };
                let arguments = params.get("arguments").cloned().unwrap_or_default();

                let (result, parse_error): (
                    Option<Result<serde_json::Value>>,
                    Option<JsonRpcError>,
                ) = match tool_name {
                    "search_memory" => {
                        dispatch_tool!(arguments, SearchMemoryInput, self.provider, search_memory)
                    }
                    "get_memory" => {
                        dispatch_tool!(arguments, GetMemoryInput, self.provider, get_memory)
                    }
                    "related_files" => {
                        dispatch_tool!(arguments, RelatedFilesInput, self.provider, related_files)
                    }
                    "timeline" => dispatch_tool!(arguments, TimelineInput, self.provider, timeline),
                    "recent_failures" => dispatch_tool!(
                        arguments,
                        RecentFailuresInput,
                        self.provider,
                        recent_failures
                    ),
                    "query_stats" => {
                        dispatch_tool!(arguments, QueryStatsInput, self.provider, query_stats)
                    }
                    "session_brief" => {
                        dispatch_tool!(arguments, SessionBriefInput, self.provider, session_brief)
                    }
                    "architectural_decisions" => dispatch_tool!(
                        arguments,
                        ArchitecturalDecisionsInput,
                        self.provider,
                        architectural_decisions
                    ),
                    "distill_backlog" => {
                        dispatch_tool!(
                            arguments,
                            DistillBacklogInput,
                            self.provider,
                            distill_backlog
                        )
                    }
                    "create_episodic" => dispatch_tool!(
                        arguments,
                        CreateEpisodicInput,
                        self.provider,
                        create_episodic
                    ),
                    "create_decision" => dispatch_tool!(
                        arguments,
                        CreateDecisionInput,
                        self.provider,
                        create_decision
                    ),
                    "create_failure" => {
                        dispatch_tool!(arguments, CreateFailureInput, self.provider, create_failure)
                    }
                    "create_procedural" => dispatch_tool!(
                        arguments,
                        CreateProceduralInput,
                        self.provider,
                        create_procedural
                    ),
                    "ingest_commits" => {
                        dispatch_tool!(arguments, IngestCommitsInput, self.provider, ingest_commits)
                    }
                    "collect_sources" => dispatch_tool!(
                        arguments,
                        CollectSourcesInput,
                        self.provider,
                        collect_sources
                    ),
                    "forget_memory" => {
                        dispatch_tool!(arguments, ForgetMemoryInput, self.provider, forget_memory)
                    }
                    "restore_memory" => {
                        dispatch_tool!(arguments, RestoreMemoryInput, self.provider, restore_memory)
                    }
                    "update_memory" => {
                        dispatch_tool!(arguments, UpdateMemoryInput, self.provider, update_memory)
                    }
                    "mark_relevance" => {
                        dispatch_tool!(arguments, MarkRelevanceInput, self.provider, mark_relevance)
                    }
                    "forget_batch" => {
                        dispatch_tool!(arguments, ForgetBatchInput, self.provider, forget_batch)
                    }
                    "list_archived" => {
                        dispatch_tool!(arguments, ListArchivedInput, self.provider, list_archived)
                    }
                    "consolidate_memories" => dispatch_tool!(
                        arguments,
                        ConsolidateInput,
                        self.provider,
                        consolidate_memories
                    ),
                    "reflect" => {
                        dispatch_tool!(arguments, ReflectInput, self.provider, reflect)
                    }
                    "list_suggestions" => dispatch_tool!(
                        arguments,
                        ListSuggestionsInput,
                        self.provider,
                        list_suggestions
                    ),
                    "confirm_suggestion" => dispatch_tool!(
                        arguments,
                        SuggestionIdInput,
                        self.provider,
                        confirm_suggestion
                    ),
                    "reject_suggestion" => dispatch_tool!(
                        arguments,
                        SuggestionIdInput,
                        self.provider,
                        reject_suggestion
                    ),
                    _ => (
                        None,
                        Some(JsonRpcError {
                            // MCP spec: an unknown tool name on tools/call is a
                            // parameter problem of THIS method, not a missing
                            // method (-32601 was wrong — the method exists).
                            code: -32602,
                            message: format!(
                                "Invalid params: unknown tool {tool_name:?}; see tools/list"
                            ),
                            data: None,
                        }),
                    ),
                };

                if let Some(err) = parse_error {
                    JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: request.id,
                        result: None,
                        error: Some(err),
                    }
                } else {
                    match result.expect("result exists when no parse error") {
                        Ok(value) => JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: request.id,
                            result: Some(serde_json::json!({
                                "content": [{ "type": "text", "text": value.to_string() }]
                            })),
                            error: None,
                        },
                        // Tool business failures (memory not found, bad
                        // severity, …) are NOT protocol errors: the MCP spec
                        // requires `result.isError = true` so clients can show
                        // the message to the agent, which can then fix its
                        // arguments. JSON-RPC errors stay reserved for
                        // protocol/transport failures and panics.
                        Err(e) => JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: request.id,
                            result: Some(serde_json::json!({
                                "content": [{ "type": "text", "text": format!("error: {e}") }],
                                "isError": true,
                            })),
                            error: None,
                        },
                    }
                }
            }
            _ => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32601,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            },
        }
    }
}

impl RequestHandler for McpServer {
    fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        self.handle_request(req)
    }

    fn fast_path_methods(&self) -> &'static [&'static str] {
        // Static, read-only responses that never touch the repository — safe
        // to answer inline (concurrently with a busy worker) so client health
        // probes stay responsive during long tool calls like ingest/reindex.
        &["ping", "tools/list", "prompts/list", "initialize"]
    }
}

/// Test-only repo accessor (the field is private; tests build their own
/// provider via `make_provider` and need to seed memories).
#[cfg(test)]
impl DefaultMemoryProvider {
    pub fn repo_for_test(&self) -> &MemoryRepository {
        &self.repo
    }
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod server_tests;
