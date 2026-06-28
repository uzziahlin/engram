use crate::collectors;
use crate::config::Config;
use crate::consolidation::ConsolidationEngine;
use crate::context::composer::{ContextBudget, ContextComposer};
use crate::git_integration::GitIntegration;
use crate::models::*;
use crate::retrieval::bm25::BM25Retriever;
#[cfg(feature = "semantic")]
use crate::retrieval::bm25::SearchResult;
use crate::retrieval::intent_classifier::IntentClassifier;
use crate::retrieval::planner::RetrievalPlanner;
use crate::retrieval::reranker::Reranker;
use crate::storage::MemoryKind;
use crate::storage::MemoryRepository;
use crate::storage::ScoredMemory;

use anyhow::{Context as AnyhowContext, Result};
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};

/// Jaccard similarity threshold for near-duplicate consolidation.
const CONSOLIDATE_JACCARD_THRESHOLD: f64 = 0.85;

/// JSON-RPC request for MCP protocol.
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    params: Option<serde_json::Value>,
}

/// JSON-RPC response for MCP protocol.
#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<serde_json::Value>,
}

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

/// search_memory tool input.
#[derive(Debug, Deserialize)]
pub struct SearchMemoryInput {
    project_id: String,
    query: String,
    #[serde(default)]
    memory_type: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    10
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
    #[serde(default = "default_limit")]
    limit: usize,
}

/// architectural_decisions tool input.
#[derive(Debug, Deserialize)]
pub struct ArchitecturalDecisionsInput {
    project_id: String,
    #[serde(default)]
    topic: Option<String>,
    #[serde(default = "default_limit")]
    limit: usize,
}

/// query_stats tool input — retrieval feedback aggregated by query string.
#[derive(Debug, Deserialize)]
pub struct QueryStatsInput {
    project_id: String,
    #[serde(default = "default_days")]
    days: i64,
    #[serde(default = "default_limit")]
    limit: usize,
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
#[derive(Debug, Deserialize)]
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
    #[serde(default = "default_limit")]
    pub limit: usize,
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
    fn related_files(&self, input: RelatedFilesInput) -> Result<serde_json::Value>;
    fn timeline(&self, input: TimelineInput) -> Result<serde_json::Value>;
    fn recent_failures(&self, input: RecentFailuresInput) -> Result<serde_json::Value>;
    fn query_stats(&self, input: QueryStatsInput) -> Result<serde_json::Value>;
    fn architectural_decisions(
        &self,
        input: ArchitecturalDecisionsInput,
    ) -> Result<serde_json::Value>;

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
    fn forget_batch(&self, input: ForgetBatchInput) -> Result<serde_json::Value>;
    fn list_archived(&self, input: ListArchivedInput) -> Result<serde_json::Value>;
    fn consolidate_memories(&self, input: ConsolidateInput) -> Result<serde_json::Value>;
}

/// Result of a `reindex_embeddings` run; serialized as the CLI's JSON output.
#[cfg(feature = "semantic")]
#[derive(Debug, Default, Serialize)]
pub struct ReindexReport {
    pub total: usize,
    pub embedded: usize,
    pub skipped: usize,
    pub failed: usize,
    pub dry_run: bool,
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
    /// Local embedding model, present only when semantic search is enabled and
    /// the model loaded successfully. `None` ⇒ pure-BM25 path (unchanged).
    #[cfg(feature = "semantic")]
    embedder: Option<Box<dyn crate::retrieval::embedding::EmbeddingProvider>>,
}

impl DefaultMemoryProvider {
    pub fn new(repo: MemoryRepository, config: Config) -> Self {
        #[cfg(feature = "semantic")]
        let embedder = Self::init_embedder(&config);
        // Read the ranking weights before `config` is moved into the struct.
        let plan_weights = crate::retrieval::planner::PlanWeights {
            relevance: config.retrieval.weight_relevance,
            recency: config.retrieval.weight_recency,
            importance: config.retrieval.weight_importance,
            type_weight: config.retrieval.weight_type,
        };
        Self {
            repo,
            config,
            classifier: IntentClassifier::new(),
            planner: RetrievalPlanner::new(plan_weights),
            reranker: Reranker::new(),
            composer: ContextComposer::new(),
            #[cfg(feature = "semantic")]
            embedder,
        }
    }

    /// Borrow the repository. No lock needed: MemoryRepository holds an r2d2
    /// pool (Send+Sync); each method borrows its own pooled connection.
    fn lock_repo(&self) -> &MemoryRepository {
        &self.repo
    }

    /// Build the embedding model from config (semantic enabled + model available).
    /// Returns None (logging a warning) on any failure so the server still runs
    /// in pure-BM25 mode.
    #[cfg(feature = "semantic")]
    fn init_embedder(
        config: &Config,
    ) -> Option<Box<dyn crate::retrieval::embedding::EmbeddingProvider>> {
        use crate::retrieval::embedding::{ensure_model, CandleBertEmbedder};
        if !config.semantic.enabled {
            return None;
        }
        let dir = config.semantic.model_path.clone().unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_default()
                .join(".engram/models")
                .join(
                    config
                        .semantic
                        .model_id
                        .rsplit('/')
                        .next()
                        .unwrap_or("model"),
                )
        });
        // Only auto-fetch when no explicit path was given (air-gapped users
        // provide model_path and we never touch the network).
        if config.semantic.model_path.is_none() {
            if let Err(e) = ensure_model(&config.semantic.model_id, &dir) {
                tracing::warn!(
                    "semantic enabled but model fetch failed ({e}); disabling semantic search"
                );
                return None;
            }
        }
        match CandleBertEmbedder::from_local(&dir, &config.semantic.model_id) {
            Ok(e) => {
                tracing::info!(
                    "semantic search enabled: model {}",
                    config.semantic.model_id
                );
                Some(Box::new(e))
            }
            Err(e) => {
                tracing::warn!("semantic model load failed ({e}); disabling semantic search");
                None
            }
        }
    }

    /// Embed `text` and store the vector for `id`. No-op when the embedder is
    /// absent. Failures are logged, never fatal (embedding is an enhancement).
    #[cfg(feature = "semantic")]
    fn index_embedding(&self, memory_type: &str, id: &str, project_id: &str, text: &str) {
        if let Some(e) = self.embedder.as_ref() {
            match e.embed(&[text]) {
                Ok(v) if !v.is_empty() => {
                    if let Err(err) = self.repo.upsert_embedding(
                        id,
                        memory_type,
                        project_id,
                        &v[0],
                        e.model_id(),
                        e.dim(),
                    ) {
                        tracing::warn!("upsert_embedding failed for {id}: {err}");
                    }
                }
                Ok(_) => {}
                Err(err) => tracing::warn!("embed failed for {id}: {err}"),
            }
        }
    }

    /// Embed one memory during reindex, updating `report`. Skips when already
    /// embedded (unless `force`); counts only when `dry_run`. Per-memory errors
    /// are logged and counted as `failed`, never aborting the batch.
    #[cfg(feature = "semantic")]
    #[allow(clippy::too_many_arguments)]
    fn reindex_one(
        &self,
        report: &mut ReindexReport,
        already: &HashSet<String>,
        memory_type: &str,
        id: &str,
        project_id: &str,
        text: String,
        force: bool,
        dry_run: bool,
    ) {
        report.total += 1;
        if !force && already.contains(id) {
            report.skipped += 1;
            return;
        }
        if dry_run {
            report.embedded += 1; // would embed
            return;
        }
        let embedder = self
            .embedder
            .as_ref()
            .expect("embedder presence is checked by reindex_embeddings");
        match embedder.embed(&[text.as_str()]) {
            Ok(v) if !v.is_empty() => {
                match self.repo.upsert_embedding(
                    id,
                    memory_type,
                    project_id,
                    &v[0],
                    embedder.model_id(),
                    embedder.dim(),
                ) {
                    Ok(()) => report.embedded += 1,
                    Err(e) => {
                        tracing::warn!("reindex upsert failed for {id}: {e}");
                        report.failed += 1;
                    }
                }
            }
            Ok(_) => {
                tracing::warn!("reindex embed returned empty for {id}");
                report.failed += 1;
            }
            Err(e) => {
                tracing::warn!("reindex embed failed for {id}: {e}");
                report.failed += 1;
            }
        }
    }

    /// Backfill embeddings for active memories. `project=None` = all projects.
    /// Default skips memories already embedded for the current model; `force`
    /// re-embeds all. `dry_run` counts without writing. Errors when the embedder
    /// is absent (semantic disabled or model load failed).
    #[cfg(feature = "semantic")]
    pub fn reindex_embeddings(
        &self,
        project: Option<&str>,
        force: bool,
        dry_run: bool,
    ) -> Result<ReindexReport> {
        let embedder = self.embedder.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "semantic search is not active: enable [semantic] in ~/.engram/config.toml \
                 and ensure the model is available"
            )
        })?;
        let model_id = embedder.model_id().to_string();
        let already = if force {
            HashSet::new()
        } else {
            self.repo.embedded_ids(project, &model_id)?
        };

        let mut report = ReindexReport {
            dry_run,
            ..Default::default()
        };
        for m in self.repo.list_active_episodic(project)? {
            self.reindex_one(
                &mut report,
                &already,
                "episodic",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        for m in self.repo.list_active_decision(project)? {
            self.reindex_one(
                &mut report,
                &already,
                "decision",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        for m in self.repo.list_active_failure(project)? {
            self.reindex_one(
                &mut report,
                &already,
                "failure",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        for m in self.repo.list_active_procedural(project)? {
            self.reindex_one(
                &mut report,
                &already,
                "procedural",
                &m.id,
                &m.project_id,
                m.embedding_text(),
                force,
                dry_run,
            );
        }
        tracing::info!(
            "reindex: total={} embedded={} skipped={} failed={} dry_run={}",
            report.total,
            report.embedded,
            report.skipped,
            report.failed,
            report.dry_run
        );
        Ok(report)
    }
}

impl MemoryToolProvider for DefaultMemoryProvider {
    fn search_memory(&self, input: SearchMemoryInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let classifier = &self.classifier;
        let planner = &self.planner;
        let reranker = &self.reranker;
        let composer = &self.composer;

        let intents = classifier.classify(&input.query);
        let plan = planner.plan(&intents);
        let now_ts = chrono::Utc::now().timestamp();

        let mut results = if let Some(ref mt) = input.memory_type {
            // Explicit type filter takes precedence over intent routing.
            BM25Retriever::search_by_type(repo, &input.query, &input.project_id, mt, input.limit)?
        } else if self.config.retrieval.intent_routing {
            // Route to only the memory types implied by the classified intent
            // (General intent expands to all four, matching the old search_all).
            BM25Retriever::search_by_types(
                repo,
                &input.query,
                &input.project_id,
                &plan.sources,
                input.limit,
            )?
        } else {
            BM25Retriever::search_all(repo, &input.query, &input.project_id, input.limit)?
        };

        // Semantic fusion: when an embedder is present, blend vector top-K with
        // the BM25 candidates via RRF. Vector-only hits are materialized so they
        // can surface even when BM25 missed them entirely.
        #[cfg(feature = "semantic")]
        {
            if let Some(e) = self.embedder.as_ref() {
                if let Ok(mut qv) = e.embed(&[input.query.as_str()]) {
                    if !qv.is_empty() {
                        let qvec = qv.remove(0);
                        let loaded =
                            repo.load_active_embeddings(&input.project_id, e.model_id())?;
                        let type_of: std::collections::HashMap<String, String> = loaded
                            .iter()
                            .map(|(id, ty, _)| (id.clone(), ty.clone()))
                            .collect();
                        let cands: Vec<(String, Vec<f32>)> =
                            loaded.into_iter().map(|(id, _, v)| (id, v)).collect();
                        let vec_ids = crate::retrieval::vector::top_k_cosine(
                            &qvec,
                            &cands,
                            self.config.semantic.top_k,
                        );
                        let bm25_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
                        let fused = crate::retrieval::fusion::rrf_fuse(
                            &[bm25_ids, vec_ids],
                            self.config.semantic.rrf_k,
                        );
                        let mut by_id: std::collections::HashMap<String, SearchResult> =
                            results.into_iter().map(|r| (r.id.clone(), r)).collect();
                        let missing: Vec<(String, String)> = fused
                            .iter()
                            .filter(|id| !by_id.contains_key(*id))
                            .filter_map(|id| type_of.get(id).map(|ty| (ty.clone(), id.clone())))
                            .collect();
                        for sr in BM25Retriever::fetch_by_ids(repo, &missing)? {
                            by_id.entry(sr.id.clone()).or_insert(sr);
                        }
                        results = fused
                            .into_iter()
                            .filter_map(|id| by_id.remove(&id))
                            .collect();
                    }
                }
            }
        }

        reranker.deduplicate(&mut results);
        let half_life_seconds = (self.config.retrieval.recency_half_life_days as f32) * 86400.0;
        reranker.rerank(&mut results, &plan, now_ts, half_life_seconds);

        let budget = ContextBudget::new(
            self.config.context.context_window_tokens,
            self.config.context.memory_budget_percent,
        );
        let context = composer.compose_context(&results, &budget);

        let result_items: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "memory_type": r.memory_type,
                    "summary": r.summary,
                    "relevance_score": (r.relevance_score.clamp(0.0, 1.0) * 100.0).round() / 100.0,
                    "importance": (r.importance * 100.0).round() / 100.0,
                    "created_at": r.created_at,
                })
            })
            .collect();

        // Best-effort retrieval feedback: log this query + its hits so
        // `query_stats` / `engram queries` can surface hit-rate signal. A
        // logging failure must never break search.
        let result_ids: Vec<String> = results.iter().map(|r| r.id.clone()).collect();
        if let Err(e) = repo.record_query(
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

    fn related_files(&self, input: RelatedFilesInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        // Resolve the file's graph neighborhood directly from the indexed
        // graph_relations table — no full-project graph load into memory.
        let (entity_id, edges) = repo.related_files_for(&input.file, &input.project_id)?;
        let relations: Vec<_> = edges
            .iter()
            .map(|e| {
                serde_json::json!({
                    "to": e.other_name,
                    "type": e.relation_type,
                })
            })
            .collect();
        Ok(serde_json::json!({
            "entities": [{
                "id": entity_id.unwrap_or_default(),
                "type": "File",
                "name": input.file,
                "relations": relations,
            }]
        }))
    }

    fn timeline(&self, input: TimelineInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let since = chrono::Utc::now().timestamp() - (input.days * 86400);
        let conn = repo.connection()?;

        let mut stmt = conn.prepare(
            "SELECT date(created_at, 'unixepoch') as day, COUNT(*) as cnt
             FROM episodic_memories
             WHERE project_id = ?1 AND created_at >= ?2 AND archived_at IS NULL
             GROUP BY day ORDER BY day DESC",
        )?;

        let rows = stmt.query_map(rusqlite::params![input.project_id, since], |row| {
            let day: String = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok(serde_json::json!({
                "date": day,
                "episodic_count": count,
            }))
        })?;

        let events: Vec<serde_json::Value> = rows.filter_map(|r| r.ok()).collect();

        Ok(serde_json::json!({ "events": events }))
    }

    fn query_stats(&self, input: QueryStatsInput) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let since = chrono::Utc::now().timestamp() - (input.days * 86400);
        let rows = repo.query_stats(&input.project_id, since, input.limit)?;
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
        let query = input.service.as_deref().unwrap_or("");
        let results = if query.is_empty() {
            repo.list_recent_failures(&input.project_id, input.limit)?
                .into_iter()
                .map(|m| ScoredMemory {
                    memory: m,
                    bm25_score: 0.0,
                })
                .collect()
        } else {
            repo.search_failures(query, &input.project_id, input.limit)?
        };
        let failures: Vec<serde_json::Value> = results
            .iter()
            .map(|f| {
                serde_json::json!({
                    "id": f.memory.id,
                    "incident": f.memory.incident,
                    "severity": f.memory.severity,
                    "created_at": f.memory.created_at,
                })
            })
            .collect();

        Ok(serde_json::json!({ "failures": failures }))
    }

    fn architectural_decisions(
        &self,
        input: ArchitecturalDecisionsInput,
    ) -> Result<serde_json::Value> {
        let repo = self.lock_repo();
        let query = input.topic.as_deref().unwrap_or("");
        let results = if query.is_empty() {
            repo.list_recent_decisions(&input.project_id, input.limit)?
                .into_iter()
                .map(|m| ScoredMemory {
                    memory: m,
                    bm25_score: 0.0,
                })
                .collect()
        } else {
            repo.search_decisions(query, &input.project_id, input.limit)?
        };
        let decisions: Vec<serde_json::Value> = results
            .iter()
            .map(|d| {
                serde_json::json!({
                    "id": d.memory.id,
                    "title": d.memory.title,
                    "rationale": d.memory.rationale,
                    "created_at": d.memory.created_at,
                })
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

        repo.create_episodic(&memory)?;
        #[cfg(feature = "semantic")]
        self.index_embedding(
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
        let id = uuid::Uuid::new_v4().to_string();

        let memory = DecisionMemory {
            id: id.clone(),
            project_id: input.project_id,
            title: input.title,
            context: input.context,
            rationale: input.rationale,
            tradeoffs: input.tradeoffs,
            related_files: input.related_files,
            tags: input.tags,
            created_at: now,
            updated_at: now,
        };

        repo.create_decision(&memory)?;
        #[cfg(feature = "semantic")]
        self.index_embedding(
            "decision",
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

        repo.create_failure(&memory)?;
        #[cfg(feature = "semantic")]
        self.index_embedding(
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
            created_at: now,
            updated_at: now,
        };

        repo.create_procedural(&memory)?;
        #[cfg(feature = "semantic")]
        self.index_embedding(
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

    fn ingest_commits(&self, input: IngestCommitsInput) -> Result<serde_json::Value> {
        let repo_path = std::path::Path::new(&input.repo_path);
        let git = GitIntegration::new(repo_path)?;
        let session_id = input.session_id.unwrap_or_else(|| "auto-ingest".into());

        let memories = git.process_recent_commits(&input.project_id, &session_id, input.count)?;

        let repo = self.lock_repo();

        // Deduplicate: skip commits already ingested
        let ingested_hashes = repo.get_ingested_commits(&input.project_id)?;
        let total_before_dedup = memories.len();
        let new_memories: Vec<_> = memories
            .into_iter()
            .filter(|m| {
                m.related_commits
                    .iter()
                    .all(|c| !ingested_hashes.contains(c))
            })
            .collect();

        let mut ingested = Vec::new();
        for mem in &new_memories {
            repo.create_episodic(mem)?;
            ingested.push(serde_json::json!({
                "id": mem.id,
                "summary": mem.summary,
                "files_touched": mem.files_touched,
            }));
        }

        let skipped = total_before_dedup - new_memories.len();

        Ok(serde_json::json!({
            "ingested": ingested.len(),
            "total_commits_scanned": input.count,
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
            max_commits: input.max_commits,
            ingested_commit_hashes: ingested_hashes,
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

        // Guard: the fetched memory must belong to the caller's project.
        macro_rules! guarded_update {
            ($get:ident, $update:ident) => {{
                let existing = repo
                    .$get(&input.id)?
                    .ok_or_else(|| anyhow::anyhow!("memory not found: {}", input.id))?;
                if existing.project_id != input.project_id {
                    anyhow::bail!("memory does not belong to project {}", input.project_id);
                }
                let updated = merge_patch(&existing, &input.patch, now)?;
                repo.$update(&updated)?;
            }};
        }

        match kind {
            MemoryKind::Episodic => guarded_update!(get_episodic, update_episodic),
            MemoryKind::Decision => guarded_update!(get_decision, update_decision),
            MemoryKind::Failure => guarded_update!(get_failure, update_failure),
            MemoryKind::Procedural => guarded_update!(get_procedural, update_procedural),
        }

        Ok(serde_json::json!({
            "id": input.id,
            "status": "updated",
            "updated_at": now,
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
        let mut archived: Vec<serde_json::Value> = Vec::new();
        for kind in kinds {
            for row in repo.list_archived(kind, &input.project_id, input.limit)? {
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
        Ok(serde_json::json!({
            "applied": input.apply,
            "plans": plans,
            "total_archived": total_archived,
        }))
    }
}

/// Merge a patch object into an existing memory: overwrite only keys that already
/// exist on the model (ignores unknown keys), never touch id/project_id/created_at,
/// and stamp updated_at. Returns the rebuilt typed value.
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
    const PROTECTED: [&str; 4] = ["id", "project_id", "created_at", "memory_type"];
    for (k, v) in patch {
        if PROTECTED.contains(&k.as_str()) {
            continue;
        }
        if obj.contains_key(k) {
            obj.insert(k.clone(), v.clone());
        }
    }
    obj.insert("updated_at".into(), serde_json::json!(now));
    Ok(serde_json::from_value(serde_json::Value::Object(obj))?)
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

impl McpServer {
    pub fn new() -> Self {
        struct NoopProvider;
        impl MemoryToolProvider for NoopProvider {
            fn search_memory(&self, _: SearchMemoryInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"results": [], "total": 0}))
            }
            fn related_files(&self, _: RelatedFilesInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"entities": []}))
            }
            fn timeline(&self, _: TimelineInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"events": []}))
            }
            fn recent_failures(&self, _: RecentFailuresInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"failures": []}))
            }
            fn query_stats(&self, _: QueryStatsInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"queries": []}))
            }
            fn architectural_decisions(
                &self,
                _: ArchitecturalDecisionsInput,
            ) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"decisions": []}))
            }
            fn create_episodic(&self, _: CreateEpisodicInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"id": "", "status": "noop"}))
            }
            fn create_decision(&self, _: CreateDecisionInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"id": "", "status": "noop"}))
            }
            fn create_failure(&self, _: CreateFailureInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"id": "", "status": "noop"}))
            }
            fn create_procedural(&self, _: CreateProceduralInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"id": "", "status": "noop"}))
            }
            fn ingest_commits(&self, _: IngestCommitsInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"ingested": 0}))
            }
            fn collect_sources(&self, _: CollectSourcesInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"summary": {"total_items": 0}}))
            }
            fn forget_memory(&self, _: ForgetMemoryInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"archived": false, "status": "noop"}))
            }
            fn restore_memory(&self, _: RestoreMemoryInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"restored": false, "status": "noop"}))
            }
            fn update_memory(&self, _: UpdateMemoryInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"id": "", "status": "noop"}))
            }
            fn forget_batch(&self, _: ForgetBatchInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"applied": false, "matched": [], "count": 0}))
            }
            fn list_archived(&self, _: ListArchivedInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"archived": [], "count": 0}))
            }
            fn consolidate_memories(&self, _: ConsolidateInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"applied": false, "plans": [], "total_archived": 0}))
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
                description: "Search across all memory types using BM25 full-text search".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "query": { "type": "string", "description": "Search query" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"], "description": "Filter by memory type" },
                        "limit": { "type": "integer", "description": "Max results", "default": 10 },
                    },
                    "required": ["project_id", "query"],
                }),
            },
            ToolDefinition {
                name: "related_files".into(),
                description: "Find entities related to a specific file via the relationship graph".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "file": { "type": "string", "description": "File name to find relations for" },
                    },
                    "required": ["project_id", "file"],
                }),
            },
            ToolDefinition {
                name: "timeline".into(),
                description: "Get a timeline of memory events for the past N days".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "days": { "type": "integer", "description": "Number of days to look back", "default": 7 },
                    },
                    "required": ["project_id"],
                }),
            },
            ToolDefinition {
                name: "recent_failures".into(),
                description: "Get recent failure memories for a project".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "service": { "type": "string", "description": "Filter by service name" },
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
                name: "architectural_decisions".into(),
                description: "Get architectural decision memories for a project".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "topic": { "type": "string", "description": "Filter by topic keyword" },
                        "limit": { "type": "integer", "description": "Max results", "default": 10 },
                    },
                    "required": ["project_id"],
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
                description: "Record an architectural or design decision with rationale and tradeoffs".into(),
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
                description: "Patch fields of an existing memory (correction). Provide memory_type, id, and any fields to change.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "project_id": { "type": "string", "description": "Project identifier (required)" },
                        "memory_type": { "type": "string", "enum": ["episodic", "decision", "failure", "procedural"] },
                        "id": { "type": "string", "description": "Memory id to update" },
                    },
                    "required": ["project_id", "memory_type", "id"],
                    "additionalProperties": true,
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
        ]
    }

    /// Run the MCP server: read JSON-RPC from stdin, dispatch to a bounded
    /// worker pool, write responses (id-correlated, order-independent) under a
    /// stdout lock. `worker_threads=1` degenerates to sequential processing.
    pub fn run(self: Arc<Self>) -> Result<()> {
        use std::sync::mpsc;
        let n_workers = self.worker_threads.max(1);
        let stdout = Arc::new(Mutex::new(io::stdout()));
        let (tx, rx) = mpsc::channel::<JsonRpcRequest>();
        let rx = Arc::new(Mutex::new(rx));

        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let server = Arc::clone(&self);
            let rx = Arc::clone(&rx);
            let stdout = Arc::clone(&stdout);
            handles.push(std::thread::spawn(move || loop {
                // Poisoned-mutex recovery: a panic in a sibling worker must not
                // deadlock the survivors. into_inner reclaims the lock regardless.
                let req = {
                    let lock = rx.lock().unwrap_or_else(|e| e.into_inner());
                    match lock.recv() {
                        Ok(r) => r,
                        Err(_) => break, // channel closed → drain done
                    }
                };
                let response = server.handle_request(req);
                if let Ok(s) = serde_json::to_string(&response) {
                    let mut out = stdout.lock().unwrap_or_else(|e| e.into_inner());
                    let _ = writeln!(out, "{s}");
                    let _ = out.flush();
                }
            }));
        }

        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line.context("failed to read from stdin")?;
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            // JSON-RPC notifications (no `id`) get no response. Cache the id
            // for error recovery on malformed payloads.
            let cached_id = if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if val.get("id").is_none() {
                    continue;
                }
                val.get("id").cloned()
            } else {
                None
            };

            match serde_json::from_str::<JsonRpcRequest>(&line) {
                Ok(req) => {
                    if tx.send(req).is_err() {
                        break; // workers all exited
                    }
                }
                Err(e) => {
                    let response = JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: cached_id,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {e}"),
                            data: None,
                        }),
                    };
                    if let Ok(s) = serde_json::to_string(&response) {
                        let mut out = stdout.lock().unwrap_or_else(|e| e.into_inner());
                        let _ = writeln!(out, "{s}");
                        let _ = out.flush();
                    }
                }
            }
        }

        drop(tx); // close channel → workers exit after draining
        for h in handles {
            let _ = h.join();
        }
        Ok(())
    }

    fn handle_request(&self, request: JsonRpcRequest) -> JsonRpcResponse {
        match request.method.as_str() {
            "initialize" => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: request.id,
                result: Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": { "tools": {}, "prompts": {} },
                    "serverInfo": { "name": "engram", "version": "0.1.0" },
                })),
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
                    }],
                })),
                error: None,
            },
            "prompts/get" => {
                let params = request.params.unwrap_or_default();
                let name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if name != BOOTSTRAP_PROMPT_NAME {
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
                    let project_id = args
                        .get("project_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<project_id>");
                    let repo_path = args
                        .get("repo_path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("<repo_path>");
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
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or_default();

                let (result, parse_error): (
                    Option<Result<serde_json::Value>>,
                    Option<JsonRpcError>,
                ) = match tool_name {
                    "search_memory" => {
                        dispatch_tool!(arguments, SearchMemoryInput, self.provider, search_memory)
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
                    "architectural_decisions" => dispatch_tool!(
                        arguments,
                        ArchitecturalDecisionsInput,
                        self.provider,
                        architectural_decisions
                    ),
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
                    _ => (
                        None,
                        Some(JsonRpcError {
                            code: -32601,
                            message: format!("Unknown tool: {tool_name}"),
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
                        Err(e) => JsonRpcResponse {
                            jsonrpc: "2.0".into(),
                            id: request.id,
                            result: None,
                            error: Some(JsonRpcError {
                                code: -32603,
                                message: format!("Internal error: {e}"),
                                data: None,
                            }),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_provider() -> DefaultMemoryProvider {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        DefaultMemoryProvider::new(repo, Config::default())
    }

    /// End-to-end semantic recall with the REAL MiniLM model.
    ///
    /// Proves the core claim of direction ③: a query that shares **zero tokens**
    /// with a stored memory is recalled through the embedding path, while pure
    /// BM25 (feature compiled but `semantic.enabled=false`, so no embedder) misses
    /// it entirely. A second, unrelated memory makes the corpus non-trivial.
    ///
    /// Ignored by default (loads/downloads ~90MB of weights). Run with:
    ///   cargo test --features semantic -- --ignored semantic_recall
    #[cfg(feature = "semantic")]
    #[test]
    #[ignore = "loads the real MiniLM model; run with --features semantic --ignored"]
    fn semantic_recall_finds_lexically_disjoint_match() {
        const PROJECT: &str = "sem-recall-test";
        // Relevant memory — about auth/login, but shares no token with the query.
        let summary = "OAuth token refresh loop";
        let content = "The service repeatedly exchanges a refresh credential for a \
                       new bearer; the renewal cycle never terminates.";
        // Distractor — unrelated lexically AND semantically.
        let distractor_summary = "Postgres connection pool tuning";
        let distractor_content =
            "Raised the database pool size to handle the analytics dashboard load.";
        // Query — semantically about the auth problem, lexically disjoint from both.
        let query = "login keeps failing and re-authenticating";

        // Build a provider with the given semantic flag and seed both memories.
        fn seed(enabled: bool, mems: &[(&str, &str)]) -> DefaultMemoryProvider {
            let repo = MemoryRepository::new_in_memory().unwrap();
            repo.initialize_schema().unwrap();
            let mut config = Config::default();
            config.semantic.enabled = enabled;
            let provider = DefaultMemoryProvider::new(repo, config);
            for (summary, content) in mems {
                provider
                    .create_episodic(CreateEpisodicInput {
                        project_id: PROJECT.into(),
                        session_id: "s".into(),
                        summary: (*summary).into(),
                        content: (*content).into(),
                        files_touched: vec![],
                        related_commits: vec![],
                        importance: 0.5,
                        tags: vec![],
                    })
                    .unwrap();
            }
            provider
        }

        let mems = [(summary, content), (distractor_summary, distractor_content)];
        let recalled = |v: &serde_json::Value| -> bool {
            v["results"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["summary"] == summary)
        };

        // BM25-only baseline: no lexical overlap → the relevant memory is unreachable.
        let bm25 = seed(false, &mems);
        let bm25_res = bm25
            .search_memory(SearchMemoryInput {
                project_id: PROJECT.into(),
                query: query.into(),
                memory_type: None,
                limit: 10,
            })
            .unwrap();
        assert!(
            !recalled(&bm25_res),
            "BM25-only must MISS the lexically-disjoint query, got: {bm25_res}"
        );

        // Semantic enabled: the embedding path recalls the relevant memory.
        let sem = seed(true, &mems);
        let sem_res = sem
            .search_memory(SearchMemoryInput {
                project_id: PROJECT.into(),
                query: query.into(),
                memory_type: None,
                limit: 10,
            })
            .unwrap();
        assert!(
            recalled(&sem_res),
            "semantic search must RECALL the lexically-disjoint match, got: {sem_res}"
        );
    }

    /// reindex backfills embeddings for memories written BEFORE semantic was on.
    /// Seeds via the repo directly (no embedding), then reindex makes them
    /// semantically recallable. Real MiniLM; run with:
    ///   cargo test --features semantic -- --ignored reindex_backfills
    #[cfg(feature = "semantic")]
    #[test]
    #[ignore = "loads the real MiniLM model; run with --features semantic --ignored"]
    fn reindex_backfills_then_recalls() {
        const P: &str = "reindex-test";

        // 1) Seed an episodic straight through the repo → NO embedding stored.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let ts = 1_700_000_000i64;
        repo.create_episodic(&EpisodicMemory {
            id: "oauth".into(),
            project_id: P.into(),
            session_id: "s".into(),
            summary: "OAuth token refresh loop".into(),
            content: "The service repeatedly exchanges a refresh credential for a \
                      new bearer; the renewal cycle never terminates."
                .into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: ts,
            updated_at: ts,
        })
        .unwrap();

        // 2) Build a semantic-enabled provider over THAT repo (moves repo in).
        let mut config = Config::default();
        config.semantic.enabled = true;
        let provider = DefaultMemoryProvider::new(repo, config);

        let query = "login keeps failing and re-authenticating"; // lexically disjoint
        let recalled = |p: &DefaultMemoryProvider| -> bool {
            let v = p
                .search_memory(SearchMemoryInput {
                    project_id: P.into(),
                    query: query.into(),
                    memory_type: None,
                    limit: 10,
                })
                .unwrap();
            v["results"]
                .as_array()
                .unwrap()
                .iter()
                .any(|r| r["summary"] == "OAuth token refresh loop")
        };

        // 3) BEFORE reindex: no vectors → lexically-disjoint query misses.
        assert!(!recalled(&provider), "no embedding yet → should miss");

        // 4) reindex embeds it.
        let rep = provider.reindex_embeddings(Some(P), false, false).unwrap();
        assert_eq!(rep.total, 1);
        assert_eq!(rep.embedded, 1);
        assert_eq!(rep.failed, 0);

        // 5) AFTER reindex: recalled via the embedding path.
        assert!(recalled(&provider), "after reindex → should recall");

        // 6) Idempotent: a second non-force reindex skips the already-embedded one.
        let rep2 = provider.reindex_embeddings(Some(P), false, false).unwrap();
        assert_eq!(rep2.embedded, 0);
        assert_eq!(rep2.skipped, 1);

        // 7) --force re-embeds.
        let rep3 = provider.reindex_embeddings(Some(P), true, false).unwrap();
        assert_eq!(rep3.embedded, 1);

        // 8) --dry-run writes nothing; already-embedded corpus → all skipped.
        let rep4 = provider.reindex_embeddings(Some(P), false, true).unwrap();
        assert!(rep4.dry_run);
        assert_eq!(rep4.embedded, 0, "all already embedded → none would embed");
        assert_eq!(rep4.skipped, 1);
    }

    #[test]
    fn test_list_tools() {
        let tools = McpServer::list_tools();
        assert_eq!(tools.len(), 18);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search_memory"));
        assert!(names.contains(&"related_files"));
        assert!(names.contains(&"timeline"));
        assert!(names.contains(&"recent_failures"));
        assert!(names.contains(&"architectural_decisions"));
        assert!(names.contains(&"query_stats"));
        assert!(names.contains(&"create_episodic"));
        assert!(names.contains(&"create_decision"));
        assert!(names.contains(&"create_failure"));
        assert!(names.contains(&"create_procedural"));
        assert!(names.contains(&"ingest_commits"));
        assert!(names.contains(&"collect_sources"));
    }

    #[test]
    fn test_tool_schemas_require_project_id() {
        let tools = McpServer::list_tools();
        for tool in &tools {
            let required = tool
                .input_schema
                .get("required")
                .and_then(|r| r.as_array())
                .unwrap();
            let has_project_id = required.iter().any(|v| v.as_str() == Some("project_id"));
            assert!(has_project_id, "Tool {} must require project_id", tool.name);
        }
    }

    #[test]
    fn test_handle_initialize() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "initialize".into(),
            params: None,
        });

        assert!(response.result.is_some());
        assert!(response.error.is_none());
        let result = response.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "engram");
    }

    #[test]
    fn test_handle_tools_list() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(2)),
            method: "tools/list".into(),
            params: None,
        });

        assert!(response.result.is_some());
        let result = response.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 18);
    }

    #[test]
    fn test_create_episodic_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        // Create episodic memory
        let result = provider
            .create_episodic(CreateEpisodicInput {
                project_id: "test-project".into(),
                session_id: "session-1".into(),
                summary: "Fixed OAuth refresh loop".into(),
                content: "The refresh token was looping due to stale cache".into(),
                files_touched: vec!["auth.ts".into()],
                related_commits: vec!["abc123".into()],
                importance: 0.8,
                tags: vec!["auth".into(), "oauth".into()],
            })
            .unwrap();

        assert_eq!(result["status"], "created");
        let id = result["id"].as_str().unwrap();

        // Verify it can be searched
        let search_result = provider
            .search_memory(SearchMemoryInput {
                project_id: "test-project".into(),
                query: "OAuth refresh".into(),
                memory_type: None,
                limit: 10,
            })
            .unwrap();

        let results = search_result["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["id"], id);
    }

    #[test]
    fn related_files_resolves_via_repo_index() {
        // related_files must work without the (now-removed) in-memory graph:
        // it resolves the file's neighborhood straight from graph_relations.
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let provider = Arc::new(DefaultMemoryProvider::new(repo, Config::default()));

        provider
            .create_episodic(CreateEpisodicInput {
                project_id: "p".into(),
                session_id: "s".into(),
                summary: "summary".into(),
                content: "content".into(),
                files_touched: vec!["auth.ts".into()],
                related_commits: vec![],
                importance: 0.0,
                tags: vec![],
            })
            .unwrap();

        let result = provider
            .related_files(RelatedFilesInput {
                project_id: "p".into(),
                file: "auth.ts".into(),
            })
            .unwrap();

        let entities = result["entities"].as_array().unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0]["type"], "File");
        assert_eq!(entities[0]["name"], "auth.ts");
        let relations = entities[0]["relations"].as_array().unwrap();
        assert!(
            relations.iter().any(|r| r["type"] == "Touches"),
            "expected a Touches relation, got: {relations:?}"
        );
    }

    #[test]
    fn search_output_includes_importance() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        provider
            .create_episodic(CreateEpisodicInput {
                project_id: "p".into(),
                session_id: "s".into(),
                summary: "Fixed OAuth refresh loop".into(),
                content: "The refresh token was looping due to stale cache".into(),
                files_touched: vec!["auth.ts".into()],
                related_commits: vec!["abc123".into()],
                importance: 0.8,
                tags: vec!["auth".into(), "oauth".into()],
            })
            .unwrap();

        let out = provider
            .search_memory(SearchMemoryInput {
                project_id: "p".into(),
                query: "OAuth refresh".into(),
                memory_type: Some("episodic".into()),
                limit: 10,
            })
            .unwrap();

        let results = out["results"].as_array().unwrap();
        assert!(!results.is_empty(), "should have at least one result");
        let first = &results[0];
        assert!(
            first.get("importance").is_some(),
            "result must expose importance"
        );
        // episodic importance=0.8 should map through (passthrough, clamped)
        let imp = first["importance"]
            .as_f64()
            .expect("importance must be a number");
        assert!(
            (imp - 0.8).abs() < 1e-6,
            "importance should be 0.8, got {}",
            imp
        );
    }

    #[test]
    fn test_create_decision_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        let result = provider
            .create_decision(CreateDecisionInput {
                project_id: "test-project".into(),
                title: "Use Redis for session caching".into(),
                context: "Auth service needs sub-ms latency".into(),
                rationale: "Redis provides sub-millisecond reads".into(),
                tradeoffs: "Added infrastructure complexity".into(),
                related_files: vec!["auth.ts".into()],
                tags: vec!["architecture".into()],
            })
            .unwrap();

        assert_eq!(result["status"], "created");

        // Verify via search
        let search = provider
            .search_memory(SearchMemoryInput {
                project_id: "test-project".into(),
                query: "Redis".into(),
                memory_type: Some("decision".into()),
                limit: 5,
            })
            .unwrap();
        assert_eq!(search["results"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_create_failure_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        let result = provider
            .create_failure(CreateFailureInput {
                project_id: "test-project".into(),
                incident: "Auth token expiry mismatch".into(),
                root_cause: "Clock skew between services".into(),
                fix: "Added clock tolerance window".into(),
                prevention: "Monitor clock sync".into(),
                severity: 4,
                tags: vec!["auth".into()],
            })
            .unwrap();

        assert_eq!(result["status"], "created");
        assert_eq!(result["severity"], 4);
    }

    #[test]
    fn test_create_procedural_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        let result = provider
            .create_procedural(CreateProceduralInput {
                project_id: "test-project".into(),
                workflow_name: "deployment".into(),
                steps: vec![
                    "run tests".into(),
                    "build docker".into(),
                    "push to registry".into(),
                ],
                related_tools: vec!["docker".into()],
                tags: vec!["deploy".into()],
            })
            .unwrap();

        assert_eq!(result["status"], "created");
    }

    #[test]
    fn test_ingest_commits_with_temp_repo() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        // Create temp git repo (via system git — see git_integration::make_test_repo)
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_path = temp_dir.path();
        crate::git_integration::make_test_repo(
            repo_path,
            "main.rs",
            "fn main() {}",
            "feat: initial commit",
        );

        // Ingest
        let result = provider
            .ingest_commits(IngestCommitsInput {
                project_id: "test-project".into(),
                repo_path: repo_path.to_string_lossy().to_string(),
                count: 10,
                session_id: Some("test-session".into()),
            })
            .unwrap();

        assert_eq!(result["ingested"], 1);
    }

    #[test]
    fn test_handle_unknown_method() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(3)),
            method: "unknown/method".into(),
            params: None,
        });

        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[test]
    fn test_search_memory_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));
        let server = McpServer::with_provider(provider);

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(4)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "search_memory",
                "arguments": {
                    "project_id": "test-project",
                    "query": "oauth"
                }
            })),
        });

        assert!(response.result.is_some());
        assert!(response.error.is_none());
    }

    #[test]
    fn search_memory_logs_query_for_feedback() {
        // search_memory must record each query to query_log (retrieval feedback).
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        repo.create_episodic(&crate::models::EpisodicMemory {
            id: "e1".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "oauth token bug".into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 1_700_000_000,
            updated_at: 1_700_000_000,
        })
        .unwrap();
        let config = Config::default();
        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));
        let server = McpServer::with_provider(provider.clone());

        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "search_memory",
                "arguments": { "project_id": "p", "query": "oauth" }
            })),
        });
        assert!(response.result.is_some());

        // The query must have been logged for retrieval feedback.
        let stats = provider.repo.query_stats("p", 0, 10).unwrap();
        assert!(
            stats.iter().any(|s| s.query == "oauth"),
            "query 'oauth' must be logged for feedback: {stats:?}"
        );
    }

    #[test]
    fn test_search_memory_missing_project_id() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(5)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "search_memory",
                "arguments": {
                    "query": "oauth"
                }
            })),
        });

        // Should get an error because project_id is required
        assert!(response.error.is_some());
    }

    // ─── MCP JSON-RPC Integration Tests ────────────────────────────

    fn setup_server() -> McpServer {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();
        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));
        McpServer::with_provider(provider)
    }

    #[test]
    fn test_jsonrpc_notification_no_response() {
        // Notifications (no id) should be silently skipped — handle_request
        // only processes requests with an `id` field. The run() loop filters
        // them out, but handle_request itself would still process them.
        // This test verifies the behavior at the handle_request level.
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "initialize".into(),
            params: None,
        });
        // Should get a valid response for requests with id
        assert!(response.result.is_some());
    }

    #[test]
    fn test_jsonrpc_parse_error_response() {
        let server = McpServer::new();
        // Malformed method name still returns valid response
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "nonexistent/method".into(),
            params: None,
        });
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[test]
    fn test_jsonrpc_tools_call_unknown_tool() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "nonexistent_tool",
                "arguments": {}
            })),
        });
        assert!(response.error.is_some());
        let err = response.error.unwrap();
        assert_eq!(err.code, -32601);
        assert!(err.message.contains("Unknown tool"));
    }

    #[test]
    fn test_jsonrpc_tools_call_invalid_params() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "create_episodic",
                "arguments": {
                    "project_id": "test"
                    // missing required fields: session_id, summary, content
                }
            })),
        });
        assert!(response.error.is_some());
        let err = response.error.unwrap();
        assert_eq!(err.code, -32602);
        assert!(err.message.contains("Invalid params"));
    }

    #[test]
    fn test_jsonrpc_create_and_search_roundtrip() {
        let server = setup_server();
        // Create episodic memory via JSON-RPC
        let create_resp = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "create_episodic",
                "arguments": {
                    "project_id": "roundtrip-test",
                    "session_id": "s1",
                    "summary": "Fixed authentication token expiry bug",
                    "content": "Token was not refreshed properly causing 401 errors",
                    "files_touched": ["auth.rs", "token.rs"],
                    "tags": ["auth", "bug"],
                    "importance": 0.9
                }
            })),
        });
        let create_result = create_resp.result.unwrap();
        let text: &str = create_result["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(parsed["status"], "created");

        // Search for it
        let search_resp = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(2)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "search_memory",
                "arguments": {
                    "project_id": "roundtrip-test",
                    "query": "authentication token"
                }
            })),
        });
        assert!(search_resp.error.is_none());
        let search_result = search_resp.result.unwrap();
        let search_text: &str = search_result["content"][0]["text"].as_str().unwrap();
        let search_parsed: serde_json::Value = serde_json::from_str(search_text).unwrap();
        let results = search_parsed["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0]["summary"]
            .as_str()
            .unwrap()
            .contains("authentication"));
    }

    #[test]
    fn test_jsonrpc_recent_failures_list_without_query() {
        let server = setup_server();

        // Create a failure memory
        server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "create_failure",
                "arguments": {
                    "project_id": "failure-test",
                    "incident": "Database connection timeout",
                    "root_cause": "Connection pool exhausted",
                    "fix": "Increased pool size to 20",
                    "prevention": "Monitor pool usage metrics",
                    "severity": 4
                }
            })),
        });

        // List recent failures (no service filter)
        let list_resp = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(2)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "recent_failures",
                "arguments": {
                    "project_id": "failure-test"
                }
            })),
        });
        assert!(list_resp.error.is_none());
        let list_result = list_resp.result.unwrap();
        let list_text: &str = list_result["content"][0]["text"].as_str().unwrap();
        let list_parsed: serde_json::Value = serde_json::from_str(list_text).unwrap();
        let failures = list_parsed["failures"].as_array().unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0]["severity"], 4);
    }

    #[test]
    fn test_jsonrpc_timeline() {
        let server = setup_server();

        // Create an episodic memory
        server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "create_episodic",
                "arguments": {
                    "project_id": "timeline-test",
                    "session_id": "s1",
                    "summary": "Test event",
                    "content": "Test content"
                }
            })),
        });

        let timeline_resp = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(2)),
            method: "tools/call".into(),
            params: Some(serde_json::json!({
                "name": "timeline",
                "arguments": {
                    "project_id": "timeline-test",
                    "days": 1
                }
            })),
        });
        assert!(timeline_resp.error.is_none());
        let timeline_result = timeline_resp.result.unwrap();
        let timeline_text: &str = timeline_result["content"][0]["text"].as_str().unwrap();
        let timeline_parsed: serde_json::Value = serde_json::from_str(timeline_text).unwrap();
        let events = timeline_parsed["events"].as_array().unwrap();
        assert!(!events.is_empty());
    }

    // ─── collect_sources + prompts Integration Tests ────────────────

    #[test]
    fn test_initialize_advertises_prompts_capability() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "initialize".into(),
            params: None,
        });
        let caps = &response.result.unwrap()["capabilities"];
        assert!(
            caps.get("prompts").is_some(),
            "prompts capability advertised"
        );
        assert!(caps.get("tools").is_some());
    }

    #[test]
    fn test_prompts_list_returns_bootstrap() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "prompts/list".into(),
            params: None,
        });
        assert!(response.error.is_none());
        let result = response.result.unwrap();
        let prompts = result["prompts"].as_array().unwrap();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0]["name"], "engram.bootstrap");
        let args = prompts[0]["arguments"].as_array().unwrap();
        assert!(args
            .iter()
            .any(|a| a["name"] == "project_id" && a["required"] == true));
    }

    #[test]
    fn test_prompts_get_renders_template() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "prompts/get".into(),
            params: Some(serde_json::json!({
                "name": "engram.bootstrap",
                "arguments": {
                    "project_id": "myproj",
                    "repo_path": "/tmp/x",
                    "dimensions": "git"
                }
            })),
        });
        assert!(response.error.is_none());
        let result = response.result.unwrap();
        let text = result["messages"][0]["content"]["text"].as_str().unwrap();
        assert!(text.contains("myproj"));
        assert!(text.contains("/tmp/x"));
        assert!(!text.contains("{{PROJECT_ID}}"), "placeholder substituted");
        assert!(text.contains("Iron rules"), "guidance body present");
    }

    #[test]
    fn test_prompts_get_unknown_prompt_errors() {
        let server = McpServer::new();
        let response = server.handle_request(JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "prompts/get".into(),
            params: Some(serde_json::json!({ "name": "bogus" })),
        });
        assert!(response.error.is_some());
        assert_eq!(response.error.unwrap().code, -32601);
    }

    #[test]
    fn test_collect_sources_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();
        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        crate::git_integration::make_test_repo(path, "main.rs", "fn main() {}", "feat: initial");
        std::fs::write(path.join("README.md"), "# Project\n").unwrap();
        std::fs::write(path.join("Cargo.toml"), "[dependencies]\nserde = \"1\"\n").unwrap();

        let result = provider
            .collect_sources(CollectSourcesInput {
                project_id: "test".into(),
                repo_path: path.to_string_lossy().into_owned(),
                dimensions: Some("git,decisions".into()),
                max_commits: 50,
            })
            .unwrap();

        assert!(
            result["summary"]["total_items"].as_u64().unwrap() > 0,
            "found some material: {result}"
        );
        assert!(result["git"].is_object());
        assert!(result["decisions"].is_object());
        // Unrequested dimensions are omitted (skip_serializing_if Option::is_none).
        assert!(result.get("failures").is_none());
        assert!(result.get("workflow").is_none());
    }

    #[test]
    fn test_collect_sources_rejects_empty_dimensions() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let config = Config::default();
        let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

        let dir = tempfile::tempdir().unwrap();
        let err = provider
            .collect_sources(CollectSourcesInput {
                project_id: "test".into(),
                repo_path: dir.path().to_string_lossy().into_owned(),
                dimensions: Some("bogus,also-bogus".into()),
                max_commits: 50,
            })
            .unwrap_err();
        assert!(err.to_string().contains("no valid dimensions"));
    }

    #[test]
    fn test_update_memory_patches_fields_and_guards_project() {
        let provider = make_provider();
        let created = provider
            .create_episodic(CreateEpisodicInput {
                project_id: "p".into(),
                session_id: "s".into(),
                summary: "old summary".into(),
                content: "old content".into(),
                files_touched: vec![],
                related_commits: vec![],
                importance: 0.5,
                tags: vec![],
            })
            .unwrap();
        let id = created["id"].as_str().unwrap().to_string();

        // patch summary。
        let mut patch = serde_json::Map::new();
        patch.insert("summary".into(), serde_json::json!("new summary"));
        provider
            .update_memory(UpdateMemoryInput {
                project_id: "p".into(),
                memory_type: "episodic".into(),
                id: id.clone(),
                patch: patch.clone(),
            })
            .unwrap();

        // 新词搜得到、旧词搜不到。
        let hit_new = provider
            .search_memory(SearchMemoryInput {
                project_id: "p".into(),
                query: "new".into(),
                memory_type: Some("episodic".into()),
                limit: 10,
            })
            .unwrap();
        assert_eq!(hit_new["results"].as_array().unwrap().len(), 1);

        // 跨 project 守卫：用错误 project 更新应报错。
        let err = provider.update_memory(UpdateMemoryInput {
            project_id: "other".into(),
            memory_type: "episodic".into(),
            id: id.clone(),
            patch,
        });
        assert!(err.is_err());
    }

    #[test]
    fn test_forget_and_restore_memory_tools() {
        let provider = make_provider(); // 既有测试若无此 helper，见下方 3e
                                        // 先写一条 episodic。
        let created = provider
            .create_episodic(CreateEpisodicInput {
                project_id: "p".into(),
                session_id: "s".into(),
                summary: "forget me".into(),
                content: "c".into(),
                files_touched: vec![],
                related_commits: vec![],
                importance: 0.5,
                tags: vec![],
            })
            .unwrap();
        let id = created["id"].as_str().unwrap().to_string();

        // forget 命中。
        let r = provider
            .forget_memory(ForgetMemoryInput {
                project_id: "p".into(),
                memory_type: "episodic".into(),
                id: id.clone(),
            })
            .unwrap();
        assert_eq!(r["archived"], serde_json::json!(true));

        // 搜不到了。
        let s = provider
            .search_memory(SearchMemoryInput {
                project_id: "p".into(),
                query: "forget".into(),
                memory_type: Some("episodic".into()),
                limit: 10,
            })
            .unwrap();
        assert_eq!(s["results"].as_array().unwrap().len(), 0);

        // restore 命中。
        let r2 = provider
            .restore_memory(RestoreMemoryInput {
                project_id: "p".into(),
                memory_type: "episodic".into(),
                id: id.clone(),
            })
            .unwrap();
        assert_eq!(r2["restored"], serde_json::json!(true));
    }

    #[test]
    fn test_forget_batch_dry_run_then_apply_and_list() {
        let provider = make_provider();
        for s in ["one", "two"] {
            provider
                .create_episodic(CreateEpisodicInput {
                    project_id: "p".into(),
                    session_id: "s".into(),
                    summary: s.into(),
                    content: "c".into(),
                    files_touched: vec![],
                    related_commits: vec![],
                    importance: 0.5,
                    tags: vec!["bootstrap".into()],
                })
                .unwrap();
        }

        // dry-run：返回候选但不归档。
        let dry = provider
            .forget_batch(ForgetBatchInput {
                project_id: "p".into(),
                memory_type: Some("episodic".into()),
                tags: vec!["bootstrap".into()],
                before: None,
                apply: false,
            })
            .unwrap();
        assert_eq!(dry["matched"].as_array().unwrap().len(), 2);
        assert_eq!(dry["applied"], serde_json::json!(false));
        assert_eq!(
            provider
                .list_archived(ListArchivedInput {
                    project_id: "p".into(),
                    memory_type: Some("episodic".into()),
                    limit: 10
                })
                .unwrap()["archived"]
                .as_array()
                .unwrap()
                .len(),
            0
        );

        // apply：真正归档。
        let applied = provider
            .forget_batch(ForgetBatchInput {
                project_id: "p".into(),
                memory_type: Some("episodic".into()),
                tags: vec!["bootstrap".into()],
                before: None,
                apply: true,
            })
            .unwrap();
        assert_eq!(applied["applied"], serde_json::json!(true));
        let listed = provider
            .list_archived(ListArchivedInput {
                project_id: "p".into(),
                memory_type: Some("episodic".into()),
                limit: 10,
            })
            .unwrap();
        assert_eq!(listed["archived"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_consolidate_memories_tool_dry_run() {
        let provider = make_provider();
        for _ in 0..2 {
            provider
                .create_episodic(CreateEpisodicInput {
                    project_id: "p".into(),
                    session_id: "s".into(),
                    summary: "dup".into(),
                    content: "same".into(),
                    files_touched: vec![],
                    related_commits: vec![],
                    importance: 0.5,
                    tags: vec![],
                })
                .unwrap();
        }
        let out = provider
            .consolidate_memories(ConsolidateInput {
                project_id: "p".into(),
                memory_type: Some("episodic".into()),
                include_near_dup: false,
                apply: false,
            })
            .unwrap();
        // dry-run：报告一组重复，但两条都仍可搜到。
        assert_eq!(out["applied"], serde_json::json!(false));
        // 工具层应把引擎找到的重复组透传出来（episodic 这组有 1 个 group）。
        let plans = out["plans"].as_array().unwrap();
        let groups: usize = plans
            .iter()
            .map(|p| p["groups"].as_array().unwrap().len())
            .sum();
        assert_eq!(groups, 1);
        assert_eq!(
            provider
                .search_memory(SearchMemoryInput {
                    project_id: "p".into(),
                    query: "dup".into(),
                    memory_type: Some("episodic".into()),
                    limit: 10
                })
                .unwrap()["results"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn concurrent_handle_request_is_safe() {
        // Concurrent requests against a shared Arc<McpServer> must all succeed
        // without panic/deadlock. Mixes tools/list (static, no repo) with
        // tools/call search_memory (exercises lock_repo → Mutex<MemoryRepository>),
        // so this actually guards the worker-pool + repo-lock interplay, not
        // just Arc sharing of a read-only branch.
        let server = Arc::new(setup_server());
        // Seed a memory so search_memory has a hit and actually touches the repo.
        server
            .provider
            .create_episodic(CreateEpisodicInput {
                project_id: "p".into(),
                session_id: "s".into(),
                summary: "seed concurrency probe".into(),
                content: "details".into(),
                files_touched: vec![],
                related_commits: vec![],
                importance: 0.5,
                tags: vec![],
            })
            .unwrap();

        let mut handles = vec![];
        for i in 0..16 {
            let server = Arc::clone(&server);
            handles.push(std::thread::spawn(move || {
                // Alternate between the repo-touching path and the static path.
                let req = if i % 2 == 0 {
                    JsonRpcRequest {
                        jsonrpc: "2.0".into(),
                        id: Some(serde_json::json!(i)),
                        method: "tools/call".into(),
                        params: Some(serde_json::json!({
                            "name": "search_memory",
                            "arguments": {"project_id": "p", "query": "seed", "limit": 5},
                        })),
                    }
                } else {
                    JsonRpcRequest {
                        jsonrpc: "2.0".into(),
                        id: Some(serde_json::json!(i)),
                        method: "tools/list".into(),
                        params: None,
                    }
                };
                let resp = server.handle_request(req);
                assert!(
                    resp.error.is_none(),
                    "concurrent request {} errored: {:?}",
                    i,
                    resp.error
                );
            }));
        }
        for h in handles {
            h.join().expect("worker thread panicked");
        }
    }
}
