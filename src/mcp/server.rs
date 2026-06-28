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
const CONSOLIDATE_JACCARD_THRESHOLD: f64 = 0.85;

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
        // the BM25 candidates via RRF via the embedding service. Vector-only
        // hits are materialized so they can surface even when BM25 missed them.
        #[cfg(feature = "semantic")]
        {
            if self.embedding.is_active() {
                results = self.embedding.fuse(
                    repo,
                    &input.query,
                    &input.project_id,
                    results,
                    self.config.semantic.top_k,
                    self.config.semantic.rrf_k,
                )?;
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
        let rows = repo.timeline(&input.project_id, since)?;
        let events: Vec<serde_json::Value> = rows
            .iter()
            .map(|r| {
                serde_json::json!({
                    "date": r.day,
                    "episodic_count": r.count,
                })
            })
            .collect();
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
        self.embedding.index(
            repo,
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
            created_at: now,
            updated_at: now,
        };

        repo.create_procedural(&memory)?;
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
            fn reflect(&self, _: ReflectInput) -> Result<serde_json::Value> {
                Ok(
                    serde_json::json!({"applied": false, "proposed": 0, "created": 0, "suggestions": []}),
                )
            }
            fn list_suggestions(&self, _: ListSuggestionsInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"count": 0, "suggestions": []}))
            }
            fn confirm_suggestion(&self, _: SuggestionIdInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"id": "", "status": "noop"}))
            }
            fn reject_suggestion(&self, _: SuggestionIdInput) -> Result<serde_json::Value> {
                Ok(serde_json::json!({"id": "", "status": "noop"}))
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

impl RequestHandler for McpServer {
    fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        self.handle_request(req)
    }
}

#[cfg(test)]
#[path = "server_tests.rs"]
mod server_tests;
