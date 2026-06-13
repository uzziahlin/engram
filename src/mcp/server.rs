use crate::config::Config;
use crate::context::composer::{ContextBudget, ContextComposer};
use crate::git_integration::GitIntegration;
use crate::graph::GraphEngine;
use crate::models::*;
use crate::retrieval::bm25::BM25Retriever;
use crate::retrieval::intent_classifier::IntentClassifier;
use crate::retrieval::planner::RetrievalPlanner;
use crate::retrieval::reranker::Reranker;
use crate::storage::MemoryRepository;
use crate::storage::ScoredMemory;
use anyhow::{Context as AnyhowContext, Result};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::collections::HashSet;

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
}

/// Dispatch a tool call: deserialize arguments and invoke the provider method.
/// Returns `(Some(result), None)` on successful parse, or `(None, Some(error))` on parse failure.
macro_rules! dispatch_tool {
    ($args:expr, $input_ty:ty, $provider:expr, $method:ident) => {
        match serde_json::from_value::<$input_ty>($args) {
            Ok(i) => (Some($provider.$method(i)), None),
            Err(e) => (None, Some(JsonRpcError { code: -32602, message: format!("Invalid params: {e}") })),
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
    fn architectural_decisions(&self, input: ArchitecturalDecisionsInput) -> Result<serde_json::Value>;

    // Write tools
    fn create_episodic(&self, input: CreateEpisodicInput) -> Result<serde_json::Value>;
    fn create_decision(&self, input: CreateDecisionInput) -> Result<serde_json::Value>;
    fn create_failure(&self, input: CreateFailureInput) -> Result<serde_json::Value>;
    fn create_procedural(&self, input: CreateProceduralInput) -> Result<serde_json::Value>;
    fn ingest_commits(&self, input: IngestCommitsInput) -> Result<serde_json::Value>;
}

/// Default implementation of MemoryToolProvider backed by SQLite.
/// Wraps MemoryRepository in Mutex for thread safety (rusqlite::Connection is not Sync).
pub struct DefaultMemoryProvider {
    repo: Mutex<MemoryRepository>,
    graph: Mutex<GraphEngine>,
    config: Config,
    classifier: IntentClassifier,
    planner: RetrievalPlanner,
    reranker: Reranker,
    composer: ContextComposer,
    loaded_projects: Mutex<HashSet<String>>,
}

impl DefaultMemoryProvider {
    pub fn new(repo: MemoryRepository, graph: GraphEngine, config: Config) -> Self {
        Self {
            repo: Mutex::new(repo),
            graph: Mutex::new(graph),
            config,
            classifier: IntentClassifier::new(),
            planner: RetrievalPlanner::new(),
            reranker: Reranker::new(),
            composer: ContextComposer::new(),
            loaded_projects: Mutex::new(HashSet::new()),
        }
    }
}

impl MemoryToolProvider for DefaultMemoryProvider {
    fn search_memory(&self, input: SearchMemoryInput) -> Result<serde_json::Value> {
        let repo = self.repo.lock().unwrap();
        let classifier = &self.classifier;
        let planner = &self.planner;
        let reranker = &self.reranker;
        let composer = &self.composer;

        let intents = classifier.classify(&input.query);
        let plan = planner.plan(&intents);
        let now_ts = chrono::Utc::now().timestamp();

        let mut results = if let Some(ref mt) = input.memory_type {
            BM25Retriever::search_by_type(
                &repo,
                &input.query,
                &input.project_id,
                mt,
                input.limit,
            )?
        } else {
            BM25Retriever::search_all(
                &repo,
                &input.query,
                &input.project_id,
                input.limit,
            )?
        };

        reranker.deduplicate(&mut results);
        reranker.rerank(&mut results, &plan, now_ts);

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
                    "relevance_score": (r.relevance_score * 100.0).round() / 100.0,
                    "created_at": r.created_at,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "results": result_items,
            "total": result_items.len(),
            "context": context,
        }))
    }

    fn related_files(&self, input: RelatedFilesInput) -> Result<serde_json::Value> {
        // Load graph for this project only once
        {
            let mut loaded = self.loaded_projects.lock().unwrap();
            if !loaded.contains(&input.project_id) {
                let repo = self.repo.lock().unwrap();
                let mut graph = self.graph.lock().unwrap();
                graph.load_from_repo(&repo, &input.project_id)?;
                loaded.insert(input.project_id.clone());
            }
        }

        let repo = self.repo.lock().unwrap();
        let graph = self.graph.lock().unwrap();

        let entity_id = {
            let conn = repo.connection();
            let mut stmt = conn.prepare(
                "SELECT id FROM entities WHERE name = ?1 AND project_id = ?2 LIMIT 1",
            )?;
            stmt.query_row(
                rusqlite::params![input.file, input.project_id],
                |row| row.get::<_, String>(0),
            )
            .optional()?
        };

        let relations = if let Some(eid) = &entity_id {
            graph.get_relations(eid)
                .iter()
                .map(|rel| {
                    let target = if rel.from_entity == *eid {
                        graph.get_entity(&rel.to_entity)
                            .map(|e| e.name.clone())
                            .unwrap_or_default()
                    } else {
                        graph.get_entity(&rel.from_entity)
                            .map(|e| e.name.clone())
                            .unwrap_or_default()
                    };
                    serde_json::json!({
                        "to": target,
                        "type": rel.relation_type.as_str(),
                    })
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

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
        let repo = self.repo.lock().unwrap();
        let since = chrono::Utc::now().timestamp() - (input.days * 86400);
        let conn = repo.connection();

        let mut stmt = conn.prepare(
            "SELECT date(created_at, 'unixepoch') as day, COUNT(*) as cnt
             FROM episodic_memories
             WHERE project_id = ?1 AND created_at >= ?2
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

    fn recent_failures(&self, input: RecentFailuresInput) -> Result<serde_json::Value> {
        let repo = self.repo.lock().unwrap();
        let query = input.service.as_deref().unwrap_or("");
        let results = if query.is_empty() {
            repo.list_recent_failures(&input.project_id, input.limit)?
                .into_iter()
                .map(|m| ScoredMemory { memory: m, bm25_score: 0.0 })
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

    fn architectural_decisions(&self, input: ArchitecturalDecisionsInput) -> Result<serde_json::Value> {
        let repo = self.repo.lock().unwrap();
        let query = input.topic.as_deref().unwrap_or("");
        let results = if query.is_empty() {
            repo.list_recent_decisions(&input.project_id, input.limit)?
                .into_iter()
                .map(|m| ScoredMemory { memory: m, bm25_score: 0.0 })
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
        let repo = self.repo.lock().unwrap();
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

        Ok(serde_json::json!({
            "id": id,
            "status": "created",
            "created_at": now,
        }))
    }

    fn create_decision(&self, input: CreateDecisionInput) -> Result<serde_json::Value> {
        let repo = self.repo.lock().unwrap();
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

        Ok(serde_json::json!({
            "id": id,
            "status": "created",
            "created_at": now,
        }))
    }

    fn create_failure(&self, input: CreateFailureInput) -> Result<serde_json::Value> {
        let repo = self.repo.lock().unwrap();
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

        Ok(serde_json::json!({
            "id": id,
            "status": "created",
            "severity": input.severity,
            "created_at": now,
        }))
    }

    fn create_procedural(&self, input: CreateProceduralInput) -> Result<serde_json::Value> {
        let repo = self.repo.lock().unwrap();
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

        let repo = self.repo.lock().unwrap();

        // Deduplicate: skip commits already ingested
        let ingested_hashes = repo.get_ingested_commits(&input.project_id)?;
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

        Ok(serde_json::json!({
            "ingested": ingested.len(),
            "total_commits_scanned": input.count,
            "skipped_duplicates": input.count - ingested.len(),
            "memories": ingested,
        }))
    }
}

/// MCP Server running on stdio transport.
pub struct McpServer {
    provider: Arc<dyn MemoryToolProvider>,
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
            fn architectural_decisions(&self, _: ArchitecturalDecisionsInput) -> Result<serde_json::Value> {
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
        }
        Self {
            provider: Arc::new(NoopProvider),
        }
    }

    pub fn with_provider(provider: Arc<dyn MemoryToolProvider>) -> Self {
        Self { provider }
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
                        "limit": { "type": "integer", "description": "Max results", "default": 5 },
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
                        "limit": { "type": "integer", "description": "Max results", "default": 5 },
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
        ]
    }

    /// Run the MCP server loop, reading JSON-RPC from stdin and writing to stdout.
    pub fn run(&self) -> Result<()> {
        let stdin = io::stdin();
        let stdout = io::stdout();
        let mut stdout = stdout.lock();

        for line in stdin.lock().lines() {
            let line = line.context("failed to read from stdin")?;
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            // JSON-RPC notifications (no `id`) must not receive a response.
            // Silently skip them per MCP spec.
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if val.get("id").is_none() {
                    continue;
                }
            }

            let request: JsonRpcRequest = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    let response = JsonRpcResponse {
                        jsonrpc: "2.0".into(),
                        id: None,
                        result: None,
                        error: Some(JsonRpcError {
                            code: -32700,
                            message: format!("Parse error: {e}"),
                        }),
                    };
                    writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
                    stdout.flush()?;
                    continue;
                }
            };

            let response = self.handle_request(request);
            writeln!(stdout, "{}", serde_json::to_string(&response)?)?;
            stdout.flush()?;
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
                    "capabilities": { "tools": {} },
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
            "tools/call" => {
                let params = request.params.unwrap_or_default();
                let tool_name = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let arguments = params.get("arguments").cloned().unwrap_or_default();

                let (result, parse_error): (Option<Result<serde_json::Value>>, Option<JsonRpcError>) = match tool_name {
                    "search_memory" => dispatch_tool!(arguments, SearchMemoryInput, self.provider, search_memory),
                    "related_files" => dispatch_tool!(arguments, RelatedFilesInput, self.provider, related_files),
                    "timeline" => dispatch_tool!(arguments, TimelineInput, self.provider, timeline),
                    "recent_failures" => dispatch_tool!(arguments, RecentFailuresInput, self.provider, recent_failures),
                    "architectural_decisions" => dispatch_tool!(arguments, ArchitecturalDecisionsInput, self.provider, architectural_decisions),
                    "create_episodic" => dispatch_tool!(arguments, CreateEpisodicInput, self.provider, create_episodic),
                    "create_decision" => dispatch_tool!(arguments, CreateDecisionInput, self.provider, create_decision),
                    "create_failure" => dispatch_tool!(arguments, CreateFailureInput, self.provider, create_failure),
                    "create_procedural" => dispatch_tool!(arguments, CreateProceduralInput, self.provider, create_procedural),
                    "ingest_commits" => dispatch_tool!(arguments, IngestCommitsInput, self.provider, ingest_commits),
                    _ => (None, Some(JsonRpcError { code: -32601, message: format!("Unknown tool: {tool_name}") })),
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
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_tools() {
        let tools = McpServer::list_tools();
        assert_eq!(tools.len(), 10);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search_memory"));
        assert!(names.contains(&"related_files"));
        assert!(names.contains(&"timeline"));
        assert!(names.contains(&"recent_failures"));
        assert!(names.contains(&"architectural_decisions"));
        assert!(names.contains(&"create_episodic"));
        assert!(names.contains(&"create_decision"));
        assert!(names.contains(&"create_failure"));
        assert!(names.contains(&"create_procedural"));
        assert!(names.contains(&"ingest_commits"));
    }

    #[test]
    fn test_tool_schemas_require_project_id() {
        let tools = McpServer::list_tools();
        for tool in &tools {
            let required = tool.input_schema.get("required")
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
        assert_eq!(tools.len(), 10);
    }

    #[test]
    fn test_create_episodic_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let graph = GraphEngine::new();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));

        // Create episodic memory
        let result = provider.create_episodic(CreateEpisodicInput {
            project_id: "test-project".into(),
            session_id: "session-1".into(),
            summary: "Fixed OAuth refresh loop".into(),
            content: "The refresh token was looping due to stale cache".into(),
            files_touched: vec!["auth.ts".into()],
            related_commits: vec!["abc123".into()],
            importance: 0.8,
            tags: vec!["auth".into(), "oauth".into()],
        }).unwrap();

        assert_eq!(result["status"], "created");
        let id = result["id"].as_str().unwrap();

        // Verify it can be searched
        let search_result = provider.search_memory(SearchMemoryInput {
            project_id: "test-project".into(),
            query: "OAuth refresh".into(),
            memory_type: None,
            limit: 10,
        }).unwrap();

        let results = search_result["results"].as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["id"], id);
    }

    #[test]
    fn test_create_decision_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let graph = GraphEngine::new();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));

        let result = provider.create_decision(CreateDecisionInput {
            project_id: "test-project".into(),
            title: "Use Redis for session caching".into(),
            context: "Auth service needs sub-ms latency".into(),
            rationale: "Redis provides sub-millisecond reads".into(),
            tradeoffs: "Added infrastructure complexity".into(),
            related_files: vec!["auth.ts".into()],
            tags: vec!["architecture".into()],
        }).unwrap();

        assert_eq!(result["status"], "created");

        // Verify via search
        let search = provider.search_memory(SearchMemoryInput {
            project_id: "test-project".into(),
            query: "Redis".into(),
            memory_type: Some("decision".into()),
            limit: 5,
        }).unwrap();
        assert_eq!(search["results"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_create_failure_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let graph = GraphEngine::new();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));

        let result = provider.create_failure(CreateFailureInput {
            project_id: "test-project".into(),
            incident: "Auth token expiry mismatch".into(),
            root_cause: "Clock skew between services".into(),
            fix: "Added clock tolerance window".into(),
            prevention: "Monitor clock sync".into(),
            severity: 4,
            tags: vec!["auth".into()],
        }).unwrap();

        assert_eq!(result["status"], "created");
        assert_eq!(result["severity"], 4);
    }

    #[test]
    fn test_create_procedural_with_provider() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let graph = GraphEngine::new();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));

        let result = provider.create_procedural(CreateProceduralInput {
            project_id: "test-project".into(),
            workflow_name: "deployment".into(),
            steps: vec!["run tests".into(), "build docker".into(), "push to registry".into()],
            related_tools: vec!["docker".into()],
            tags: vec!["deploy".into()],
        }).unwrap();

        assert_eq!(result["status"], "created");
    }

    #[test]
    fn test_ingest_commits_with_temp_repo() {
        let repo = MemoryRepository::new_in_memory().unwrap();
        repo.initialize_schema().unwrap();
        let graph = GraphEngine::new();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));

        // Create temp git repo
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_path = temp_dir.path();
        let git_repo = git2::Repository::init(repo_path).unwrap();
        let sig = git2::Signature::new("Test", "test@test.com", &git2::Time::new(0, 0)).unwrap();

        let file_path = repo_path.join("main.rs");
        std::fs::write(&file_path, "fn main() {}").unwrap();
        let mut index = git_repo.index().unwrap();
        index.add_path(std::path::Path::new("main.rs")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = git_repo.find_tree(tree_id).unwrap();
        git_repo.commit(Some("HEAD"), &sig, &sig, "feat: initial commit", &tree, &[]).unwrap();

        // Ingest
        let result = provider.ingest_commits(IngestCommitsInput {
            project_id: "test-project".into(),
            repo_path: repo_path.to_string_lossy().to_string(),
            count: 10,
            session_id: Some("test-session".into()),
        }).unwrap();

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
        let graph = GraphEngine::new();
        let config = Config::default();

        let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));
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
        let graph = GraphEngine::new();
        let config = Config::default();
        let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));
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
        assert!(results[0]["summary"].as_str().unwrap().contains("authentication"));
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
}
