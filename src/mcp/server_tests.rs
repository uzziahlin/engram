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
            limit: Some(10),
            before: None,
            tags: vec![],
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
            limit: Some(10),
            before: None,
            tags: vec![],
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
                limit: Some(10),
                before: None,
                tags: vec![],
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
    assert_eq!(tools.len(), 26);

    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(names.contains(&"search_memory"));
    assert!(names.contains(&"get_memory"));
    assert!(names.contains(&"related_files"));
    assert!(names.contains(&"timeline"));
    assert!(names.contains(&"recent_failures"));
    assert!(names.contains(&"architectural_decisions"));
    assert!(names.contains(&"distill_backlog"));
    assert!(names.contains(&"query_stats"));
    assert!(names.contains(&"create_episodic"));
    assert!(names.contains(&"create_decision"));
    assert!(names.contains(&"create_failure"));
    assert!(names.contains(&"create_procedural"));
    assert!(names.contains(&"ingest_commits"));
    assert!(names.contains(&"collect_sources"));
    assert!(names.contains(&"reflect"));
    assert!(names.contains(&"list_suggestions"));
    assert!(names.contains(&"confirm_suggestion"));
    assert!(names.contains(&"reject_suggestion"));
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
    // serverInfo.version must track Cargo.toml (was hardcoded 0.1.0).
    assert_eq!(result["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
}

#[test]
fn test_handle_ping_returns_empty_result() {
    let server = McpServer::new();
    let response = server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(9)),
        method: "ping".into(),
        params: None,
    });
    assert!(response.error.is_none(), "ping must not error");
    assert!(response.result.is_some(), "ping must answer with a result");
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
    // Every tool the dispatcher knows must appear exactly once. A hardcoded
    // exact count made every tool addition a test edit; uniqueness + coverage
    // of the known set catches the actual regressions (dupes/dropouts).
    let mut names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
    let total = names.len();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), total, "duplicate tool names in tools/list");
    for expected in [
        "search_memory",
        "get_memory",
        "related_files",
        "timeline",
        "recent_failures",
        "query_stats",
        "session_brief",
        "architectural_decisions",
        "distill_backlog",
        "create_episodic",
        "create_decision",
        "create_failure",
        "create_procedural",
        "ingest_commits",
        "collect_sources",
        "forget_memory",
        "restore_memory",
        "update_memory",
        "mark_relevance",
        "forget_batch",
        "list_archived",
        "consolidate_memories",
        "reflect",
        "list_suggestions",
        "confirm_suggestion",
        "reject_suggestion",
    ] {
        assert!(
            names.contains(&expected),
            "tool {expected} missing from tools/list"
        );
    }
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
            limit: Some(10),
            before: None,
            tags: vec![],
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

    let created = provider
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
    let created_id = created["id"].as_str().unwrap().to_string();

    let result = provider
        .related_files(RelatedFilesInput {
            project_id: "p".into(),
            file: "auth.ts".into(),
        })
        .unwrap();

    // The file resolves and is reported as found.
    assert_eq!(result["file"], "auth.ts");
    assert_eq!(result["found"], true);
    // The primary payload: the touching memory is CONSUMABLE — its type and
    // label are present, so an agent can act on it (get_memory) instead of
    // staring at a bare UUID (the pre-fix output was exactly that).
    let memories = result["memories"].as_array().unwrap();
    assert_eq!(memories.len(), 1);
    assert_eq!(memories[0]["memory_type"], "episodic");
    assert_eq!(memories[0]["id"], created_id);
    assert_eq!(memories[0]["summary"], "summary");
    assert_eq!(memories[0]["relation"], "Touches");
    // Graph edges remain visible for completeness.
    let related = result["related_entities"].as_array().unwrap();
    assert!(
        related.iter().any(|r| r["relation"] == "Touches"),
        "expected a Touches relation, got: {related:?}"
    );

    // An unknown file reports not-found explicitly (not an empty success).
    let missing = provider
        .related_files(RelatedFilesInput {
            project_id: "p".into(),
            file: "nope.rs".into(),
        })
        .unwrap();
    assert_eq!(missing["found"], false);
    assert!(missing["detail"].is_string());
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
            limit: Some(10),
            before: None,
            tags: vec![],
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
            importance: None,
            supersedes: None,
        })
        .unwrap();

    assert_eq!(result["status"], "created");

    // Verify via search
    let search = provider
        .search_memory(SearchMemoryInput {
            project_id: "test-project".into(),
            query: "Redis".into(),
            memory_type: Some("decision".into()),
            limit: Some(5),
            before: None,
            tags: vec![],
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
            importance: None,
        })
        .unwrap();

    assert_eq!(result["status"], "created");
}

#[test]
fn test_reflection_reflect_list_confirm_closed_loop() {
    let provider = make_provider();
    let project = "test-project";

    // Seed three auth failures → one recurring tag. Each occurrence is a
    // DISTINCT incident (byte-identical repeats are stopped by the
    // write-time dedup guard — recurrence is about the shared tag).
    for i in 0..3 {
        provider
            .create_failure(CreateFailureInput {
                project_id: project.into(),
                incident: format!("token expired (occurrence {i})"),
                root_cause: "clock skew".into(),
                fix: "tolerance window".into(),
                prevention: "monitor clock sync".into(),
                severity: 3,
                tags: vec!["auth".into()],
            })
            .unwrap();
    }

    // Dry-run reflect: previews one proposal, writes nothing.
    let dry = provider
        .reflect(ReflectInput {
            project_id: project.into(),
            apply: false,
            min_occurrences: None,
        })
        .unwrap();
    assert_eq!(dry["proposed"], 1);
    assert_eq!(dry["created"], 0);
    assert_eq!(dry["suggestions"][0]["pattern_tag"], "auth");

    // Nothing persisted yet.
    let none = provider
        .list_suggestions(ListSuggestionsInput {
            project_id: project.into(),
        })
        .unwrap();
    assert_eq!(none["count"], 0);

    // Apply: persists one pending proposal.
    let applied = provider
        .reflect(ReflectInput {
            project_id: project.into(),
            apply: true,
            min_occurrences: None,
        })
        .unwrap();
    assert_eq!(applied["created"], 1);

    let pending = provider
        .list_suggestions(ListSuggestionsInput {
            project_id: project.into(),
        })
        .unwrap();
    assert_eq!(pending["count"], 1);
    let id = pending["suggestions"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();

    // Core acceptance: a pending proposal is invisible to search_procedural.
    let search_before = provider
        .search_memory(SearchMemoryInput {
            project_id: project.into(),
            query: "auth".into(),
            memory_type: Some("procedural".into()),
            limit: Some(10),
            before: None,
            tags: vec![],
        })
        .unwrap();
    assert_eq!(search_before["results"].as_array().unwrap().len(), 0);

    // Confirm → promoted into procedural_memories, now searchable.
    let confirmed = provider
        .confirm_suggestion(SuggestionIdInput {
            project_id: project.into(),
            id: id.clone(),
        })
        .unwrap();
    assert_eq!(confirmed["status"], "confirmed");

    let search_after = provider
        .search_memory(SearchMemoryInput {
            project_id: project.into(),
            query: "auth".into(),
            memory_type: Some("procedural".into()),
            limit: Some(10),
            before: None,
            tags: vec![],
        })
        .unwrap();
    assert_eq!(search_after["results"].as_array().unwrap().len(), 1);

    // No longer pending after confirmation.
    let after = provider
        .list_suggestions(ListSuggestionsInput {
            project_id: project.into(),
        })
        .unwrap();
    assert_eq!(after["count"], 0);
}

#[test]
fn test_reflection_reject_drops_pending() {
    let provider = make_provider();
    let project = "test-project";
    for i in 0..3 {
        provider
            .create_failure(CreateFailureInput {
                project_id: project.into(),
                incident: format!("x (occurrence {i})"),
                root_cause: "y".into(),
                fix: "z".into(),
                prevention: "rotate token".into(),
                severity: 2,
                tags: vec!["auth".into()],
            })
            .unwrap();
    }
    provider
        .reflect(ReflectInput {
            project_id: project.into(),
            apply: true,
            min_occurrences: None,
        })
        .unwrap();
    let pending = provider
        .list_suggestions(ListSuggestionsInput {
            project_id: project.into(),
        })
        .unwrap();
    let id = pending["suggestions"][0]["id"]
        .as_str()
        .unwrap()
        .to_string();

    let rejected = provider
        .reject_suggestion(SuggestionIdInput {
            project_id: project.into(),
            id,
        })
        .unwrap();
    assert_eq!(rejected["status"], "rejected");

    // Pending drained; no procedural created.
    let after = provider
        .list_suggestions(ListSuggestionsInput {
            project_id: project.into(),
        })
        .unwrap();
    assert_eq!(after["count"], 0);
    let search = provider
        .search_memory(SearchMemoryInput {
            project_id: project.into(),
            query: "auth".into(),
            memory_type: Some("procedural".into()),
            limit: Some(10),
            before: None,
            tags: vec![],
        })
        .unwrap();
    assert_eq!(search["results"].as_array().unwrap().len(), 0);
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
fn search_with_star_project_spans_projects() {
    // project_id = '*' is the global scope: knowledge in project A must
    // surface for an agent working in project B.
    let provider = make_provider();
    let repo = provider.repo_for_test();
    let mk = |id: &str, project: &str| crate::models::EpisodicMemory {
        id: id.into(),
        project_id: project.into(),
        session_id: "s".into(),
        summary: format!("global marker {id} GLOBALSEARCHX"),
        content: "c".into(),
        files_touched: vec![],
        related_commits: vec![],
        importance: 0.5,
        tags: vec![],
        created_at: 1,
        updated_at: 1,
    };
    repo.create_episodic(&mk("ga", "proj-a")).unwrap();
    repo.create_episodic(&mk("gb", "proj-b")).unwrap();

    let resp = provider
        .search_memory(SearchMemoryInput {
            project_id: "*".into(),
            query: "GLOBALSEARCHX".into(),
            limit: Some(10),
            memory_type: None,
            tags: vec![],
            before: None,
        })
        .unwrap();
    let results = resp.get("results").and_then(|r| r.as_array()).unwrap();
    assert_eq!(results.len(), 2, "both projects' memories must surface");

    // Project-scoped search still isolates.
    let scoped = provider
        .search_memory(SearchMemoryInput {
            project_id: "proj-a".into(),
            query: "GLOBALSEARCHX".into(),
            limit: Some(10),
            memory_type: None,
            tags: vec![],
            before: None,
        })
        .unwrap();
    let scoped_results = scoped.get("results").and_then(|r| r.as_array()).unwrap();
    assert_eq!(scoped_results.len(), 1);
}

#[test]
fn search_surfaces_graph_neighbors() {
    // Two memories touching the same file are graph-adjacent: searching
    // for one must surface the other as a low-rank `graph_neighbor` even
    // though its text doesn't match the query.
    let provider = make_provider();
    let repo = provider.repo_for_test();

    let mk = |id: &str, summary: &str, files: Vec<&str>| crate::models::EpisodicMemory {
        id: id.into(),
        project_id: "p".into(),
        session_id: "s".into(),
        summary: summary.into(),
        content: summary.into(),
        files_touched: files.into_iter().map(String::from).collect(),
        related_commits: vec![],
        importance: 0.5,
        tags: vec![],
        created_at: 1,
        updated_at: 1,
    };
    repo.create_episodic(&mk(
        "g1",
        "authentication refactor SESSIONX",
        vec!["auth.rs"],
    ))
    .unwrap();
    repo.create_episodic(&mk("g2", "rate limiter UNRELATED", vec!["auth.rs"]))
        .unwrap();

    let resp = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "authentication refactor".into(),
            limit: None,
            memory_type: None,
            tags: vec![],
            before: None,
        })
        .unwrap();
    let results = resp.get("results").and_then(|r| r.as_array()).unwrap();
    assert!(!results.is_empty());
    let neighbor = results
        .iter()
        .find(|r| r.get("id").and_then(|i| i.as_str()) == Some("g2"))
        .expect("graph-adjacent memory must surface");
    assert_eq!(
        neighbor.get("graph_neighbor").and_then(|v| v.as_bool()),
        Some(true)
    );
}

/// Wire a memory into the entity graph against a File entity another memory
/// created via `files_touched`. Failures link no entities on the MCP create
/// path, so a failure sharing a file with an episodic is constructed here
/// explicitly (the graph linker can produce such edges for failures recorded
/// against a file).
fn link_memory_to_file(repo: &MemoryRepository, project: &str, memory_id: &str, file: &str) {
    let (file_eid, _) = repo
        .related_files_for(file, project)
        .expect("file entity must exist (created via files_touched)");
    let file_eid = file_eid.expect("File entity id for the shared file");
    repo.create_entity(&crate::models::Entity {
        id: memory_id.into(),
        project_id: project.into(),
        entity_type: crate::models::EntityType::Memory,
        name: memory_id.into(),
        metadata: serde_json::json!({}),
        created_at: 1,
        updated_at: 1,
    })
    .unwrap();
    repo.create_relation(&crate::models::GraphRelation {
        id: 0, // auto-increment (INTEGER PRIMARY KEY)
        project_id: Some(project.into()),
        from_entity: memory_id.into(),
        to_entity: file_eid,
        relation_type: crate::models::RelationType::Touches,
        weight: 1.0,
        created_at: 1,
    })
    .unwrap();
}

#[test]
fn global_search_results_carry_project_id_for_followup() {
    // A `project_id = "*"` hit must travel with its OWNING project: the
    // follow-up tools (get_memory / mark_relevance / forget_memory) all
    // require the exact project, so a cross-project hit without project_id
    // was a dead end — visible but not actionable.
    let provider = make_provider();
    let repo = provider.repo_for_test();
    let mk = |id: &str, project: &str| crate::models::EpisodicMemory {
        id: id.into(),
        project_id: project.into(),
        session_id: "s".into(),
        summary: format!("global marker {id} PROJFIELDX"),
        content: "c".into(),
        files_touched: vec![],
        related_commits: vec![],
        importance: 0.5,
        tags: vec![],
        created_at: 1,
        updated_at: 1,
    };
    repo.create_episodic(&mk("pa", "proj-a")).unwrap();
    repo.create_episodic(&mk("pb", "proj-b")).unwrap();

    let resp = provider
        .search_memory(SearchMemoryInput {
            project_id: "*".into(),
            query: "PROJFIELDX".into(),
            limit: Some(10),
            memory_type: None,
            tags: vec![],
            before: None,
        })
        .unwrap();
    let results = resp["results"].as_array().unwrap();
    assert_eq!(results.len(), 2, "both projects' memories must surface");

    let mut owner_of = std::collections::HashMap::new();
    for r in results {
        let pid = r["project_id"]
            .as_str()
            .unwrap_or_else(|| panic!("global hit must carry its owning project_id: {r}"));
        owner_of.insert(r["id"].as_str().unwrap().to_string(), pid.to_string());
    }
    assert_eq!(
        owner_of.get("pa").map(String::as_str),
        Some("proj-a"),
        "project_id must be the memory's owner, not the '*' wildcard"
    );
    assert_eq!(owner_of.get("pb").map(String::as_str), Some("proj-b"));

    // The returned project_id is exactly what get_memory needs: the
    // cross-project hit round-trips with it.
    let fetched = provider
        .get_memory(GetMemoryInput {
            project_id: owner_of["pb"].clone(),
            memory_type: "episodic".into(),
            id: "pb".into(),
        })
        .unwrap();
    assert_eq!(fetched["memory"]["summary"], "global marker pb PROJFIELDX");
}

#[test]
fn graph_neighbors_respect_memory_type_filter() {
    // An episodic sharing a File entity with a failure hit must NOT surface
    // in a memory_type="failure" search — graph expansion used to append
    // neighbors without checking the caller's explicit filters.
    let provider = make_provider();
    let repo = provider.repo_for_test();

    // Graph-neighbor candidate: episodic touching shared.rs, text unrelated
    // to the query.
    repo.create_episodic(&crate::models::EpisodicMemory {
        id: "e-nb".into(),
        project_id: "p".into(),
        session_id: "s".into(),
        summary: "auth refactor notes".into(),
        content: "c".into(),
        files_touched: vec!["shared.rs".into()],
        related_commits: vec![],
        importance: 0.5,
        tags: vec![],
        created_at: 1,
        updated_at: 1,
    })
    .unwrap();
    // The failure hit: matches the query and is linked to the same file.
    repo.create_failure(&crate::models::FailureMemory {
        id: "f-hit".into(),
        project_id: "p".into(),
        incident: "crash on boot TYPEFILTERX".into(),
        root_cause: "rc".into(),
        fix: "fx".into(),
        prevention: "pv".into(),
        severity: 3,
        tags: vec![],
        created_at: 2,
        updated_at: 2,
    })
    .unwrap();
    link_memory_to_file(repo, "p", "f-hit", "shared.rs");

    // Sanity: the shared-file edge is real — the failure's graph neighbors
    // include the episodic (guards against the test passing vacuously
    // because the edge was never wired).
    let nb = repo
        .neighbor_memories("p", &["f-hit".to_string()], 2)
        .unwrap();
    assert_eq!(
        nb,
        vec![("episodic".to_string(), "e-nb".to_string())],
        "scenario must actually produce a foreign-type neighbor"
    );

    let resp = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "TYPEFILTERX".into(),
            memory_type: Some("failure".into()),
            limit: Some(10),
            tags: vec![],
            before: None,
        })
        .unwrap();
    let results = resp["results"].as_array().unwrap();
    assert!(
        results.iter().any(|r| r["id"] == "f-hit"),
        "the failure hit must remain: {resp}"
    );
    for r in results {
        if r["graph_neighbor"].as_bool() == Some(true) {
            assert_eq!(
                r["memory_type"], "failure",
                "typed search must not leak foreign-type graph neighbors: {r}"
            );
        }
    }
    assert!(
        results.iter().all(|r| r["id"] != "e-nb"),
        "the episodic neighbor must be filtered out of the failure search: {resp}"
    );
}

#[test]
fn graph_neighbors_respect_tag_filter() {
    // The tags filter applies to graph neighbors with the same semantics as
    // the vector path: a neighbor carrying none of the requested tags is
    // dropped, not appended as "context".
    let provider = make_provider();
    let repo = provider.repo_for_test();
    let mk = |id: &str, summary: &str, tags: Vec<&str>| crate::models::EpisodicMemory {
        id: id.into(),
        project_id: "p".into(),
        session_id: "s".into(),
        summary: summary.into(),
        content: summary.into(),
        files_touched: vec!["shared.rs".into()],
        related_commits: vec![],
        importance: 0.5,
        tags: tags.into_iter().map(String::from).collect(),
        created_at: 1,
        updated_at: 1,
    };
    // e-hit matches the query and carries "auth"; e-nb shares the file but
    // carries only "deploy".
    repo.create_episodic(&mk("e-hit", "login flow TAGFILTERX", vec!["auth"]))
        .unwrap();
    repo.create_episodic(&mk("e-nb", "unrelated deploy work", vec!["deploy"]))
        .unwrap();

    // Sanity: without the filter the neighbor does surface — graph
    // expansion is live in this scenario.
    let unfiltered = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "TAGFILTERX".into(),
            limit: Some(10),
            memory_type: None,
            tags: vec![],
            before: None,
        })
        .unwrap();
    assert!(
        unfiltered["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["id"] == "e-nb" && r["graph_neighbor"] == serde_json::json!(true)),
        "neighbor must surface without filters (scenario sanity): {unfiltered}"
    );

    // With tags=["auth"], e-nb must be gone — including as a neighbor.
    let filtered = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "TAGFILTERX".into(),
            limit: Some(10),
            memory_type: None,
            tags: vec!["auth".into()],
            before: None,
        })
        .unwrap();
    let arr = filtered["results"].as_array().unwrap();
    assert!(
        arr.iter().any(|r| r["id"] == "e-hit"),
        "the true hit must remain: {filtered}"
    );
    assert!(
        arr.iter().all(|r| r["id"] != "e-nb"),
        "neighbor without any requested tag must be filtered: {filtered}"
    );
}

#[test]
fn graph_neighbors_respect_before_filter() {
    // Same for the `before` cutoff: a neighbor created AFTER the cutoff must
    // not leak into a point-in-time search — not even as context.
    let provider = make_provider();
    let repo = provider.repo_for_test();
    let mk = |id: &str, summary: &str, created_at: i64| crate::models::EpisodicMemory {
        id: id.into(),
        project_id: "p".into(),
        session_id: "s".into(),
        summary: summary.into(),
        content: summary.into(),
        files_touched: vec!["shared.rs".into()],
        related_commits: vec![],
        importance: 0.5,
        tags: vec![],
        created_at,
        updated_at: created_at,
    };
    repo.create_episodic(&mk("e-old", "legacy auth flow BEFOREX", 100))
        .unwrap();
    repo.create_episodic(&mk("e-new", "unrelated recent work", 5_000))
        .unwrap();

    // Sanity: without `before` the newer neighbor surfaces.
    let unfiltered = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "BEFOREX".into(),
            limit: Some(10),
            memory_type: None,
            tags: vec![],
            before: None,
        })
        .unwrap();
    assert!(
        unfiltered["results"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r["id"] == "e-new" && r["graph_neighbor"] == serde_json::json!(true)),
        "neighbor must surface without filters (scenario sanity): {unfiltered}"
    );

    let filtered = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "BEFOREX".into(),
            limit: Some(10),
            memory_type: None,
            tags: vec![],
            before: Some(200),
        })
        .unwrap();
    let arr = filtered["results"].as_array().unwrap();
    assert!(
        arr.iter().any(|r| r["id"] == "e-old"),
        "the pre-cutoff hit must remain: {filtered}"
    );
    assert!(
        arr.iter().all(|r| r["id"] != "e-new"),
        "neighbor newer than `before` must be filtered: {filtered}"
    );
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
    // MCP spec: an unknown tool name is an invalid parameter of tools/call,
    // not a missing method (tools/call itself exists) — -32602, not -32601.
    let err = response.error.unwrap();
    assert_eq!(err.code, -32602);
    assert!(err.message.contains("unknown tool"));
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

    // Create an episodic memory and a decision on the same day.
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
    server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(2)),
        method: "tools/call".into(),
        params: Some(serde_json::json!({
            "name": "create_decision",
            "arguments": {
                "project_id": "timeline-test",
                "title": "Use SQLite",
                "context": "single-file storage",
                "rationale": "zero ops",
                "tradeoffs": "no concurrent writers"
            }
        })),
    });

    let timeline_resp = server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(3)),
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
    assert_eq!(timeline_parsed["project_id"], "timeline-test");
    let events = timeline_parsed["events"].as_array().unwrap();
    assert!(!events.is_empty());
    // Cross-type counts: every row carries the full four-type shape.
    let counts = &events[0]["counts"];
    assert!(counts["episodic"].as_i64().unwrap() >= 1);
    assert!(counts["decision"].as_i64().unwrap() >= 1);
    assert_eq!(
        counts["total"].as_i64().unwrap(),
        counts["episodic"].as_i64().unwrap()
            + counts["decision"].as_i64().unwrap()
            + counts["failure"].as_i64().unwrap()
            + counts["procedural"].as_i64().unwrap()
    );
}

// ─── timeline: cross-type grouping, filters, until ─────────────────

/// Seed one memory of the given kind at an exact timestamp (direct storage
/// writes so the tests control the day bucketing precisely).
fn seed_timeline_memory(
    repo: &MemoryRepository,
    kind: MemoryKind,
    id: &str,
    project: &str,
    created_at: i64,
) {
    match kind {
        MemoryKind::Episodic => repo
            .create_episodic(&EpisodicMemory {
                id: id.into(),
                project_id: project.into(),
                session_id: "s".into(),
                summary: format!("episodic {id}"),
                content: format!("content {id}"),
                files_touched: vec![],
                related_commits: vec![],
                importance: 0.5,
                tags: vec![],
                created_at,
                updated_at: created_at,
            })
            .unwrap(),
        MemoryKind::Decision => repo
            .create_decision(&DecisionMemory {
                id: id.into(),
                project_id: project.into(),
                title: format!("decision {id}"),
                context: "ctx".into(),
                rationale: "rationale".into(),
                tradeoffs: "tradeoffs".into(),
                related_files: vec![],
                tags: vec![],
                importance: 0.5,
                superseded_by: None,
                created_at,
                updated_at: created_at,
            })
            .unwrap(),
        MemoryKind::Failure => repo
            .create_failure(&FailureMemory {
                id: id.into(),
                project_id: project.into(),
                incident: format!("incident {id}"),
                root_cause: "root cause".into(),
                fix: "fix".into(),
                prevention: "prevention".into(),
                severity: 3,
                tags: vec![],
                created_at,
                updated_at: created_at,
            })
            .unwrap(),
        MemoryKind::Procedural => repo
            .create_procedural(&ProceduralMemory {
                id: id.into(),
                project_id: project.into(),
                workflow_name: format!("workflow {id}"),
                steps: vec!["step".into()],
                related_tools: vec![],
                tags: vec![],
                importance: 0.5,
                created_at,
                updated_at: created_at,
            })
            .unwrap(),
    }
}

fn timeline_output(
    provider: &DefaultMemoryProvider,
    project: &str,
    days: i64,
    memory_type: Option<&str>,
    until: Option<i64>,
) -> serde_json::Value {
    provider
        .timeline(TimelineInput {
            project_id: project.into(),
            days,
            memory_type: memory_type.map(|s| s.into()),
            until,
        })
        .unwrap()
}

#[test]
fn timeline_groups_all_types_per_day_most_recent_first() {
    let provider = make_provider();
    let now = chrono::Utc::now().timestamp();
    let today = now - 60;
    let yesterday = now - 86_400 - 60;

    // Today: 2 episodic, 1 decision, 1 failure, 1 procedural.
    for (kind, id) in [
        (MemoryKind::Episodic, "e1"),
        (MemoryKind::Episodic, "e2"),
        (MemoryKind::Decision, "d1"),
        (MemoryKind::Failure, "f1"),
        (MemoryKind::Procedural, "p1"),
    ] {
        seed_timeline_memory(provider.repo_for_test(), kind, id, "tl", today);
    }
    // Yesterday: 1 episodic only.
    seed_timeline_memory(
        provider.repo_for_test(),
        MemoryKind::Episodic,
        "e0",
        "tl",
        yesterday,
    );

    let out = timeline_output(&provider, "tl", 7, None, None);
    assert_eq!(out["project_id"], "tl");
    assert_eq!(out["days"], 7);
    assert!(out.get("memory_type").is_none());

    let events = out["events"].as_array().unwrap();
    assert_eq!(events.len(), 2, "one row per active day");
    // Most recent day first.
    let today_str = chrono::DateTime::from_timestamp(today, 0)
        .unwrap()
        .format("%Y-%m-%d")
        .to_string();
    let yesterday_str = chrono::DateTime::from_timestamp(yesterday, 0)
        .unwrap()
        .format("%Y-%m-%d")
        .to_string();
    assert_eq!(events[0]["day"], today_str.as_str());
    assert_eq!(events[1]["day"], yesterday_str.as_str());

    let counts = &events[0]["counts"];
    assert_eq!(counts["episodic"], 2);
    assert_eq!(counts["decision"], 1);
    assert_eq!(counts["failure"], 1);
    assert_eq!(counts["procedural"], 1);
    assert_eq!(counts["total"], 5);

    let counts = &events[1]["counts"];
    assert_eq!(counts["episodic"], 1);
    assert_eq!(counts["decision"], 0);
    assert_eq!(counts["failure"], 0);
    assert_eq!(counts["procedural"], 0);
    assert_eq!(counts["total"], 1);
}

#[test]
fn timeline_memory_type_filter_keeps_shape_zeroes_others() {
    let provider = make_provider();
    let now = chrono::Utc::now().timestamp();
    let today = now - 60;
    for (kind, id) in [
        (MemoryKind::Episodic, "e1"),
        (MemoryKind::Decision, "d1"),
        (MemoryKind::Failure, "f1"),
    ] {
        seed_timeline_memory(provider.repo_for_test(), kind, id, "tl", today);
    }

    let out = timeline_output(&provider, "tl", 7, Some("decision"), None);
    assert_eq!(out["memory_type"], "decision");
    let events = out["events"].as_array().unwrap();
    assert_eq!(events.len(), 1, "days without a decision disappear");
    let counts = &events[0]["counts"];
    assert_eq!(counts["decision"], 1);
    assert_eq!(counts["episodic"], 0);
    assert_eq!(counts["failure"], 0);
    assert_eq!(counts["procedural"], 0);
    assert_eq!(counts["total"], 1);
}

#[test]
fn timeline_until_bounds_the_window_end() {
    let provider = make_provider();
    let now = chrono::Utc::now().timestamp();
    let inside = now - 3_600;
    let far = now + 5 * 86_400; // 5 days out — always a different UTC day.
    seed_timeline_memory(
        provider.repo_for_test(),
        MemoryKind::Episodic,
        "in",
        "tl",
        inside,
    );
    seed_timeline_memory(
        provider.repo_for_test(),
        MemoryKind::Episodic,
        "out",
        "tl",
        far,
    );

    // No until: unbounded window also surfaces the future-dated day.
    let events = timeline_output(&provider, "tl", 7, None, None)["events"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(events.len(), 2);

    // until = far (exclusive): the far event sits exactly on the boundary —
    // excluded.
    let events = timeline_output(&provider, "tl", 7, None, Some(far))["events"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["counts"]["episodic"], 1);

    // One second past the boundary lets it back in.
    let events = timeline_output(&provider, "tl", 7, None, Some(far + 1))["events"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(events.len(), 2);
}

#[test]
fn timeline_excludes_archived_memories() {
    let provider = make_provider();
    let now = chrono::Utc::now().timestamp();
    seed_timeline_memory(
        provider.repo_for_test(),
        MemoryKind::Episodic,
        "gone",
        "tl",
        now - 60,
    );
    provider
        .repo_for_test()
        .archive(MemoryKind::Episodic, "gone", "tl", now)
        .unwrap();

    let out = timeline_output(&provider, "tl", 7, None, None);
    assert_eq!(out["events"].as_array().unwrap().len(), 0);
}

#[test]
fn timeline_invalid_memory_type_is_a_clear_error() {
    let provider = make_provider();
    let err = provider
        .timeline(TimelineInput {
            project_id: "tl".into(),
            days: 7,
            memory_type: Some("bogus".into()),
            until: None,
        })
        .unwrap_err();
    assert!(err.to_string().contains("invalid memory_type"));
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
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0]["name"], "engram.bootstrap");
    assert_eq!(prompts[1]["name"], "engram.distill");
    let args = prompts[0]["arguments"].as_array().unwrap();
    assert!(args
        .iter()
        .any(|a| a["name"] == "project_id" && a["required"] == true));
}

#[test]
fn test_prompts_get_renders_distill_template() {
    let server = McpServer::new();
    let response = server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(1)),
        method: "prompts/get".into(),
        params: Some(serde_json::json!({
            "name": "engram.distill",
            "arguments": { "project_id": "myproj" }
        })),
    });
    assert!(response.error.is_none(), "got {response:?}");
    let result = response.result.unwrap();
    let text = result["messages"][0]["content"]["text"].as_str().unwrap();
    assert!(text.contains("myproj"), "project id must be substituted");
    assert!(text.contains("session-import"));
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
            limit: Some(10),
            before: None,
            tags: vec![],
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
            limit: Some(10),
            before: None,
            tags: vec![],
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
                limit: Some(10)
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
            limit: Some(10),
        })
        .unwrap();
    assert_eq!(listed["archived"].as_array().unwrap().len(), 2);
}

#[test]
fn test_consolidate_memories_tool_dry_run() {
    let provider = make_provider();
    // Seed through the repo: the create_* tools now stop byte-identical
    // duplicates at write time, but consolidation must still clean up exact
    // dups written before that guard existed (legacy stores, repo-level writers).
    for i in 0..2 {
        let now = chrono::Utc::now().timestamp() + i;
        provider
            .repo_for_test()
            .create_episodic(&crate::models::EpisodicMemory {
                id: format!("dup-{i}"),
                project_id: "p".into(),
                session_id: "s".into(),
                summary: "dup".into(),
                content: "same".into(),
                files_touched: vec![],
                related_commits: vec![],
                importance: 0.5,
                tags: vec![],
                created_at: now,
                updated_at: now,
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
                limit: Some(10),
                before: None,
                tags: vec![],
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

#[test]
fn get_memory_returns_full_failure_record() {
    let repo = MemoryRepository::new_in_memory().unwrap();
    repo.initialize_schema().unwrap();
    let config = Config::default();
    let provider = DefaultMemoryProvider::new(repo, config);

    let created = provider
        .create_failure(CreateFailureInput {
            project_id: "p".into(),
            incident: "auth outage".into(),
            root_cause: "stale token cache".into(),
            fix: "invalidate on refresh".into(),
            prevention: "cache TTL < token TTL".into(),
            severity: 4,
            tags: vec!["auth".into()],
        })
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();

    let fetched = provider
        .get_memory(GetMemoryInput {
            project_id: "p".into(),
            memory_type: "failure".into(),
            id: id.clone(),
        })
        .unwrap();
    assert_eq!(fetched["memory_type"], "failure");
    assert_eq!(fetched["memory"]["root_cause"], "stale token cache");
    assert_eq!(fetched["memory"]["prevention"], "cache TTL < token TTL");

    // Wrong project must be rejected.
    assert!(provider
        .get_memory(GetMemoryInput {
            project_id: "other".into(),
            memory_type: "failure".into(),
            id,
        })
        .is_err());
}

#[test]
fn search_memory_carries_full_detail_and_supports_tag_filter() {
    let repo = MemoryRepository::new_in_memory().unwrap();
    repo.initialize_schema().unwrap();
    let config = Config::default();
    let provider = DefaultMemoryProvider::new(repo, config);

    provider
        .create_failure(CreateFailureInput {
            project_id: "p".into(),
            incident: "crash on startup".into(),
            root_cause: "null config".into(),
            fix: "guard clause".into(),
            prevention: "tests for config paths".into(),
            severity: 3,
            tags: vec!["bootstrap".into()],
        })
        .unwrap();
    provider
        .create_failure(CreateFailureInput {
            project_id: "p".into(),
            incident: "crash on shutdown".into(),
            root_cause: "double free".into(),
            fix: "drop order".into(),
            prevention: "miri".into(),
            severity: 2,
            tags: vec!["lifecycle".into()],
        })
        .unwrap();

    // No filter: full payload on every hit.
    let res = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "crash".into(),
            memory_type: None,
            limit: Some(10),
            tags: vec![],
            before: None,
        })
        .unwrap();
    assert_eq!(res["results"].as_array().unwrap().len(), 2);
    for r in res["results"].as_array().unwrap() {
        assert!(
            r.get("root_cause").is_some() && r.get("fix").is_some(),
            "search results must carry full structured fields: {r}"
        );
    }

    // Tag filter narrows to the tagged memory only.
    let res = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "crash".into(),
            memory_type: None,
            limit: Some(10),
            tags: vec!["bootstrap".into()],
            before: None,
        })
        .unwrap();
    let arr = res["results"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["summary"], "crash on startup");
}

// ─── session_brief ──────────────────────────────────────────────────

#[test]
fn session_brief_assembles_all_sections() {
    let provider = make_provider();
    provider
        .create_decision(CreateDecisionInput {
            project_id: "p".into(),
            title: "Use SQLite".into(),
            context: "local-first".into(),
            rationale: "zero config. No server needed.".into(),
            tradeoffs: "single writer".into(),
            related_files: vec![],
            tags: vec![],
            importance: None,
            supersedes: None,
        })
        .unwrap();
    provider
        .create_failure(CreateFailureInput {
            project_id: "p".into(),
            incident: "FTS5 crash on keywords".into(),
            root_cause: "unescaped MATCH".into(),
            fix: "phrase quoting".into(),
            prevention: "always sanitize".into(),
            severity: 4,
            tags: vec![],
        })
        .unwrap();
    provider
        .create_procedural(CreateProceduralInput {
            project_id: "p".into(),
            workflow_name: "Release".into(),
            steps: vec!["test".into(), "tag".into(), "push".into()],
            related_tools: vec![],
            tags: vec![],
            importance: None,
        })
        .unwrap();

    let out = provider
        .session_brief(SessionBriefInput {
            project_id: "p".into(),
            focus: Some("FTS5 crash".into()),
        })
        .unwrap();

    let brief = out["brief"].as_str().unwrap();
    assert!(brief.contains("Use SQLite"), "decision missing: {brief}");
    assert!(
        brief.contains("FTS5 crash on keywords"),
        "failure missing: {brief}"
    );
    assert!(brief.contains("Release"), "procedure missing: {brief}");
    assert!(
        brief.contains("Focused context"),
        "focus section missing: {brief}"
    );
    assert_eq!(out["decisions"].as_array().unwrap().len(), 1);
    assert_eq!(out["failures"].as_array().unwrap().len(), 1);
    assert_eq!(out["procedures"].as_array().unwrap().len(), 1);
    assert!(
        out["focused"].as_array().is_some_and(|a| !a.is_empty()),
        "focused search must surface the failure: {out}"
    );

    // One decision entry stays one line: the rationale's second sentence is
    // dropped in the text brief but kept in the structured section.
    assert!(
        !brief.contains("No server needed"),
        "rationale not truncated: {brief}"
    );
    assert!(out["decisions"][0]["rationale"]
        .as_str()
        .is_some_and(|r| r.contains("No server needed")));
}

#[test]
fn session_brief_empty_project_returns_frame_only() {
    let provider = make_provider();
    let out = provider
        .session_brief(SessionBriefInput {
            project_id: "ghost".into(),
            focus: None,
        })
        .unwrap();
    let brief = out["brief"].as_str().unwrap();
    assert!(brief.contains("=== Engram Session Brief: ghost ==="));
    assert!(brief.ends_with("=== End Brief ==="));
    assert_eq!(out["decisions"].as_array().unwrap().len(), 0);
    assert!(out["focused"].is_null());
}

// ─── distill_backlog ─────────────────────────────────────────────────

/// Build session-import shaped content the same way
/// `cli::commands::session_import_content` stitches it: fixed template
/// headers, and the Files/Errors sections only when something was recorded.
fn session_import_content_for(id: &str, files: Option<usize>, errors: Option<usize>) -> String {
    let mut content = format!(
        "User prompts:\n- prompt {id}\n\n\
         Assistant conclusions (latest last):\n- conclusion {id}.\n"
    );
    if let Some(n) = files {
        content.push_str(&format!("\nFiles touched ({n}):\nsrc/{id}.rs\n"));
    }
    if let Some(n) = errors {
        content.push_str(&format!(
            "\nErrors encountered ({n} — failure-memory candidates):\n- error {id}\n"
        ));
    }
    content
}

fn seed_session_import(
    repo: &MemoryRepository,
    id: &str,
    tags: &[&str],
    created_at: i64,
    files: Option<usize>,
    errors: Option<usize>,
) {
    repo.create_episodic(&EpisodicMemory {
        id: id.into(),
        project_id: "p".into(),
        session_id: "s".into(),
        summary: format!("session {id}"),
        content: session_import_content_for(id, files, errors),
        files_touched: vec![],
        related_commits: vec![],
        importance: 0.5,
        tags: tags.iter().map(|t| t.to_string()).collect(),
        created_at,
        updated_at: created_at,
    })
    .unwrap();
}

#[test]
fn distill_backlog_lists_raw_oldest_first_and_excludes_distilled() {
    let provider = make_provider();
    let now = chrono::Utc::now().timestamp();
    seed_session_import(
        provider.repo_for_test(),
        "raw-old",
        &["session-import"],
        now - 3 * 86_400,
        Some(2),
        Some(1),
    );
    seed_session_import(
        provider.repo_for_test(),
        "raw-new",
        &["session-import"],
        now,
        None,
        None,
    );
    // The typed memory a distillation leaves behind: source tag kept +
    // distilled marker — must NOT appear in the backlog.
    seed_session_import(
        provider.repo_for_test(),
        "typed",
        &["session-import", "distilled"],
        now,
        Some(9),
        Some(9),
    );
    seed_session_import(
        provider.repo_for_test(),
        "plain",
        &[],
        now,
        Some(9),
        Some(9),
    );

    let out = provider
        .distill_backlog(DistillBacklogInput {
            project_id: "p".into(),
            limit: None,
        })
        .unwrap();
    assert_eq!(out["total"], 2, "only raw un-distilled imports count");
    let items = out["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["id"], "raw-old", "oldest first");
    assert_eq!(items[1]["id"], "raw-new");
    assert_eq!(items[0]["memory_type"], "episodic");
    assert_eq!(items[0]["age_days"], 3);
    assert_eq!(items[0]["file_count"], 2, "parsed from the template header");
    assert_eq!(
        items[0]["error_count"], 1,
        "parsed from the template header"
    );
    // Sections absent (nothing recorded) → null, not 0.
    assert!(items[1]["file_count"].is_null());
    assert!(items[1]["error_count"].is_null());
    assert!(
        items[0]["preview"]
            .as_str()
            .unwrap()
            .contains("prompt raw-old"),
        "preview must quote the raw content: {items:?}"
    );
}

#[test]
fn distill_backlog_respects_limit() {
    let provider = make_provider();
    let now = chrono::Utc::now().timestamp();
    for (i, id) in ["a", "b", "c"].iter().enumerate() {
        seed_session_import(
            provider.repo_for_test(),
            id,
            &["session-import"],
            now - 86_400 * (3 - i as i64),
            None,
            None,
        );
    }

    let out = provider
        .distill_backlog(DistillBacklogInput {
            project_id: "p".into(),
            limit: Some(2),
        })
        .unwrap();
    assert_eq!(out["total"], 3, "total counts the backlog beyond the page");
    let items = out["items"].as_array().unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0]["id"], "a", "oldest first");
    assert_eq!(items[1]["id"], "b");
}

#[test]
fn distill_backlog_empty_returns_zero_total() {
    let provider = make_provider();
    let out = provider
        .distill_backlog(DistillBacklogInput {
            project_id: "ghost".into(),
            limit: None,
        })
        .unwrap();
    assert_eq!(out["total"], 0);
    assert_eq!(out["items"].as_array().unwrap().len(), 0);
}

#[test]
fn test_jsonrpc_distill_backlog_roundtrip() {
    let provider = make_provider();
    let now = chrono::Utc::now().timestamp();
    seed_session_import(
        provider.repo_for_test(),
        "raw",
        &["session-import"],
        now,
        Some(1),
        None,
    );
    seed_session_import(
        provider.repo_for_test(),
        "typed",
        &["session-import", "distilled"],
        now,
        None,
        None,
    );
    let server = McpServer::with_provider(Arc::new(provider));

    let response = server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(7)),
        method: "tools/call".into(),
        params: Some(serde_json::json!({
            "name": "distill_backlog",
            "arguments": { "project_id": "p" },
        })),
    });
    assert!(response.error.is_none(), "{:?}", response.error);
    let result = response.result.unwrap();
    let payload: serde_json::Value =
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(payload["total"], 1, "typed+distilled must not appear");
    assert_eq!(payload["items"][0]["id"], "raw");
    assert_eq!(payload["items"][0]["file_count"], 1);
    assert!(payload["items"][0]["error_count"].is_null());
}

// ─── write-time dedup + decision supersede ──────────────────────────

#[test]
fn create_duplicate_returns_existing_without_writing() {
    let provider = make_provider();
    let input = CreateDecisionInput {
        project_id: "p".into(),
        title: "Use SQLite".into(),
        context: "c".into(),
        rationale: "r".into(),
        tradeoffs: "t".into(),
        related_files: vec![],
        tags: vec![],
        importance: None,
        supersedes: None,
    };
    let first = provider.create_decision(input.clone()).unwrap();
    assert_eq!(first["status"], "created");
    let second = provider.create_decision(input).unwrap();
    assert_eq!(second["status"], "duplicate", "{second}");
    assert_eq!(
        second["id"], first["id"],
        "must return the existing memory's id"
    );
    // Only one row exists.
    assert_eq!(
        provider
            .repo_for_test()
            .list_recent_decisions("p", 10)
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn write_dedup_can_be_disabled_by_config() {
    let repo = MemoryRepository::new_in_memory().unwrap();
    repo.initialize_schema().unwrap();
    let mut config = Config::default();
    config.retrieval.write_dedup = false;
    let provider = DefaultMemoryProvider::new(repo, config);

    for _ in 0..2 {
        let out = provider
            .create_failure(CreateFailureInput {
                project_id: "p".into(),
                incident: "same".into(),
                root_cause: "rc".into(),
                fix: "f".into(),
                prevention: "p".into(),
                severity: 3,
                tags: vec![],
            })
            .unwrap();
        assert_eq!(out["status"], "created", "disabled guard must write");
    }
    assert_eq!(
        provider
            .repo_for_test()
            .list_recent_failures("p", 10)
            .unwrap()
            .len(),
        2
    );
}

#[test]
fn decision_supersede_chain_flags_downranks_and_filters() {
    let provider = make_provider();
    let old = provider
        .create_decision(CreateDecisionInput {
            project_id: "p".into(),
            title: "Use Redis for cache".into(),
            context: "session cache".into(),
            rationale: "fast reads".into(),
            tradeoffs: "ops burden".into(),
            related_files: vec![],
            tags: vec!["cache".into()],
            importance: Some(0.9),
            supersedes: None,
        })
        .unwrap();
    let old_id = old["id"].as_str().unwrap().to_string();

    let new = provider
        .create_decision(CreateDecisionInput {
            project_id: "p".into(),
            title: "Use in-process cache instead of Redis".into(),
            context: "session cache".into(),
            rationale: "no infra dependency".into(),
            tradeoffs: "single node only".into(),
            related_files: vec![],
            tags: vec!["cache".into()],
            importance: Some(0.5),
            supersedes: Some(old_id.clone()),
        })
        .unwrap();
    assert_eq!(new["superseded"], serde_json::json!(old_id.clone()));
    let new_id = new["id"].as_str().unwrap().to_string();

    // Chain recorded on the old decision.
    let stored_old = provider
        .repo_for_test()
        .get_decision(&old_id, "p")
        .unwrap()
        .unwrap();
    assert_eq!(stored_old.superseded_by.as_deref(), Some(new_id.as_str()));

    // architectural_decisions: current only by default; both with the flag.
    let current = provider
        .architectural_decisions(ArchitecturalDecisionsInput {
            project_id: "p".into(),
            topic: None,
            limit: None,
            include_superseded: false,
        })
        .unwrap();
    let ids: Vec<&str> = current["decisions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        vec![new_id.as_str()],
        "superseded must be excluded: {current}"
    );
    let all = provider
        .architectural_decisions(ArchitecturalDecisionsInput {
            project_id: "p".into(),
            topic: None,
            limit: None,
            include_superseded: true,
        })
        .unwrap();
    assert_eq!(all["decisions"].as_array().unwrap().len(), 2);

    // Search: both surface (history stays reachable), the superseded one is
    // flagged and cannot outrank its replacement despite higher importance.
    let res = provider
        .search_memory(SearchMemoryInput {
            project_id: "p".into(),
            query: "cache".into(),
            memory_type: Some("decision".into()),
            limit: Some(10),
            tags: vec![],
            before: None,
        })
        .unwrap();
    let arr = res["results"].as_array().unwrap();
    assert_eq!(arr.len(), 2, "{res}");
    assert_eq!(
        arr[0]["id"],
        serde_json::json!(new_id.clone()),
        "replacement ranks first: {res}"
    );
    let old_row = arr
        .iter()
        .find(|r| r["id"] == serde_json::json!(old_id))
        .unwrap();
    assert_eq!(old_row["superseded"], serde_json::json!(true));
    assert_eq!(old_row["superseded_by"], serde_json::json!(new_id));

    // Session brief carries CURRENT constraints only.
    let brief = provider
        .session_brief(SessionBriefInput {
            project_id: "p".into(),
            focus: None,
        })
        .unwrap();
    let brief_ids: Vec<&str> = brief["decisions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["id"].as_str().unwrap())
        .collect();
    assert_eq!(brief_ids, vec![new_id.as_str()]);
}

#[test]
fn superseding_a_missing_decision_fails_loudly() {
    let provider = make_provider();
    let err = provider
        .create_decision(CreateDecisionInput {
            project_id: "p".into(),
            title: "t".into(),
            context: "c".into(),
            rationale: "r".into(),
            tradeoffs: "t".into(),
            related_files: vec![],
            tags: vec![],
            importance: None,
            supersedes: Some("no-such-id".into()),
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("no decision no-such-id"),
        "unexpected error: {err}"
    );
}

// ─── mark_relevance feedback loop ─────────────────────────────────

/// The recommended agent flow is search → get_memory → mark_relevance.
/// get_memory writes an adoption row; the feedback cooldown must key on
/// explicit feedback, not on reads — otherwise the flow ALWAYS hit cooldown
/// and importance never moved.
#[test]
fn mark_relevance_after_get_memory_takes_effect() {
    let provider = make_provider();
    const P: &str = "fb-flow";

    let created = provider
        .create_episodic(CreateEpisodicInput {
            project_id: P.into(),
            session_id: "s".into(),
            summary: "cooldown fix".into(),
            content: "explicit feedback must survive a prior passive read".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
        })
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();

    // search → get_memory: the read writes an adoption row (the OLD cooldown
    // trigger — this is the exact sequence that used to eat the feedback).
    provider
        .search_memory(SearchMemoryInput {
            project_id: P.into(),
            query: "cooldown".into(),
            memory_type: None,
            limit: Some(10),
            before: None,
            tags: vec![],
        })
        .unwrap();
    provider
        .get_memory(GetMemoryInput {
            project_id: P.into(),
            memory_type: "episodic".into(),
            id: id.clone(),
        })
        .unwrap();

    // Core assertion: feedback after a read is NOT cooled down.
    let res = provider
        .mark_relevance(MarkRelevanceInput {
            project_id: P.into(),
            memory_type: "episodic".into(),
            id: id.clone(),
            feedback: "useful".into(),
        })
        .unwrap();
    assert!(
        (res["importance"].as_f64().unwrap() - 0.6).abs() < 1e-6,
        "feedback must apply after a read: {res}"
    );

    // Second feedback inside 24h (either direction) → cooldown, frozen value.
    let res2 = provider
        .mark_relevance(MarkRelevanceInput {
            project_id: P.into(),
            memory_type: "episodic".into(),
            id: id.clone(),
            feedback: "irrelevant".into(),
        })
        .unwrap();
    assert_eq!(res2["status"], "cooldown", "{res2}");
    let mem = provider
        .get_memory(GetMemoryInput {
            project_id: P.into(),
            memory_type: "episodic".into(),
            id,
        })
        .unwrap();
    let importance = mem["memory"]["importance"].as_f64().unwrap();
    assert!(
        (importance - 0.6).abs() < 1e-6,
        "cooldown must not change importance: {mem}"
    );
}

/// 'irrelevant' applies on first call (−0.1) and shares the same 24h
/// per-memory cooldown.
#[test]
fn mark_relevance_irrelevant_applies_then_cools_down() {
    let provider = make_provider();
    const P: &str = "fb-irrelevant";

    let created = provider
        .create_episodic(CreateEpisodicInput {
            project_id: P.into(),
            session_id: "s".into(),
            summary: "irrelevant applies once".into(),
            content: "negative feedback is rate-limited symmetrically".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
        })
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();

    let res = provider
        .mark_relevance(MarkRelevanceInput {
            project_id: P.into(),
            memory_type: "episodic".into(),
            id: id.clone(),
            feedback: "irrelevant".into(),
        })
        .unwrap();
    assert!(
        (res["importance"].as_f64().unwrap() - 0.4).abs() < 1e-6,
        "first 'irrelevant' must apply: {res}"
    );

    let res2 = provider
        .mark_relevance(MarkRelevanceInput {
            project_id: P.into(),
            memory_type: "episodic".into(),
            id,
            feedback: "irrelevant".into(),
        })
        .unwrap();
    assert_eq!(res2["status"], "cooldown", "{res2}");
}

/// Failures have no importance field — feedback still adjusts severity
/// (severity IS their importance signal in the reranker).
#[test]
fn mark_relevance_failure_adjusts_severity() {
    let provider = make_provider();
    const P: &str = "fb-failure";

    let created = provider
        .create_failure(CreateFailureInput {
            project_id: P.into(),
            incident: "severity feedback".into(),
            root_cause: "rc".into(),
            fix: "fix".into(),
            prevention: "prev".into(),
            severity: 3,
            tags: vec![],
        })
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();

    let res = provider
        .mark_relevance(MarkRelevanceInput {
            project_id: P.into(),
            memory_type: "failure".into(),
            id: id.clone(),
            feedback: "useful".into(),
        })
        .unwrap();
    // severity 3 → 4; the reported importance equivalent is 4/5.
    assert!(
        (res["importance"].as_f64().unwrap() - 0.8).abs() < 1e-6,
        "failure feedback must adjust severity: {res}"
    );
    let mem = provider
        .get_memory(GetMemoryInput {
            project_id: P.into(),
            memory_type: "failure".into(),
            id,
        })
        .unwrap();
    assert_eq!(mem["memory"]["severity"], serde_json::json!(4));
}

// ─── update_memory contract (patch validation + echo) ─────────────

#[test]
fn update_memory_patch_applies_and_echoes() {
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

    let mut patch = serde_json::Map::new();
    patch.insert("content".into(), serde_json::json!("new content"));
    patch.insert("importance".into(), serde_json::json!(0.9));
    let res = provider
        .update_memory(UpdateMemoryInput {
            project_id: "p".into(),
            memory_type: "episodic".into(),
            id: id.clone(),
            patch,
        })
        .unwrap();

    // `applied` echoes exactly the two patched fields…
    assert_eq!(res["status"], "updated");
    assert_eq!(res["applied"]["content"], "new content");
    assert_eq!(res["applied"]["importance"], serde_json::json!(0.9));
    assert_eq!(
        res["applied"].as_object().unwrap().len(),
        2,
        "applied must contain exactly the patched fields: {res}"
    );
    // …nothing was skipped, so no `ignored` key at all.
    assert!(
        res.get("ignored").is_none(),
        "ignored must be omitted when nothing was skipped: {res}"
    );

    // The row actually changed — this is the anti-silent-success guarantee.
    let stored = provider
        .repo_for_test()
        .get_episodic(&id, "p")
        .unwrap()
        .unwrap();
    assert_eq!(stored.content, "new content");
    assert!(
        (stored.importance - 0.9).abs() < 1e-6,
        "importance must be 0.9, got {}",
        stored.importance
    );
}

#[test]
fn update_memory_empty_patch_errors() {
    let provider = make_provider();
    let created = provider
        .create_episodic(CreateEpisodicInput {
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "old".into(),
            content: "old".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
        })
        .unwrap();

    // Only the three fixed params: a strict client that stripped undeclared
    // schema keys used to produce exactly this call — and got "updated".
    let err = provider
        .update_memory(UpdateMemoryInput {
            project_id: "p".into(),
            memory_type: "episodic".into(),
            id: created["id"].as_str().unwrap().to_string(),
            patch: serde_json::Map::new(),
        })
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("empty patch"), "unexpected error: {msg}");
    assert!(
        msg.contains("no fields to update"),
        "error must state there is nothing to apply: {msg}"
    );
}

#[test]
fn update_memory_protected_only_patch_errors() {
    let provider = make_provider();
    let created = provider
        .create_episodic(CreateEpisodicInput {
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "old".into(),
            content: "old".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
        })
        .unwrap();

    let mut patch = serde_json::Map::new();
    patch.insert("id".into(), serde_json::json!("forged-id"));
    patch.insert("created_at".into(), serde_json::json!(0));
    let err = provider
        .update_memory(UpdateMemoryInput {
            project_id: "p".into(),
            memory_type: "episodic".into(),
            id: created["id"].as_str().unwrap().to_string(),
            patch,
        })
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("protected"),
        "must explain the keys are protected: {msg}"
    );
    assert!(
        msg.contains("no fields to update"),
        "error must state there is nothing to apply: {msg}"
    );
    // Distinguishable from the plain empty-patch message so the caller's
    // retry can be a fix, not another loop.
    assert!(
        !msg.contains("empty patch"),
        "protected-only must have its own message: {msg}"
    );
}

#[test]
fn update_memory_wrong_type_field_reports() {
    let provider = make_provider();
    let created = provider
        .create_failure(CreateFailureInput {
            project_id: "p".into(),
            incident: "incident".into(),
            root_cause: "rc".into(),
            fix: "old fix".into(),
            prevention: "pv".into(),
            severity: 3,
            tags: vec![],
        })
        .unwrap();
    let id = created["id"].as_str().unwrap().to_string();

    // importance alone against a failure: an error that names the offending
    // field AND the applicable ones (severity is the failure ranking signal).
    let mut patch = serde_json::Map::new();
    patch.insert("importance".into(), serde_json::json!(0.9));
    let err = provider
        .update_memory(UpdateMemoryInput {
            project_id: "p".into(),
            memory_type: "failure".into(),
            id: id.clone(),
            patch,
        })
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("importance"),
        "must name the inapplicable field: {msg}"
    );
    assert!(
        msg.contains("memory_type 'failure'"),
        "must name the memory_type: {msg}"
    );
    assert!(
        msg.contains("severity"),
        "must list the applicable fields (incl. severity): {msg}"
    );

    // Mixed patch: the valid field applies, the inapplicable one is reported
    // in `ignored` with a reason instead of failing the whole call.
    let mut patch = serde_json::Map::new();
    patch.insert("fix".into(), serde_json::json!("new fix"));
    patch.insert("importance".into(), serde_json::json!(0.9));
    let res = provider
        .update_memory(UpdateMemoryInput {
            project_id: "p".into(),
            memory_type: "failure".into(),
            id: id.clone(),
            patch,
        })
        .unwrap();
    assert_eq!(res["applied"]["fix"], "new fix", "{res}");
    let ignored = res["ignored"].as_object().expect("ignored reported: {res}");
    assert_eq!(ignored.len(), 1, "{res}");
    let reason = ignored
        .get("importance")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("importance must be in ignored: {res}"));
    assert!(
        reason.contains("not a field of memory_type 'failure'"),
        "unexpected reason: {reason}"
    );
    // DB: the valid field landed, the row exists.
    let stored = provider
        .repo_for_test()
        .get_failure(&id, "p")
        .unwrap()
        .unwrap();
    assert_eq!(stored.fix, "new fix");
}

#[test]
fn update_memory_schema_declares_all_patch_fields() {
    // Anti-drift: the update_memory schema must declare EXACTLY the three
    // fixed params plus every model field a patch can set — the lightweight
    // stand-in for generating the schema from the models (schemars PoC said
    // most MCP clients can't digest generated anyOf/$ref). Adding a model
    // field without updating the schema turns this red: strict clients strip
    // undeclared keys, which is the silent-empty-patch bug this contract
    // fixed. Declaring a field the models don't have fails here too (typo
    // guard — plain superset checking would miss it).
    let tool = McpServer::list_tools()
        .into_iter()
        .find(|t| t.name == "update_memory")
        .expect("update_memory in tools/list");
    assert_eq!(
        tool.input_schema["additionalProperties"],
        serde_json::json!(false),
        "unknown keys must be rejected by the schema, not stripped silently"
    );
    let schema_keys: std::collections::BTreeSet<&str> = tool.input_schema["properties"]
        .as_object()
        .expect("properties object")
        .keys()
        .map(String::as_str)
        .collect();

    let samples = [
        serde_json::to_value(&crate::models::EpisodicMemory {
            id: "i".into(),
            project_id: "p".into(),
            session_id: "s".into(),
            summary: "sm".into(),
            content: "c".into(),
            files_touched: vec![],
            related_commits: vec![],
            importance: 0.5,
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        })
        .unwrap(),
        serde_json::to_value(&crate::models::DecisionMemory {
            id: "i".into(),
            project_id: "p".into(),
            title: "t".into(),
            context: "c".into(),
            rationale: "r".into(),
            tradeoffs: "to".into(),
            related_files: vec![],
            tags: vec![],
            importance: 0.5,
            superseded_by: None,
            created_at: 0,
            updated_at: 0,
        })
        .unwrap(),
        serde_json::to_value(&crate::models::FailureMemory {
            id: "i".into(),
            project_id: "p".into(),
            incident: "in".into(),
            root_cause: "rc".into(),
            fix: "f".into(),
            prevention: "pv".into(),
            severity: 3,
            tags: vec![],
            created_at: 0,
            updated_at: 0,
        })
        .unwrap(),
        serde_json::to_value(&crate::models::ProceduralMemory {
            id: "i".into(),
            project_id: "p".into(),
            workflow_name: "w".into(),
            steps: vec![],
            related_tools: vec![],
            tags: vec![],
            importance: 0.5,
            created_at: 0,
            updated_at: 0,
        })
        .unwrap(),
    ];
    let mut expected: std::collections::BTreeSet<&str> = Default::default();
    for sample in &samples {
        for key in sample.as_object().unwrap().keys() {
            if !PROTECTED_PATCH_KEYS.contains(&key.as_str()) {
                expected.insert(key.as_str());
            }
        }
    }
    // The three fixed routing params share the properties object.
    expected.insert("project_id");
    expected.insert("memory_type");
    expected.insert("id");
    assert_eq!(
        schema_keys, expected,
        "update_memory schema drifted from the model structs' patchable fields"
    );
}

#[test]
fn update_memory_jsonrpc_undeclared_keys_reach_patch() {
    // The input struct captures patch fields via serde(flatten); this is the
    // deserialization path every real tools/call takes. Keys the struct does
    // not declare explicitly (summary/importance/…) must still arrive in
    // `patch` and apply — proving `additionalProperties: false` in the
    // SCHEMA (a client-side contract) does not affect server-side parsing.
    let server = setup_server();
    let create_resp = server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(1)),
        method: "tools/call".into(),
        params: Some(serde_json::json!({
            "name": "create_episodic",
            "arguments": {
                "project_id": "p", "session_id": "s",
                "summary": "old summary", "content": "old content"
            }
        })),
    });
    let created: serde_json::Value = serde_json::from_str(
        create_resp.result.unwrap()["content"][0]["text"]
            .as_str()
            .unwrap(),
    )
    .unwrap();
    let id = created["id"].as_str().unwrap().to_string();

    let resp = server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(2)),
        method: "tools/call".into(),
        params: Some(serde_json::json!({
            "name": "update_memory",
            "arguments": {
                "project_id": "p", "memory_type": "episodic", "id": id,
                "summary": "patched via json-rpc",
                "importance": 0.8
            }
        })),
    });
    let result = resp.result.expect("update must succeed");
    assert!(result.get("isError").is_none(), "not an error: {result}");
    let parsed: serde_json::Value =
        serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(parsed["status"], "updated");
    assert_eq!(parsed["applied"]["summary"], "patched via json-rpc");
    assert_eq!(parsed["applied"]["importance"], serde_json::json!(0.8));

    // The empty-patch error surfaces as MCP isError (result.isError=true),
    // not a JSON-RPC protocol error — the agent can read and fix it.
    let empty_resp = server.handle_request(JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(serde_json::json!(3)),
        method: "tools/call".into(),
        params: Some(serde_json::json!({
            "name": "update_memory",
            "arguments": { "project_id": "p", "memory_type": "episodic", "id": id }
        })),
    });
    let empty_result = empty_resp
        .result
        .expect("tool errors are results, not protocol errors");
    assert_eq!(empty_result["isError"], serde_json::json!(true));
    let text = empty_result["content"][0]["text"].as_str().unwrap();
    assert!(
        text.contains("empty patch"),
        "unexpected error text: {text}"
    );
}
