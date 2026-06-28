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
    assert_eq!(tools.len(), 22);

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
    assert_eq!(tools.len(), 22);
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
fn test_reflection_reflect_list_confirm_closed_loop() {
    let provider = make_provider();
    let project = "test-project";

    // Seed three auth failures → one recurring tag.
    for _ in 0..3 {
        provider
            .create_failure(CreateFailureInput {
                project_id: project.into(),
                incident: "token expired".into(),
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
            limit: 10,
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
            limit: 10,
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
    for _ in 0..3 {
        provider
            .create_failure(CreateFailureInput {
                project_id: project.into(),
                incident: "x".into(),
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
            limit: 10,
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
