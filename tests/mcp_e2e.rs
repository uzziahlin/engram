//! Rust end-to-end MCP JSON-RPC integration tests.
//!
//! Spawns the built `engram` binary as an MCP server (stdio), feeds JSON-RPC
//! lines on stdin, and asserts on stdout. `HOME` is redirected to an isolated
//! tempdir so the server's `~/.engram/memory.db` is per-test (mirrors
//! `tests/integration_test.sh`). This net guards the hardening refactor
//! (connection pool, concurrent request handling) against regressions.

use std::io::Write;
use std::process::{Command, Stdio};

/// Spawn the built engram binary as an MCP server, feed JSON-RPC lines on
/// stdin, return the full stdout. HOME is redirected to an isolated tempdir so
/// the server's `~/.engram/memory.db` is per-test.
fn run_mcp(home: &std::path::Path, requests: &[&str]) -> String {
    let bin = env!("CARGO_BIN_EXE_engram");
    let mut child = Command::new(bin)
        .env("HOME", home)
        .env("RUST_LOG", "off")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn engram");
    {
        let mut stdin = child.stdin.take().unwrap();
        for r in requests {
            writeln!(stdin, "{r}").unwrap();
        }
    } // drop stdin → EOF → server loop ends
    let out = child.wait_with_output().expect("wait engram");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

#[test]
fn initialize_and_tools_list() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run_mcp(
        tmp.path(),
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        ],
    );
    assert!(out.contains("\"serverInfo\""), "missing serverInfo: {out}");
    assert!(out.contains("search_memory"), "missing tool list: {out}");
}

#[test]
fn create_then_search_episodic() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run_mcp(
        tmp.path(),
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"create_episodic","arguments":{"project_id":"p","session_id":"s","summary":"fix FTS5 crash","content":"details","importance":0.9}}}"#,
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"search_memory","arguments":{"project_id":"p","query":"FTS5 crash","limit":5}}}"#,
        ],
    );
    assert!(
        out.contains("fix FTS5 crash"),
        "search did not return created memory: {out}"
    );
}

#[test]
fn unknown_tool_returns_error() {
    let tmp = tempfile::tempdir().unwrap();
    let out = run_mcp(
        tmp.path(),
        &[
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"does_not_exist","arguments":{}}}"#,
        ],
    );
    assert!(
        out.contains("error") || out.contains("Unknown"),
        "expected error for unknown tool: {out}"
    );
}

#[test]
fn pipelined_requests_all_get_responses() {
    // Pipelined independent requests (tools/list x10) must all get a response.
    // Default worker_threads=1 processes them sequentially (FIFO), so all ids
    // return. This guards the worker-pool run() loop end-to-end.
    let tmp = tempfile::tempdir().unwrap();
    let reqs: Vec<String> = (1..=10)
        .map(|i| format!(r#"{{"jsonrpc":"2.0","id":{i},"method":"tools/list"}}"#))
        .collect();
    let refs: Vec<&str> = reqs.iter().map(|s| s.as_str()).collect();
    let out = run_mcp(tmp.path(), &refs);
    for i in 1..=10 {
        assert!(
            out.contains(&format!("\"id\":{i}")),
            "missing response id {i} in: {out}"
        );
    }
}
