use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};

use anyhow::{Context as AnyhowContext, Result};
use serde::{Deserialize, Serialize};

/// JSON-RPC request for MCP protocol.
#[derive(Debug, Deserialize)]
pub struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    pub params: Option<serde_json::Value>,
}

/// JSON-RPC response for MCP protocol.
#[derive(Debug, Serialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// Business-side handler invoked by the stdio transport for every request.
///
/// Keeping the transport coupled only to this trait (not to `McpServer`) is
/// what lets a future HTTP transport swap in without touching business code.
pub trait RequestHandler: Send + Sync {
    fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse;
}

/// Run the JSON-RPC stdio transport: read frames from stdin, dispatch to a
/// bounded worker pool, write responses (id-correlated, order-independent)
/// under a stdout lock. `worker_threads=1` degenerates to sequential processing.
///
/// Concurrency model, frame parsing, poisoned-mutex recovery, and write-back
/// ordering are preserved verbatim from the original `McpServer::run`.
pub fn run_stdio<H: RequestHandler + Send + Sync + 'static>(
    handler: Arc<H>,
    worker_threads: usize,
) -> Result<()> {
    use std::sync::mpsc;
    let n_workers = worker_threads.max(1);
    let stdout = Arc::new(Mutex::new(io::stdout()));
    let (tx, rx) = mpsc::channel::<JsonRpcRequest>();
    let rx = Arc::new(Mutex::new(rx));

    let mut handles = Vec::with_capacity(n_workers);
    for _ in 0..n_workers {
        let handler = Arc::clone(&handler);
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
            let response = handler.handle(req);
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
