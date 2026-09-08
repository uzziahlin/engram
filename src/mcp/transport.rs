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

    /// Methods the transport may answer inline on its reader thread instead
    /// of queueing to the worker pool. With the default single worker, one
    /// long tool call (ingest/reindex) would otherwise starve `ping` and
    /// `tools/list` long enough for client health probes to time out. Only
    /// list methods that never touch mutable state belong here.
    fn fast_path_methods(&self) -> &'static [&'static str] {
        &[]
    }
}

/// Render a caught panic payload as a best-effort string for logging.
fn panic_message(panic: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Serialize + write one response under the stdout lock. Returns false on
/// BrokenPipe (client gone) so callers can stop cleanly.
fn write_response(stdout: &Arc<Mutex<io::Stdout>>, response: &JsonRpcResponse) -> bool {
    let Ok(s) = serde_json::to_string(response) else {
        return true;
    };
    let mut out = stdout.lock().unwrap_or_else(|e| e.into_inner());
    match writeln!(out, "{s}").and_then(|_| out.flush()) {
        Ok(()) => true,
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => false,
        Err(e) => {
            tracing::warn!("response write failed: {e}");
            true
        }
    }
}

/// Run `handler` under panic isolation and write the response. Returns false
/// on BrokenPipe.
fn dispatch_isolated<H: RequestHandler>(
    handler: &H,
    req: JsonRpcRequest,
    stdout: &Arc<Mutex<io::Stdout>>,
) -> bool {
    // Panic isolation: one bad request must not take down the whole MCP
    // server (the release profile uses panic=unwind for exactly this). The id
    // is cloned out first so the error response can be correlated even though
    // the handler consumed the request.
    let response = {
        let id = req.id.clone();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler.handle(req))) {
            Ok(resp) => resp,
            Err(panic) => {
                let msg = panic_message(&panic);
                tracing::error!("request handler panicked: {msg}");
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32603,
                        message: "Internal error: request handler panicked".into(),
                        data: Some(serde_json::json!({ "panic": msg })),
                    }),
                }
            }
        }
    };
    write_response(stdout, &response)
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
            if !dispatch_isolated(handler.as_ref(), req, &stdout) {
                break; // BrokenPipe: MCP client gone
            }
        }));
    }

    let fast_path = handler.fast_path_methods();
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.context("failed to read from stdin")?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        // DoS guard: reject oversized requests before parsing them, so a
        // misbehaving client can't exhaust memory with a huge single frame.
        const MAX_REQ_BYTES: usize = 16 * 1024 * 1024; // 16 MiB
        if line.len() > MAX_REQ_BYTES {
            let response = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: format!(
                        "Parse error: request too large ({} bytes; max {})",
                        line.len(),
                        MAX_REQ_BYTES
                    ),
                    data: None,
                }),
            };
            if !write_response(&stdout, &response) {
                break;
            }
            continue;
        }

        // Two failure classes, kept distinct per the JSON-RPC 2.0 spec:
        //   -32700 Parse error: the frame is not valid JSON;
        //   -32600 Invalid Request: valid JSON, but not a compliant Request
        //   object (missing/typo'd method, wrong types, wrong jsonrpc version).
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: None, // serializes as null per spec (id undetectable)
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!("Parse error: {e}"),
                        data: None,
                    }),
                };
                if !write_response(&stdout, &response) {
                    break;
                }
                continue;
            }
        };

        // Notifications (no `id`) get no response — but only once the frame
        // is known to be a structurally valid Request; an invalid frame with
        // no id must still get a -32600 with id null.
        let cached_id = value.get("id").cloned();
        let req = match serde_json::from_value::<JsonRpcRequest>(value) {
            Ok(req) => req,
            Err(e) => {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: cached_id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32600,
                        message: format!("Invalid Request: {e}"),
                        data: None,
                    }),
                };
                if !write_response(&stdout, &response) {
                    break;
                }
                continue;
            }
        };
        // The `jsonrpc` field must be exactly "2.0" (MCP inherits JSON-RPC's
        // versioning rule); accepting anything else silently protocol-splits.
        if req.jsonrpc != "2.0" {
            let response = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: req.id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32600,
                    message: format!(
                        "Invalid Request: jsonrpc must be \"2.0\", got {:?}",
                        req.jsonrpc
                    ),
                    data: None,
                }),
            };
            if !write_response(&stdout, &response) {
                break;
            }
            continue;
        }
        if req.id.is_none() {
            continue; // notification
        }

        // Read-only methods (ping/tools/list/prompts/list/initialize) are
        // answered inline so a long-running tool on the worker pool cannot
        // starve client health probes. MCP allows out-of-order responses.
        if fast_path.contains(&req.method.as_str()) {
            if !dispatch_isolated(handler.as_ref(), req, &stdout) {
                break;
            }
        } else if tx.send(req).is_err() {
            break; // workers all exited
        }
    }

    drop(tx); // close channel → workers exit after draining
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}
