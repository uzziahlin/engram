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

/// A frame that passed the shared validation gate: either a request that
/// MUST be answered, or a notification that MUST NOT be.
///
/// The distinction is the `id` KEY's presence, not `req.id != None`: per
/// JSON-RPC 2.0 `id: null` is a legal request id, but it deserializes into
/// `Option::None` — indistinguishable from a missing id. Treating `None` as
/// "notification" silently DROPPED `id: null` requests; the key check fixes
/// that (see `validate_frame`).
#[derive(Debug)]
pub(crate) enum IncomingFrame {
    Request(JsonRpcRequest),
    /// A structurally valid request object with no `id` key — validated (so
    /// garbage notifications still get -32600) but never answered.
    Notification,
}

/// Why a frame was rejected, with its JSON-RPC error class:
/// -32700 (not valid JSON) or -32600 (valid JSON, not a compliant Request).
/// The message is the complete `error.message` — identical for every
/// transport, which is the cross-transport equivalence contract.
#[derive(Debug)]
pub(crate) enum FrameRejection {
    Parse(String),
    Invalid {
        id: Option<serde_json::Value>,
        message: String,
    },
}

/// The SHARED protocol gate for one raw JSON-RPC frame — stdio and HTTP both
/// route through this function so their -32700/-32600 behavior cannot drift
/// again (the batch rejection used to be a clear message on stdio and an
/// opaque serde type error on HTTP).
///
/// Rules, in order:
/// 1. not valid JSON → Parse (-32700);
/// 2. top-level array → Invalid (-32600): MCP never batches, fail loudly;
/// 3. not a valid Request object (missing/wrong-typed `method`, missing
///    `jsonrpc`, …) → Invalid, echoing the id when one is detectable;
/// 4. `jsonrpc != "2.0"` → Invalid (MCP inherits JSON-RPC's version rule);
/// 5. no `id` key → Notification; `id: null` (or any id) → Request.
pub(crate) fn validate_frame(raw: &str) -> Result<IncomingFrame, FrameRejection> {
    let value: serde_json::Value = serde_json::from_str(raw)
        .map_err(|e| FrameRejection::Parse(format!("Parse error: {e}")))?;

    // Batch (top-level array) is explicitly rejected: MCP (2025-06-18)
    // removed batching and the server speaks 2024-11-05, whose clients never
    // batch either.
    if value.is_array() {
        return Err(FrameRejection::Invalid {
            id: None,
            message: "Invalid Request: batch requests (top-level arrays) are not supported; \
                      send one request per frame"
                .into(),
        });
    }

    // Read everything the id-key decision needs BEFORE from_value consumes
    // the value (avoids cloning the frame on this per-request hot path).
    let has_id_key = value.as_object().is_some_and(|o| o.contains_key("id"));
    let cached_id = value.get("id").cloned();
    let mut req: JsonRpcRequest =
        serde_json::from_value(value).map_err(|e| FrameRejection::Invalid {
            id: cached_id,
            message: format!("Invalid Request: {e}"),
        })?;

    // The `jsonrpc` field must be exactly "2.0"; accepting anything else
    // silently protocol-splits.
    if req.jsonrpc != "2.0" {
        return Err(FrameRejection::Invalid {
            message: format!(
                "Invalid Request: jsonrpc must be \"2.0\", got {:?}",
                req.jsonrpc
            ),
            id: req.id,
        });
    }

    // `id: null` deserialized as None but the KEY is present: this is a
    // request with the null id, not a notification. Restore it so the
    // response can be correlated (serializes as `"id": null`, per spec).
    if has_id_key && req.id.is_none() {
        req.id = Some(serde_json::Value::Null);
    }

    if has_id_key {
        Ok(IncomingFrame::Request(req))
    } else {
        Ok(IncomingFrame::Notification)
    }
}

/// Build the error response for a rejected frame. `id` is null when the
/// frame's id was undetectable (spec: "id undetectable → null").
pub(crate) fn rejection_response(rejection: FrameRejection) -> JsonRpcResponse {
    match rejection {
        FrameRejection::Parse(message) => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: None,
            result: None,
            error: Some(JsonRpcError {
                code: -32700,
                message,
                data: None,
            }),
        },
        FrameRejection::Invalid { id, message } => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message,
                data: None,
            }),
        },
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

/// Run `handler` under panic isolation and return the JSON-RPC response —
/// one bad request must not take down the whole server (the release profile
/// uses panic=unwind for exactly this). The id is cloned out first so the
/// error response can be correlated even though the handler consumed the
/// request. Shared by stdio and HTTP (the HTTP copy used to drop the panic
/// payload from `error.data` — the two had already diverged).
pub(crate) fn handle_isolated<H: RequestHandler>(
    handler: &H,
    req: JsonRpcRequest,
) -> JsonRpcResponse {
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
}

/// Serialize + write one response under the stdout lock. Returns false on
/// BrokenPipe (client gone) so callers can stop cleanly.
fn write_response<W: Write>(stdout: &Arc<Mutex<W>>, response: &JsonRpcResponse) -> bool {
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
fn dispatch_isolated<W: Write, H: RequestHandler>(
    handler: &H,
    req: JsonRpcRequest,
    stdout: &Arc<Mutex<W>>,
) -> bool {
    write_response(stdout, &handle_isolated(handler, req))
}

/// Pending-work capacity of a [`Dispatcher`] queue. 64 frames is already an
/// enormous client pipeline; beyond that submitters block, which
/// back-pressures the client's own writes (stdio) or the accept reader
/// (HTTP) instead of accumulating frames without limit.
const QUEUE_CAPACITY: usize = 64;

/// A unit of work submitted to the [`Dispatcher`] pool. The task returns
/// `false` to stop its worker after running (stdio uses this for BrokenPipe:
/// the MCP client is gone, so draining further frames is pointless).
pub(crate) type DispatchTask = Box<dyn FnOnce() -> bool + Send>;

/// Bounded multi-worker task pool shared by the transports.
///
/// Every transport MUST route its handler invocations through one of these
/// (or answer on a fast path): the server's write paths are check-then-act
/// (write-dedup guard, supersede precheck, relevance read-modify-write) and
/// are only correct when handler calls are serialized. `worker_threads=1`
/// degenerates to strict FIFO — the default, and the mode those guards are
/// correct under.
pub(crate) struct Dispatcher {
    tx: std::sync::mpsc::SyncSender<DispatchTask>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Dispatcher {
    /// Spawn `worker_threads.max(1)` workers draining one bounded queue.
    pub(crate) fn new(worker_threads: usize) -> Self {
        use std::sync::mpsc;
        let n_workers = worker_threads.max(1);
        let (tx, rx) = mpsc::sync_channel::<DispatchTask>(QUEUE_CAPACITY);
        let rx = Arc::new(Mutex::new(rx));

        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let rx = Arc::clone(&rx);
            handles.push(std::thread::spawn(move || loop {
                // Poisoned-mutex recovery: a panic in a sibling worker must
                // not deadlock the survivors. into_inner reclaims the lock
                // regardless.
                let task = {
                    let lock = rx.lock().unwrap_or_else(|e| e.into_inner());
                    match lock.recv() {
                        Ok(t) => t,
                        Err(_) => break, // channel closed → drain done
                    }
                };
                // Panic isolation net: a panicking task must not take the
                // worker (or, through the poisoned queue lock, its
                // siblings) down. Transports that must ANSWER the client on
                // panic catch it inside the task (see dispatch_isolated and
                // the HTTP execute path).
                let keep_going = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(task))
                {
                    Ok(cont) => cont,
                    Err(panic) => {
                        tracing::error!("dispatched task panicked: {}", panic_message(&panic));
                        true
                    }
                };
                if !keep_going {
                    break; // task said stop (stdio BrokenPipe)
                }
            }));
        }
        Self { tx, handles }
    }

    /// Enqueue a task. Blocks while the queue is full (back-pressure); errs
    /// only when every worker has exited, in which case the unrun task is
    /// handed back to the caller.
    pub(crate) fn submit(&self, task: DispatchTask) -> Result<(), DispatchTask> {
        self.tx.send(task).map_err(|e| e.0)
    }

    /// Close the queue and wait for the workers to drain it.
    pub(crate) fn join(self) {
        let Self { tx, handles } = self;
        drop(tx); // close channel → workers exit after draining
        for h in handles {
            let _ = h.join();
        }
    }
}

/// Run the JSON-RPC stdio transport: read frames from stdin, dispatch to a
/// bounded worker pool, write responses (id-correlated, order-independent)
/// under a stdout lock. `worker_threads=1` degenerates to sequential processing.
///
/// Frame parsing, fast-path handling, and write-back ordering are preserved
/// verbatim from the original `McpServer::run`; the pool itself now lives in
/// [`Dispatcher`] so the HTTP transport shares the same serialization
/// contract.
pub fn run_stdio<H: RequestHandler + Send + Sync + 'static>(
    handler: Arc<H>,
    worker_threads: usize,
) -> Result<()> {
    run_transport(io::stdin().lock(), io::stdout(), handler, worker_threads)
}

/// Outcome of one capped line read.
enum ReadLine {
    /// A complete line (newline stripped, trailing `\r` tolerated).
    Line(String),
    /// A line longer than the cap — fully consumed and discarded; the byte
    /// count is what was read before bailing on accumulation.
    TooLarge(usize),
    /// Clean EOF with nothing pending.
    Eof,
}

/// Read one line with a hard byte cap, enforced WHILE reading. `BufRead::lines()`
/// has no length limit — a multi-GB single frame was fully allocated before
/// the post-hoc size check ever ran, so the "DoS guard" guarded nothing.
fn read_capped_line<R: BufRead>(reader: &mut R, cap: usize) -> io::Result<ReadLine> {
    let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    let mut too_big = false;
    loop {
        let available = match reader.fill_buf() {
            Ok(a) => a,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if available.is_empty() {
            // EOF: pending unterminated line, or an oversized tail.
            return Ok(if too_big {
                ReadLine::TooLarge(buf.len())
            } else if buf.is_empty() {
                ReadLine::Eof
            } else {
                ReadLine::Line(String::from_utf8_lossy(&buf).trim_end().to_string())
            });
        }
        match available.iter().position(|&b| b == b'\n') {
            Some(i) => {
                let over = too_big || buf.len() + i > cap;
                if !over {
                    buf.extend_from_slice(&available[..i]);
                }
                reader.consume(i + 1);
                if over {
                    return Ok(ReadLine::TooLarge(buf.len()));
                }
                return Ok(ReadLine::Line(
                    String::from_utf8_lossy(&buf).trim_end().to_string(),
                ));
            }
            None => {
                let n = available.len();
                if !too_big && buf.len() + n > cap {
                    too_big = true;
                }
                if !too_big {
                    buf.extend_from_slice(available);
                }
                reader.consume(n);
            }
        }
    }
}

/// Transport core over any reader/writer pair (stdio in production; piped
/// buffers in tests). Reading happens on the caller's thread; requests go to
/// a bounded worker pool through a BOUNDED queue (a runaway client pipeline
/// back-pressures the reader instead of accumulating frames without limit);
/// responses are written under a writer lock, id-correlated and
/// order-independent.
pub fn run_transport<R, W, H>(
    reader: R,
    writer: W,
    handler: Arc<H>,
    worker_threads: usize,
) -> Result<()>
where
    R: BufRead,
    W: Write + Send + 'static,
    H: RequestHandler + Send + Sync + 'static,
{
    let stdout = Arc::new(Mutex::new(writer));
    // Worker pool semantics (bounded queue, N workers, poisoned-lock
    // recovery, worker stops on BrokenPipe) live in `Dispatcher`; this loop
    // only decides fast-path vs queued, exactly as before.
    let dispatcher = Dispatcher::new(worker_threads);

    let fast_path = handler.fast_path_methods();
    let mut reader = reader;
    // DoS guard budget: enforced DURING the read (see read_capped_line), so
    // a misbehaving client can't exhaust memory with a huge single frame.
    const MAX_REQ_BYTES: usize = 16 * 1024 * 1024; // 16 MiB
    loop {
        let line = match read_capped_line(&mut reader, MAX_REQ_BYTES) {
            Ok(ReadLine::Line(l)) => l,
            Ok(ReadLine::TooLarge(seen)) => {
                let response = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: None,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -32700,
                        message: format!(
                            "Parse error: request too large (> {MAX_REQ_BYTES} bytes; gave up \
                             after {seen})"
                        ),
                        data: None,
                    }),
                };
                if !write_response(&stdout, &response) {
                    break;
                }
                continue;
            }
            Ok(ReadLine::Eof) => break,
            Err(e) => return Err(anyhow::anyhow!(e)).context("failed to read from transport"),
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        // Frame validation is the SHARED gate (validate_frame): parse error
        // vs invalid-request classes, batch rejection, jsonrpc version, and
        // the id-key notification/request distinction — identical to the
        // HTTP transport by construction (see the equivalence tests).
        let frame = match validate_frame(&line) {
            Ok(frame) => frame,
            Err(rejection) => {
                if !write_response(&stdout, &rejection_response(rejection)) {
                    break;
                }
                continue;
            }
        };
        // Notifications get no response — but only once the frame is known
        // to be a structurally valid Request-shaped object (an invalid frame
        // with no id already got a -32600 with id null above).
        let req = match frame {
            IncomingFrame::Notification => continue,
            IncomingFrame::Request(req) => req,
        };

        // Read-only methods (ping/tools/list/prompts/list/initialize) are
        // answered inline so a long-running tool on the worker pool cannot
        // starve client health probes. MCP allows out-of-order responses.
        if fast_path.contains(&req.method.as_str()) {
            if !dispatch_isolated(handler.as_ref(), req, &stdout) {
                break;
            }
        } else {
            let handler = Arc::clone(&handler);
            let stdout = Arc::clone(&stdout);
            // False = BrokenPipe: MCP client gone → worker stops, mirroring
            // the original worker loop.
            let task: DispatchTask =
                Box::new(move || dispatch_isolated(handler.as_ref(), req, &stdout));
            if dispatcher.submit(task).is_err() {
                break; // workers all exited
            }
        }
    }

    dispatcher.join();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Handler that panics on "boom", echoes method otherwise; declares
    /// "ping" as fast-path so inline dispatch is exercised.
    struct TestHandler {
        calls: AtomicUsize,
    }
    impl RequestHandler for TestHandler {
        fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match req.method.as_str() {
                "boom" => panic!("intentional test panic"),
                m => JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: Some(serde_json::json!({ "echo": m })),
                    error: None,
                },
            }
        }
        fn fast_path_methods(&self) -> &'static [&'static str] {
            &["ping"]
        }
    }

    fn drive(input: &str, workers: usize) -> Vec<serde_json::Value> {
        let handler = Arc::new(TestHandler {
            calls: AtomicUsize::new(0),
        });
        let out: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&out);
        run_transport(
            Cursor::new(input.to_string().into_bytes()),
            Sink(sink),
            handler,
            workers,
        )
        .unwrap();
        let buf = out.lock().unwrap_or_else(|e| e.into_inner());
        let text = String::from_utf8(buf.clone()).unwrap();
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// Writer adapter that appends into the shared buffer.
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn responses_come_back_id_correlated() {
        let input = concat!(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":2,"method":"ping"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":"abc","method":"other"}"#,
            "\n",
        );
        let responses = drive(input, 1);
        assert_eq!(responses.len(), 3);
        let ids: Vec<_> = responses.iter().map(|r| r["id"].clone()).collect();
        assert!(ids.contains(&serde_json::json!(1)));
        assert!(ids.contains(&serde_json::json!(2)));
        assert!(ids.contains(&serde_json::json!("abc")));
        assert!(responses.iter().all(|r| r.get("error").is_none()));
    }

    #[test]
    fn notifications_get_no_response() {
        let input = concat!(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","id":7,"method":"ping"}"#,
            "\n",
        );
        let responses = drive(input, 1);
        assert_eq!(responses.len(), 1, "only the request with an id replies");
        assert_eq!(responses[0]["id"], 7);
    }

    #[test]
    fn id_null_is_a_request_not_a_notification() {
        // JSON-RPC 2.0: `id: null` is a legal request id, but it deserializes
        // into Option::None — indistinguishable from a MISSING id, which is a
        // notification. The id KEY's presence is authoritative now: id:null
        // gets a correlated response; a true notification stays silent.
        let input = concat!(
            r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#,
            "\n",
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            "\n",
        );
        let responses = drive(input, 1);
        assert_eq!(
            responses.len(),
            1,
            "id:null must be answered; missing id must not"
        );
        assert_eq!(responses[0]["id"], serde_json::Value::Null);
        assert!(
            responses[0].get("result").is_some(),
            "ping answered: {}",
            responses[0]
        );
    }

    /// The malformed-frame matrix both transports must reject with the SAME
    /// JSON-RPC error code. `validate_frame` is the single shared gate — these
    /// assertions pin its contract and (via `drive`) the stdio end-to-end
    /// behavior; the HTTP side is pinned by
    /// `http::tests::http_frame_validation_matches_stdio_codes` posting the
    /// same frames.
    #[test]
    fn malformed_frames_are_rejected_identically_across_transports() {
        let matrix: &[(&str, i32)] = &[
            // not JSON at all → Parse error
            ("this is not json", -32700),
            // valid JSON but not an object → Invalid Request
            ("42", -32600),
            // method wrong type → Invalid Request (NOT a parse error)
            (r#"{"jsonrpc":"2.0","id":1,"method":42}"#, -32600),
            // missing jsonrpc field → Invalid Request
            (r#"{"id":1,"method":"ping"}"#, -32600),
            // jsonrpc 1.0 → Invalid Request
            (r#"{"jsonrpc":"1.0","id":2,"method":"ping"}"#, -32600),
            // batch array → Invalid Request with the explicit message
            (r#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#, -32600),
        ];
        for (frame, code) in matrix {
            // The shared gate rejects with that code…
            let rejection = validate_frame(frame).unwrap_err();
            let resp = rejection_response(rejection);
            assert_eq!(
                resp.error.as_ref().unwrap().code,
                *code,
                "shared gate frame: {frame}"
            );
            // …and the stdio transport emits exactly the same response.
            let responses = drive(&format!("{frame}\n"), 1);
            assert_eq!(responses.len(), 1, "frame: {frame}");
            assert_eq!(
                responses[0]["error"]["code"], *code,
                "stdio end-to-end frame: {frame}"
            );
            assert_eq!(
                responses[0]["error"]["message"],
                resp.error.unwrap().message,
                "message must be identical through the whole stdio path"
            );
        }
    }

    #[test]
    fn malformed_json_is_parse_error_with_null_id() {
        let responses = drive("this is not json\n", 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["error"]["code"], -32700);
        assert!(responses[0]["id"].is_null());
    }

    #[test]
    fn valid_json_non_compliant_request_is_invalid_request() {
        // Valid JSON, but `method` is a number — -32600, NOT -32700.
        let responses = drive(r#"{"jsonrpc":"2.0","id":1,"method":42}"#, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["error"]["code"], -32600);
        assert_eq!(responses[0]["id"], 1);
    }

    #[test]
    fn wrong_jsonrpc_version_is_rejected() {
        let responses = drive(r#"{"jsonrpc":"1.0","id":2,"method":"ping"}"#, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["error"]["code"], -32600);
        assert!(responses[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("2.0"));
    }

    #[test]
    fn panicking_request_returns_internal_error_not_crash() {
        let responses = drive(r#"{"jsonrpc":"2.0","id":9,"method":"boom"}"#, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["error"]["code"], -32603);
        assert!(responses[0]["error"]["data"]["panic"]
            .as_str()
            .unwrap()
            .contains("intentional"));
    }

    #[test]
    fn oversized_frame_is_rejected_without_parsing() {
        let huge = format!(
            r#"{{"jsonrpc":"2.0","id":1,"method":"ping","pad":"{}"}}"#,
            "x".repeat(17 * 1024 * 1024)
        );
        let responses = drive(&huge, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["error"]["code"], -32700);
        assert!(responses[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("too large"));
    }

    #[test]
    fn oversized_frame_rejection_never_allocates_the_frame() {
        // The read-time cap must reject the line WITHOUT the reader having
        // buffered the whole thing — verified by behavior: a 2 MB line under
        // a deliberately tiny cap still leaves the stream positioned at the
        // NEXT line, which must be processed normally.
        let mut input = String::new();
        input.push_str(&"x".repeat(2 * 1024 * 1024));
        input.push('\n');
        input.push_str(r#"{"jsonrpc":"2.0","id":5,"method":"ping"}"#);
        input.push('\n');
        let responses = drive(&input, 1);
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0]["error"]["code"], -32700);
        assert_eq!(responses[1]["id"], 5);
    }

    #[test]
    fn unterminated_oversized_tail_is_rejected() {
        // No trailing newline: the oversized line must still be rejected at
        // EOF, not silently accepted as the final line.
        let input = "y".repeat(2 * 1024 * 1024);
        let responses = drive(&input, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["error"]["code"], -32700);
    }

    #[test]
    fn batch_requests_are_rejected_loudly() {
        // MCP never batches; a top-level array is -32600 with an explicit
        // message (previously it fell into request deserialization and
        // produced an opaque type error with the same code).
        let input = concat!(
            r#"[{"jsonrpc":"2.0","id":1,"method":"ping"},{"jsonrpc":"2.0","id":2,"method":"tools/list"}]"#,
            "\n",
        );
        let responses = drive(input, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["error"]["code"], -32600);
        assert!(responses[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("batch"));
    }

    #[test]
    fn crlf_frames_are_accepted() {
        let input = "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"ping\"}\r\n";
        let responses = drive(input, 1);
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0]["id"], 3);
        assert!(responses[0].get("error").is_none());
    }

    #[test]
    fn multiple_workers_all_respond() {
        let mut input = String::new();
        for i in 0..20 {
            input.push_str(&format!(
                "{{\"jsonrpc\":\"2.0\",\"id\":{i},\"method\":\"m{i}\"}}\n"
            ));
        }
        let responses = drive(&input, 4);
        assert_eq!(responses.len(), 20);
    }

    #[test]
    fn dispatcher_single_worker_is_strictly_serial() {
        // One worker must never interleave two tasks: each task holds an
        // "active" slot for a while; a second enterer would observe != 0.
        let d = Dispatcher::new(1);
        let active = Arc::new(AtomicUsize::new(0));
        let overlapped = Arc::new(AtomicUsize::new(0));
        for _ in 0..8 {
            let active = Arc::clone(&active);
            let overlapped = Arc::clone(&overlapped);
            assert!(d
                .submit(Box::new(move || {
                    if active.fetch_add(1, Ordering::SeqCst) != 0 {
                        overlapped.fetch_add(1, Ordering::SeqCst);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                    active.fetch_sub(1, Ordering::SeqCst);
                    true
                }))
                .is_ok());
        }
        d.join();
        assert_eq!(
            overlapped.load(Ordering::SeqCst),
            0,
            "tasks must never interleave with a single worker"
        );
    }

    #[test]
    fn dispatcher_full_queue_blocks_submit() {
        // Back-pressure contract: with the queue full, submit blocks instead
        // of growing without bound.
        let d = Dispatcher::new(1);
        let (release, gate) = std::sync::mpsc::channel::<()>();
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        assert!(d
            .submit(Box::new(move || {
                started_tx.send(()).expect("test alive");
                let _ = gate.recv(); // park the only worker until released
                true
            }))
            .is_ok());
        started_rx.recv().unwrap(); // worker is now parked inside the task
        for _ in 0..QUEUE_CAPACITY {
            assert!(d.submit(Box::new(|| true)).is_ok());
        }
        let done = Arc::new(AtomicUsize::new(0));
        let done_worker = Arc::clone(&done);
        let submitter = std::thread::spawn(move || {
            assert!(d.submit(Box::new(|| true)).is_ok());
            done_worker.fetch_add(1, Ordering::SeqCst);
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(
            done.load(Ordering::SeqCst),
            0,
            "submit must block while the queue is full"
        );
        drop(release); // unblock the worker → queue drains → submit returns
        submitter.join().unwrap();
        assert_eq!(done.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dispatcher_survives_panicking_tasks() {
        // A panicking task must not kill the worker or poison anything for
        // later tasks (the default panic hook still prints to stderr, like
        // the stdio panic tests above).
        let d = Dispatcher::new(1);
        assert!(d
            .submit(Box::new(|| panic!("intentional test panic")))
            .is_ok());
        let ran = Arc::new(AtomicUsize::new(0));
        for _ in 0..4 {
            let ran = Arc::clone(&ran);
            assert!(d
                .submit(Box::new(move || {
                    ran.fetch_add(1, Ordering::SeqCst);
                    true
                }))
                .is_ok());
        }
        d.join();
        assert_eq!(
            ran.load(Ordering::SeqCst),
            4,
            "worker must keep draining after a task panic"
        );
    }
}
