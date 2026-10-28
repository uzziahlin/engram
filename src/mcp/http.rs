//! MCP Streamable-HTTP transport (optional, `[http]` config).
//!
//! A minimal synchronous implementation on `tiny_http` — engram has no async
//! runtime by design, and the subset of the protocol engram needs is small:
//!
//! - `POST /mcp` with a JSON-RPC message → one `application/json` response
//!   (the spec allows plain JSON replies when the server never pushes);
//! - notifications (no `id`) → `202 Accepted`;
//! - `DELETE /mcp` with a valid session id → session terminated, `200`;
//! - no GET/SSE stream (the spec permits `405` for servers without push);
//! - every request must carry `Authorization: Bearer <token>`, compared in
//!   constant time (it is the only auth factor);
//! - sessions: `initialize` is answered with an `mcp-session-id` header;
//!   subsequent requests must carry a known session id (unknown/missing →
//!   `404`, per the Streamable HTTP spec). A server restart invalidates all
//!   sessions — clients re-initialize.
//!
//! Connections are read and validated on a small pool of accept threads;
//! every handler invocation is serialized through the same bounded
//! [`Dispatcher`] the stdio transport uses (default one worker), because the
//! server's write paths are hash-dedup-guarded check-then-act writes and are
//! only correct under serialized handler calls. Read-only fast-path methods
//! (see [`RequestHandler::fast_path_methods`]) answer inline on the accept
//! thread so client health probes stay responsive during long tool calls.
//! Frame validation and panic isolation are THE shared implementations in
//! `transport.rs` (`validate_frame` / `handle_isolated`) — the stdio and
//! HTTP transports cannot drift on error codes or messages.

use std::collections::HashSet;
use std::io::Read;
use std::sync::{Arc, Mutex};

use anyhow::Result;

use super::transport::{
    handle_isolated, rejection_response, validate_frame, Dispatcher, IncomingFrame, JsonRpcRequest,
    JsonRpcResponse, RequestHandler,
};

/// Fixed number of accept-reader threads. They only do cheap connection
/// work (auth/path/session/body checks, JSON parse) and never run handler
/// code concurrently — handler calls go through the [`Dispatcher`], so more
/// accept readers do not mean more concurrent writes.
pub const ACCEPT_READERS: usize = 4;

/// Largest accepted request body (parity with the stdio frame cap).
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Session registry shared by the accept threads: session ids issued by
/// `initialize` responses. A restart empties it (all old ids become invalid).
/// `enforcement_on` latches: once ANY session has been issued, requests must
/// present a known session id — deleting the last session does not quietly
/// revert the server to stateless-permissive mode.
type Sessions = Arc<Mutex<SessionState>>;

struct SessionState {
    sessions: HashSet<String>,
    enforcement_on: bool,
}

/// Byte-wise constant-time equality. The bearer token is the only auth
/// factor, so the comparison must not short-circuit on the first differing
/// byte (timing oracles over a local network are a real, if narrow, channel).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..a.len() {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

/// Serve MCP over HTTP forever. Blocks the calling thread.
pub fn run_http<H: RequestHandler + Send + Sync + 'static>(
    handler: Arc<H>,
    bind: &str,
    auth_token: &str,
    worker_threads: usize,
) -> Result<()> {
    let server = tiny_http::Server::http(bind)
        .map_err(|e| anyhow::anyhow!("HTTP transport failed to bind {bind}: {e}"))?;
    tracing::info!(
        "engram MCP HTTP transport listening on {} ({} accept readers, {} workers, token auth required)",
        server.server_addr(),
        ACCEPT_READERS,
        worker_threads.max(1)
    );
    serve(server, handler, auth_token, worker_threads)
}

/// Serve on an already-bound server (tests pass their own to learn the
/// port). `worker_threads` has the same meaning as for stdio: the number of
/// dispatcher workers that serialize handler calls.
pub(crate) fn serve<H: RequestHandler + Send + Sync + 'static>(
    server: tiny_http::Server,
    handler: Arc<H>,
    auth_token: &str,
    worker_threads: usize,
) -> Result<()> {
    let server = Arc::new(server);
    let sessions: Sessions = Arc::new(Mutex::new(SessionState {
        sessions: HashSet::new(),
        enforcement_on: false,
    }));
    // Handler calls MUST go through the dispatcher (the same contract stdio
    // enforces): with workers running inline on the accept threads, two
    // concurrent identical create_* POSTs would both pass the write-dedup
    // guard and double-write.
    let dispatcher = Arc::new(Dispatcher::new(worker_threads));
    let fast_path = handler.fast_path_methods();
    let mut handles = Vec::with_capacity(ACCEPT_READERS);
    for _ in 0..ACCEPT_READERS {
        let server = Arc::clone(&server);
        let handler = Arc::clone(&handler);
        let token = auth_token.to_string();
        let sessions = Arc::clone(&sessions);
        let dispatcher = Arc::clone(&dispatcher);
        handles.push(std::thread::spawn(move || loop {
            let Ok(request) = server.recv() else {
                break; // server closed
            };
            handle_request(
                request,
                Arc::clone(&handler),
                &token,
                &sessions,
                dispatcher.as_ref(),
                fast_path,
            );
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// Connection work happens on the accept thread: auth, path, method, body
/// read, frame validation, and the session gate. Only the handler call is
/// dispatched — inline for fast-path methods, queued to the [`Dispatcher`]
/// otherwise (see the module docs for why handler calls must serialize).
fn handle_request<H: RequestHandler + Send + Sync + 'static>(
    mut request: tiny_http::Request,
    handler: Arc<H>,
    token: &str,
    sessions: &Sessions,
    dispatcher: &Dispatcher,
    fast_path: &'static [&'static str],
) {
    // Auth first: never process an unauthenticated body. Constant-time token
    // comparison; 401 carries WWW-Authenticate per RFC 6750.
    let expected = format!("Bearer {token}");
    let authorized = request.headers().iter().any(|h| {
        h.field.equiv("Authorization")
            && constant_time_eq(h.value.as_str().as_bytes(), expected.as_bytes())
    });
    if !authorized {
        respond_with_headers(
            request,
            401,
            r#"{"error":"unauthorized"}"#,
            "application/json",
            &[("WWW-Authenticate", "Bearer")],
        );
        return;
    }

    // Exact path match: anything but /mcp is not this server.
    if request.url() != "/mcp" {
        respond(
            request,
            404,
            r#"{"error":"not found; POST /mcp only"}"#,
            "application/json",
        );
        return;
    }

    // DELETE /mcp with a valid session id terminates the session (spec).
    if request.method() == &tiny_http::Method::Delete {
        let sid = header_value(&request, "mcp-session-id");
        let removed = sid
            .map(|s| {
                sessions
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .sessions
                    .remove(&s)
            })
            .unwrap_or(false);
        respond(
            request,
            if removed { 200 } else { 404 },
            if removed {
                ""
            } else {
                r#"{"error":"unknown session"}"#
            },
            if removed {
                "text/plain"
            } else {
                "application/json"
            },
        );
        return;
    }

    if request.method() != &tiny_http::Method::Post {
        respond(
            request,
            405,
            r#"{"error":"POST /mcp only"}"#,
            "application/json",
        );
        return;
    }

    // Bounded body read — a huge POST must not be slurped into memory whole.
    let mut body = Vec::with_capacity(64 * 1024);
    if request
        .as_reader()
        .take(MAX_BODY_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .is_err()
    {
        respond(
            request,
            400,
            r#"{"error":"unreadable body"}"#,
            "application/json",
        );
        return;
    }
    if body.len() > MAX_BODY_BYTES {
        respond(
            request,
            413,
            r#"{"error":"body too large"}"#,
            "application/json",
        );
        return;
    }

    // Frame validation is THE shared gate (see validate_frame): stdio and
    // HTTP reject the same frames with the same code and message. Only the
    // HTTP status differs (400 wraps both error classes).
    let raw = match std::str::from_utf8(&body) {
        Ok(raw) => raw,
        Err(_) => {
            let resp = rejection_response(super::transport::FrameRejection::Parse(
                "Parse error: request body is not valid UTF-8".into(),
            ));
            respond(request, 400, &resp.serialize(), "application/json");
            return;
        }
    };
    let frame = match validate_frame(raw) {
        Ok(frame) => frame,
        Err(rejection) => {
            let resp = rejection_response(rejection);
            respond(request, 400, &resp.serialize(), "application/json");
            return;
        }
    };
    let req = match frame {
        IncomingFrame::Notification => {
            // Notification: accepted, no body (per Streamable HTTP).
            respond(request, 202, "", "text/plain");
            return;
        }
        IncomingFrame::Request(req) => req,
    };

    // Session gate AFTER parsing so the method is known. initialize mints a
    // session id (returned via response header) and latches enforcement on:
    // every later request must present a known id (unknown → 404, per spec).
    // Before any initialize the server is stateless-permissive, so one-shot
    // clients (curl) keep working without an initialize round-trip.
    let presented_session = header_value(&request, "mcp-session-id");
    let mut issued_session: Option<String> = None;
    if req.method == "initialize" {
        issued_session = Some(uuid::Uuid::new_v4().to_string());
        let mut state = sessions.lock().unwrap_or_else(|e| e.into_inner());
        state.sessions.insert(issued_session.clone().unwrap());
        state.enforcement_on = true;
    } else {
        let state = sessions.lock().unwrap_or_else(|e| e.into_inner());
        let known = presented_session
            .as_deref()
            .is_some_and(|sid| state.sessions.contains(sid));
        if !known && state.enforcement_on {
            drop(state);
            respond(
                request,
                404,
                r#"{"error":"unknown or missing session — (re)initialize first"}"#,
                "application/json",
            );
            return;
        }
    }

    if fast_path.contains(&req.method.as_str()) {
        // Read-only methods (ping/tools/list/prompts/list/initialize) are
        // answered inline so a long-running tool on the worker pool cannot
        // starve client health probes — same rationale as the stdio
        // transport (see `RequestHandler::fast_path_methods`).
        execute(request, handler.as_ref(), req, issued_session);
    } else {
        // `tiny_http::Request` is Send: the whole handler call plus the
        // response write moves onto a dispatcher worker, serializing
        // handler calls exactly like stdio.
        if dispatcher
            .submit(Box::new(move || {
                execute(request, handler.as_ref(), req, issued_session);
                true
            }))
            .is_err()
        {
            // Every worker has exited (unreachable while serving); the task
            // is dropped with the request, which closes the connection
            // instead of hanging the client on a response that never comes.
            tracing::error!("dispatcher unavailable; dropping HTTP request connection");
        }
    }
}

/// Run the handler for one parsed request and answer the connection —
/// inline on the accept thread (fast path) or on a dispatcher worker.
/// Panic isolation is the shared `handle_isolated` (see transport.rs).
fn execute<H: RequestHandler>(
    request: tiny_http::Request,
    handler: &H,
    req: JsonRpcRequest,
    issued_session: Option<String>,
) {
    let response = handle_isolated(handler, req);
    let extra_headers: &[(&str, &str)] = match &issued_session {
        Some(sid) => &[("mcp-session-id", sid.as_str())],
        None => &[],
    };
    respond_with_headers(
        request,
        200,
        &response.serialize(),
        "application/json",
        extra_headers,
    );
}

/// First value of a named header, if present. `name` must be a static
/// literal (tiny_http's `equiv` compares against `&'static str`).
fn header_value(request: &tiny_http::Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str().to_string())
}

impl JsonRpcResponse {
    fn serialize(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| {
            r#"{"jsonrpc":"2.0","id":null,"error":{"code":-32603,"message":"serialize failed"}}"#
                .to_string()
        })
    }
}

fn respond(request: tiny_http::Request, status: u16, body: &str, ctype: &str) {
    respond_with_headers(request, status, body, ctype, &[]);
}

fn respond_with_headers(
    request: tiny_http::Request,
    status: u16,
    body: &str,
    ctype: &str,
    extra_headers: &[(&str, &str)],
) {
    let mut response = tiny_http::Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes())
                .expect("static header is valid"),
        );
    for (name, value) in extra_headers {
        response = response.with_header(
            tiny_http::Header::from_bytes(name.as_bytes(), value.as_bytes())
                .unwrap_or_else(|e| panic!("static header {name} is valid: {e:?}")),
        );
    }
    let _ = request.respond(response);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::server::McpServer;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn start_test_server() -> (std::net::SocketAddr, String) {
        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let addr = match server.server_addr() {
            tiny_http::ListenAddr::IP(a) => a,
            #[cfg(unix)]
            tiny_http::ListenAddr::Unix(_) => panic!("expected IP listener"),
        };
        let token = "test-token".to_string();
        let handler: Arc<McpServer> = Arc::new(McpServer::new());
        std::thread::spawn(move || {
            let _ = serve(server, handler, "test-token", 1);
        });
        (addr, token)
    }

    /// POST returning (status, headers-of-interest, body).
    fn request(
        addr: std::net::SocketAddr,
        token: &str,
        method: &str,
        path: &str,
        session: Option<&str>,
        body: &str,
    ) -> (u16, Vec<(String, String)>, String) {
        let mut stream = TcpStream::connect(addr).unwrap();
        let session_header = session
            .map(|s| format!("mcp-session-id: {s}\r\n"))
            .unwrap_or_default();
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: t\r\nAuthorization: Bearer {token}\r\n{session_header}Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut reader = BufReader::new(stream);
        let mut status_line = String::new();
        reader.read_line(&mut status_line).unwrap();
        let code: u16 = status_line
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        let mut length = 0usize;
        let mut headers = Vec::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let l = line.trim().to_ascii_lowercase();
            if l.is_empty() {
                break;
            }
            if let Some(v) = l.strip_prefix("content-length:") {
                length = v.trim().parse().unwrap_or(0);
            }
            if let Some((name, value)) = l.split_once(':') {
                headers.push((name.trim().to_string(), value.trim().to_string()));
            }
        }
        let mut body_bytes = vec![0u8; length];
        if length > 0 {
            std::io::Read::read_exact(&mut reader, &mut body_bytes).unwrap();
        }
        (
            code,
            headers,
            String::from_utf8_lossy(&body_bytes).into_owned(),
        )
    }

    fn post(addr: std::net::SocketAddr, token: &str, body: &str) -> (u16, String) {
        let (code, _, body) = request(addr, token, "POST", "/mcp", None, body);
        (code, body)
    }

    #[test]
    fn http_roundtrip_ping_and_auth() {
        let (addr, token) = start_test_server();

        // Happy path: tools/list over HTTP (stateless bootstrap — no session
        // has been issued yet on this fresh server).
        let (code, body) = post(
            addr,
            &token,
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
        );
        assert_eq!(code, 200, "body: {body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["result"]["tools"].is_array());

        // Wrong token → 401.
        let (code, _) = post(
            addr,
            "wrong-token",
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        );
        assert_eq!(code, 401);
    }

    #[test]
    fn http_session_lifecycle() {
        let (addr, token) = start_test_server();

        // initialize mints a session id via the response header.
        let (code, headers, body) = request(
            addr,
            &token,
            "POST",
            "/mcp",
            None,
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}"#,
        );
        assert_eq!(code, 200, "body: {body}");
        let sid = headers
            .iter()
            .find(|(n, _)| n == "mcp-session-id")
            .map(|(_, v)| v.clone())
            .expect("initialize must issue mcp-session-id");

        // Without the session header, requests are now rejected (a session
        // exists — the stateless-permissive bootstrap no longer applies).
        let (code, _, body) = request(
            addr,
            &token,
            "POST",
            "/mcp",
            None,
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        );
        assert_eq!(code, 404, "body: {body}");

        // With a bogus session id → 404 too.
        let (code, _, _) = request(
            addr,
            &token,
            "POST",
            "/mcp",
            Some("00000000-0000-0000-0000-000000000000"),
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/list"}"#,
        );
        assert_eq!(code, 404);

        // With the issued session → 200.
        let (code, _, body) = request(
            addr,
            &token,
            "POST",
            "/mcp",
            Some(&sid),
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/list"}"#,
        );
        assert_eq!(code, 200, "body: {body}");

        // DELETE with the session terminates it…
        let (code, _, _) = request(addr, &token, "DELETE", "/mcp", Some(&sid), "");
        assert_eq!(code, 200);
        // …after which the id is unknown.
        let (code, _, _) = request(
            addr,
            &token,
            "POST",
            "/mcp",
            Some(&sid),
            r#"{"jsonrpc":"2.0","id":5,"method":"tools/list"}"#,
        );
        assert_eq!(code, 404);

        // Non-/mcp paths are 404, not treated as MCP.
        let (code, _, _) = request(
            addr,
            &token,
            "POST",
            "/elsewhere",
            None,
            r#"{"jsonrpc":"2.0","id":6,"method":"tools/list"}"#,
        );
        assert_eq!(code, 404);
    }

    #[test]
    fn http_notification_is_202_and_bad_json_is_400() {
        let (addr, token) = start_test_server();
        let (code, _) = post(
            addr,
            &token,
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        );
        assert_eq!(code, 202);

        let (code, body) = post(addr, &token, "not json at all");
        assert_eq!(code, 400);
        assert!(body.contains("-32700"));
    }

    #[test]
    fn http_frame_validation_matches_stdio_codes() {
        // Cross-transport equivalence, HTTP side: the SAME malformed frames
        // stdio rejects (see transport::tests::
        // malformed_frames_are_rejected_identically_across_transports) must
        // get the same JSON-RPC error code here — both route through the one
        // shared `validate_frame`; only the HTTP status wrapper (400) differs.
        // `id: null` is pinned too: a REQUEST (answered), not a notification.
        let (addr, token) = start_test_server();
        let matrix: &[(&str, &str)] = &[
            ("this is not json", "-32700"),
            ("42", "-32600"),
            (r#"{"jsonrpc":"2.0","id":1,"method":42}"#, "-32600"),
            (r#"{"id":1,"method":"ping"}"#, "-32600"),
            (r#"{"jsonrpc":"1.0","id":2,"method":"ping"}"#, "-32600"),
            (r#"[{"jsonrpc":"2.0","id":1,"method":"ping"}]"#, "-32600"),
        ];
        for (frame, code) in matrix {
            let (status, body) = post(addr, &token, frame);
            assert_eq!(status, 400, "frame: {frame} body: {body}");
            assert!(
                body.contains(code),
                "frame {frame} must carry error code {code}, got: {body}"
            );
        }

        // id:null → answered as a request with id null (not a 202).
        let (status, body) = post(
            addr,
            &token,
            r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#,
        );
        assert_eq!(status, 200, "body: {body}");
        assert!(body.contains(r#""id":null"#), "body: {body}");
    }

    #[test]
    fn constant_time_eq_behaves() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"abc", b"ab"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn http_handler_calls_are_serialized_by_worker_pool() {
        // The concurrency contract this transport must honor: even with all
        // ACCEPT_READERS threads accepting at once, handler calls run only
        // on the dispatcher — with one worker, strictly one at a time.
        // (Before the dispatcher, each accept thread ran the handler inline
        // and this overlap check failed.)
        struct ProbeHandler {
            active: AtomicUsize,
            max_active: AtomicUsize,
        }
        impl RequestHandler for ProbeHandler {
            fn handle(&self, req: JsonRpcRequest) -> JsonRpcResponse {
                let now = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_active.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(50));
                self.active.fetch_sub(1, Ordering::SeqCst);
                JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: req.id,
                    result: Some(serde_json::json!({ "echo": req.method })),
                    error: None,
                }
            }
        }

        let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
        let addr = match server.server_addr() {
            tiny_http::ListenAddr::IP(a) => a,
            #[cfg(unix)]
            tiny_http::ListenAddr::Unix(_) => panic!("expected IP listener"),
        };
        let handler = Arc::new(ProbeHandler {
            active: AtomicUsize::new(0),
            max_active: AtomicUsize::new(0),
        });
        let serve_handler = Arc::clone(&handler);
        std::thread::spawn(move || {
            let _ = serve(server, serve_handler, "test-token", 1);
        });

        // Four concurrent POSTs of a NON-fast-path method.
        let start = Arc::new(std::sync::Barrier::new(4));
        let mut clients = Vec::new();
        for i in 0..4 {
            let start = Arc::clone(&start);
            clients.push(std::thread::spawn(move || {
                start.wait();
                let (code, body) = post(
                    addr,
                    "test-token",
                    &format!(r#"{{"jsonrpc":"2.0","id":{i},"method":"slow/{i}"}}"#),
                );
                assert_eq!(code, 200, "body: {body}");
            }));
        }
        for c in clients {
            c.join().unwrap();
        }
        assert_eq!(
            handler.max_active.load(Ordering::SeqCst),
            1,
            "handler calls must be serialized by the worker pool"
        );
    }
}
