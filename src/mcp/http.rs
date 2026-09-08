//! MCP Streamable-HTTP transport (optional, `[http]` config).
//!
//! A minimal synchronous implementation on `tiny_http` — engram has no async
//! runtime by design, and the subset of the protocol engram needs is small:
//!
//! - `POST /mcp` with a JSON-RPC message → one `application/json` response
//!   (the spec allows plain JSON replies when the server never pushes);
//! - notifications (no `id`) → `202 Accepted`;
//! - no GET/SSE stream (the spec permits `405` for servers without push);
//! - every request must carry `Authorization: Bearer <token>`.
//!
//! Requests are handled on a small accept-thread pool through the same
//! [`RequestHandler`] the stdio transport uses — protocol rules (jsonrpc
//! version check, invalid-request vs parse-error, panic isolation) mirror
//! `transport.rs` so both transports behave identically.

use std::io::Read;
use std::sync::Arc;

use anyhow::{bail, Result};

use super::transport::{JsonRpcError, JsonRpcRequest, JsonRpcResponse, RequestHandler};

/// Largest accepted request body (parity with the stdio frame cap).
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Serve MCP over HTTP forever. Blocks the calling thread.
pub fn run_http<H: RequestHandler + Send + Sync + 'static>(
    handler: Arc<H>,
    bind: &str,
    auth_token: &str,
) -> Result<()> {
    let server = tiny_http::Server::http(bind)
        .map_err(|e| anyhow::anyhow!("HTTP transport failed to bind {bind}: {e}"))?;
    tracing::info!(
        "engram MCP HTTP transport listening on {} (token auth required)",
        server.server_addr()
    );
    serve(server, handler, auth_token)
}

/// Serve on an already-bound server (tests pass their own to learn the port).
pub(crate) fn serve<H: RequestHandler + Send + Sync + 'static>(
    server: tiny_http::Server,
    handler: Arc<H>,
    auth_token: &str,
) -> Result<()> {
    let server = Arc::new(server);
    let mut handles = Vec::new();
    for _ in 0..4 {
        let server = Arc::clone(&server);
        let handler = Arc::clone(&handler);
        let token = auth_token.to_string();
        handles.push(std::thread::spawn(move || loop {
            let Ok(request) = server.recv() else {
                break; // server closed
            };
            handle_request(request, handler.as_ref(), &token);
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

fn handle_request<H: RequestHandler>(mut request: tiny_http::Request, handler: &H, token: &str) {
    // Auth first: never process an unauthenticated body.
    let authorized = request
        .headers()
        .iter()
        .any(|h| h.field.equiv("Authorization") && h.value.as_str() == format!("Bearer {token}"));
    if !authorized {
        respond(
            request,
            401,
            r#"{"error":"unauthorized"}"#,
            "application/json",
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

    let value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let resp = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: None,
                result: None,
                error: Some(JsonRpcError {
                    code: -32700,
                    message: format!("Parse error: {e}"),
                    data: None,
                }),
            };
            respond(request, 400, &resp.serialize(), "application/json");
            return;
        }
    };
    let cached_id = value.get("id").cloned();
    let req = match serde_json::from_value::<JsonRpcRequest>(value) {
        Ok(r) => r,
        Err(e) => {
            let resp = JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id: cached_id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32600,
                    message: format!("Invalid Request: {e}"),
                    data: None,
                }),
            };
            respond(request, 400, &resp.serialize(), "application/json");
            return;
        }
    };
    if req.jsonrpc != "2.0" {
        let resp = JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id: req.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32600,
                message: "Invalid Request: jsonrpc must be \"2.0\"".to_string(),
                data: None,
            }),
        };
        respond(request, 400, &resp.serialize(), "application/json");
        return;
    }
    if req.id.is_none() {
        // Notification: accepted, no body (per Streamable HTTP).
        respond(request, 202, "", "text/plain");
        return;
    }

    // Panic isolation mirrors the stdio transport: a panicking handler
    // yields a JSON-RPC internal error, not a dropped connection.
    let response = {
        let id = req.id.clone();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler.handle(req))) {
            Ok(resp) => resp,
            Err(_) => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: -32603,
                    message: "Internal error: request handler panicked".into(),
                    data: None,
                }),
            },
        }
    };
    respond(request, 200, &response.serialize(), "application/json");
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
    let response = tiny_http::Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes())
                .expect("static header is valid"),
        );
    let _ = request.respond(response);
}

/// Sanity helper for main(): HTTP transport must never start without a token.
pub fn validate_http_config(enabled: bool, bind: &str, token: &str) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    if token.trim().is_empty() {
        bail!(
            "[http] enabled but auth_token is empty — refusing to start an unauthenticated server"
        );
    }
    if bind.trim().is_empty() {
        bail!("[http] enabled but bind is empty");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::server::McpServer;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;

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
            let _ = serve(server, handler, "test-token");
        });
        (addr, token)
    }

    fn post(addr: std::net::SocketAddr, token: &str, body: &str) -> (u16, String) {
        let mut stream = TcpStream::connect(addr).unwrap();
        write!(
            stream,
            "POST /mcp HTTP/1.1\r\nHost: t\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
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
        // headers then body
        let mut length = 0usize;
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
        }
        let mut body_bytes = vec![0u8; length];
        if length > 0 {
            std::io::Read::read_exact(&mut reader, &mut body_bytes).unwrap();
        }
        (code, String::from_utf8_lossy(&body_bytes).into_owned())
    }

    #[test]
    fn http_roundtrip_ping_and_auth() {
        let (addr, token) = start_test_server();

        // Happy path: tools/list over HTTP.
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
    fn http_config_requires_token_when_enabled() {
        assert!(validate_http_config(true, "127.0.0.1:8742", "").is_err());
        assert!(validate_http_config(true, "127.0.0.1:8742", "secret").is_ok());
        assert!(validate_http_config(false, "", "").is_ok());
    }
}
