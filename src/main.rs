use engram::config::Config;
use engram::mcp::server::{DefaultMemoryProvider, McpServer};
use engram::storage::MemoryRepository;
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    // Initialize tracing — write to stderr so MCP JSON-RPC on stdout is not corrupted
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        // No arguments → run as MCP server (stdio)
        run_mcp_server()
    } else {
        // Has arguments → run as CLI
        engram::cli::run(&args)
    }
}

fn run_mcp_server() -> anyhow::Result<()> {
    // Broken config is fatal, not a silent default fallback: with the old
    // warn-and-continue, the server wrote to the default database while the
    // CLI (which fails hard on broken config) read another — user data got
    // split across two stores. Failing fast on startup matches the CLI and
    // surfaces the config error to the client immediately.
    let config = Config::load()?;
    tracing::info!(
        "Configuration loaded from {:?}",
        config.storage.database_path
    );

    let repo = MemoryRepository::new(&config.storage.database_path)?;
    repo.initialize_schema()?;
    tracing::info!("SQLite schema initialized");

    let use_http = config.http.enabled;
    let worker_threads = config.mcp.worker_threads;
    let http_bind = config.http.bind.clone();
    let http_token = config.http.auth_token.clone();
    let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

    let server = McpServer::with_provider_and_workers(provider, worker_threads);
    if use_http {
        // Optional Streamable-HTTP transport ([http] enabled + token, which
        // Config::load has already validated). Same handler as stdio; the
        // transports differ only in framing (see src/mcp/http.rs). The
        // worker count must be passed through: HTTP serializes handler
        // calls through the same bounded dispatcher as stdio, so the log
        // line must describe both reader and worker counts truthfully.
        tracing::info!(
            "Starting engram MCP server on HTTP {http_bind} ({} accept readers + {worker_threads} workers, token auth)...",
            engram::mcp::http::ACCEPT_READERS
        );
        engram::mcp::http::run_http(Arc::new(server), &http_bind, &http_token, worker_threads)?;
    } else {
        tracing::info!("Starting engram MCP server on stdio ({worker_threads} workers)...");
        Arc::new(server).run()?;
    }

    Ok(())
}
