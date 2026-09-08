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

    let worker_threads = config.mcp.worker_threads;
    let provider = Arc::new(DefaultMemoryProvider::new(repo, config));

    let server = McpServer::with_provider_and_workers(provider, worker_threads);
    tracing::info!("Starting engram MCP server on stdio ({worker_threads} workers)...");
    Arc::new(server).run()?;

    Ok(())
}
