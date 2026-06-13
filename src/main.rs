use engram::config::Config;
use engram::graph::GraphEngine;
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
    let config = Config::load().unwrap_or_default();
    tracing::info!("Configuration loaded from {:?}", config.storage.database_path);

    let repo = MemoryRepository::new(&config.storage.database_path)?;
    repo.initialize_schema()?;
    repo.migrate_fts5_add_tags()?;
    tracing::info!("SQLite schema initialized");

    let graph = GraphEngine::new();
    tracing::info!("Graph engine initialized");

    let provider = Arc::new(DefaultMemoryProvider::new(repo, graph, config));

    let server = McpServer::with_provider(provider);
    tracing::info!("Starting engram MCP server on stdio...");
    server.run()?;

    Ok(())
}
