mod commands;

use anyhow::Result;

/// Usage text for `engram help`/`--help`/`-h`. (No-argument invocation starts
/// the MCP server — see `main.rs` — so this is reached only via an explicit
/// help flag or an unknown command.)
const USAGE: &str = "Usage: engram <command> [options]\n\nCommands:\n  search            Search memories\n  create-episodic   Create episodic memory\n  create-decision   Create decision memory\n  create-failure    Create failure memory\n  create-procedural Create procedural memory\n  ingest            Ingest git commits\n  collect           Collect bootstrap evidence from a project\n  recent-failures   List recent failures\n  decisions         List architectural decisions\n  timeline          Show project timeline\n  queries           Show popular search queries (retrieval feedback)\n  forget            Archive (soft-delete) a memory by id\n  restore           Un-archive a memory by id\n  update            Patch fields of a memory (--set key=value)\n  forget-batch      Archive memories by --tag/--before (dry-run; --apply to commit)\n  list-archived     List archived memories\n  gc                Garbage-collect archived memories + reclaim WAL space (dry-run; --apply)\n  consolidate       Find/merge duplicate memories (--near, --apply)\n  reflect           Propose preventive rules from recurring failures (dry-run; --apply)\n  suggestions       List pending reflection proposals awaiting confirmation\n  confirm-suggestion Promote a pending proposal into a searchable procedural memory\n  reject-suggestion Discard a pending proposal without creating a procedural memory\n  reindex           Backfill embeddings for existing memories (semantic build)\n  init              Initialize database\n  init-guide        Generate ENGRAM.md agent guide; optionally @import into CLAUDE.md\n\nRun 'engram' with no arguments to start the MCP server (stdio).";

/// Run the CLI with the given arguments.
pub fn run(args: &[String]) -> Result<()> {
    // Defensive: main() routes the no-argument case to the MCP server, so an
    // empty slice should not reach here. Print usage rather than panic/index.
    if args.is_empty() {
        println!("{USAGE}");
        return Ok(());
    }

    let command = &args[0];
    let cmd_args = &args[1..];

    match command.as_str() {
        "help" | "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        "search" => commands::search(cmd_args),
        "create-episodic" => commands::create_episodic(cmd_args),
        "create-decision" => commands::create_decision(cmd_args),
        "create-failure" => commands::create_failure(cmd_args),
        "create-procedural" => commands::create_procedural(cmd_args),
        "ingest" => commands::ingest(cmd_args),
        "collect" => commands::collect(cmd_args),
        "recent-failures" => commands::recent_failures(cmd_args),
        "decisions" => commands::decisions(cmd_args),
        "timeline" => commands::timeline(cmd_args),
        "queries" => commands::queries(cmd_args),
        "forget" => commands::forget(cmd_args),
        "restore" => commands::restore(cmd_args),
        "update" => commands::update(cmd_args),
        "forget-batch" => commands::forget_batch(cmd_args),
        "list-archived" => commands::list_archived(cmd_args),
        "gc" => commands::gc(cmd_args),
        "consolidate" => commands::consolidate(cmd_args),
        "reflect" => commands::reflect(cmd_args),
        "suggestions" => commands::suggestions(cmd_args),
        "confirm-suggestion" => commands::confirm_suggestion(cmd_args),
        "reject-suggestion" => commands::reject_suggestion(cmd_args),
        "reindex" => commands::reindex(cmd_args),
        "init" => commands::init(cmd_args),
        "init-guide" => commands::init_guide(cmd_args),
        _ => anyhow::bail!("Unknown command: {command}\nRun 'engram help' for usage."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_flags_print_usage_and_succeed() {
        assert!(run(&["--help".to_string()]).is_ok());
        assert!(run(&["-h".to_string()]).is_ok());
        assert!(run(&["help".to_string()]).is_ok());
    }

    #[test]
    fn usage_lists_init_guide() {
        assert!(USAGE.contains("init-guide"));
    }

    #[test]
    fn unknown_command_errors() {
        assert!(run(&["--nonsense".to_string()]).is_err());
    }

    #[test]
    fn usage_lists_lifecycle_commands() {
        for c in [
            "forget",
            "restore",
            "update",
            "forget-batch",
            "list-archived",
            "consolidate",
        ] {
            assert!(USAGE.contains(c), "USAGE missing {c}");
        }
    }

    #[test]
    fn usage_lists_reindex() {
        assert!(USAGE.contains("reindex"));
    }

    #[test]
    fn usage_lists_gc() {
        assert!(USAGE.contains("Garbage-collect"));
    }

    #[test]
    fn usage_lists_reflection_commands() {
        for c in [
            "reflect",
            "suggestions",
            "confirm-suggestion",
            "reject-suggestion",
        ] {
            assert!(USAGE.contains(c), "USAGE missing {c}");
        }
    }
}
