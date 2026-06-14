pub mod commands;

use anyhow::Result;

/// Run the CLI with the given arguments.
pub fn run(args: &[String]) -> Result<()> {
    if args.is_empty() {
        anyhow::bail!("Usage: engram <command> [options]\n\nCommands:\n  search         Search memories\n  create-episodic  Create episodic memory\n  create-decision  Create decision memory\n  create-failure   Create failure memory\n  create-procedural Create procedural memory\n  ingest         Ingest git commits\n  collect        Collect bootstrap evidence from a project\n  recent-failures List recent failures\n  decisions      List architectural decisions\n  timeline       Show project timeline\n  init           Initialize database");
    }

    let command = &args[0];
    let cmd_args = &args[1..];

    match command.as_str() {
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
        "init" => commands::init(cmd_args),
        _ => anyhow::bail!("Unknown command: {command}\nRun 'engram' without arguments for help."),
    }
}
