# Engram Memory — Agent Guide

This project uses **engram**, a local MCP memory server, for long-term
engineering memory. Its tools are in your MCP tool list — use them
proactively, don't wait to be asked. Every call needs project_id: "engram".

**Session rhythm:** before starting a task, search_memory for prior context;
after finishing meaningful work, write the matching memory below.

## When to WRITE
- Finished a feature/task → create_episodic
  (set files_touched; importance: small fix ~0.3, core feature ≥0.7)
- Made a design/architecture decision → create_decision (context, rationale, tradeoffs)
- Fixed a bug or resolved an incident → create_failure (root_cause, fix, prevention; severity 1–5)
- Established a workflow/convention → create_procedural (steps)

## When to READ
- Before a new task → search_memory
- Before modifying a file → related_files
- Need design background → architectural_decisions
- Avoid repeating past bugs → recent_failures

## Search tips (`search_memory`)

Every search covers **all** memory types — intent keywords only adjust ranking
weights (a Debugging query boosts failures, an Architecture query boosts
decisions), they never hide a type. Results carry each memory's full payload
(a failure's root_cause/fix/prevention, a decision's rationale/tradeoffs…),
so no follow-up fetch is normally needed; use `get_memory` when you want the
complete record for one id.

Filters: `tags` (any-match) and `before` (unix timestamp) narrow results when
you know the categorization or recency you want.

Tokenization: FTS5 splits on `-`/`_`/punctuation, so `auth-utils` matches both
`auth` and `utils`. CJK is matched per-character.

When results are thin: widen the wording, try a tag or identifier as the
query, or pivot via `related_files` (memories touching a file).

## Feedback tools (read the store's health)

- `recent_failures` — before chasing a bug, check a similar failure is already
  documented (don't re-solve it).
- `query_stats` — which queries run often and their average hit count. A query
  with high count but low `result_count_avg` means the store isn't satisfying
  it → a signal to write the missing memory or refine tags.

## Maintenance (human-side, keep the store healthy)

The user can run `engram maintain` (dedup + FTS repair + query-log prune +
staleness report) on a schedule. As the agent, your part: keep tags small and
consistent, and don't re-record what's already stored (search first).

Full parameter schemas come from MCP; this guide covers *when* to use each tool
and project conventions. All tools require project_id = "engram".
