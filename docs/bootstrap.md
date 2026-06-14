# Engram Project Memory Bootstrap

You are initializing long-term memory for an existing project so future sessions start with context instead of from zero. Your job: distill the project's accumulated engineering knowledge into structured engram memories.

- **Target project:** `{{PROJECT_ID}}`
- **Repository path:** `{{REPO_PATH}}`
- **Dimensions in scope:** {{DIMENSIONS}}

## Iron rules

1. **Quality over quantity.** A few precise, well-sourced memories beat hundreds of noisy ones. If you cannot justify a memory with concrete evidence, drop it.
2. **Every memory must be traceable.** Reference the file path / line / commit hash you derived it from, in the memory's fields or tags.
3. **Evidence, not invention.** Work only from the material `collect_sources` returns. Never fabricate a decision or a root cause to fill a quota.
4. **Tag what you create** with `bootstrap` plus a dimension tag (`git` / `decisions` / `failures` / `workflow`), and use `session_id = "bootstrap"`. This makes the result reviewable and cleanable.

## Procedure

1. Call the `collect_sources` tool with the project id, repo path, and dimensions above. It returns structured evidence grouped by dimension — raw material only; it does not summarize.
2. Work each dimension against its mapping and quality bar below.
3. For each accepted item, call the matching `create_*` tool.
4. When finished, report a short table: type | recorded | dropped | top reason for drops.

## Dimension → memory type mapping & quality bars

### `decisions` → `create_decision`
Source signals: README "design decisions" sections, ADR/decision files, `//!` module docs, `// WHY:` / `// NOTE:` comments, dependency choices.

- **Required:** `rationale` must state *why*, not just *what*. "We use SQLite" is a fact; "SQLite — local-first, no server or credentials needed" is a decision. Drop it if you cannot articulate the rationale.
- `title` is a decision ("Use gix over libgit2 for a self-contained binary"), not a phenomenon.
- Cap: keep the **≤15** most consequential. Prefer architecture choices, library/tech selection, and explicit tradeoffs.
- Put source files in `related_files`.

### `failures` → `create_failure`
Source signals: `fix:` / `hotfix:` / `revert:` commits, CHANGELOG "Fixed" sections, error-named tests.

- **Required:** `root_cause` **and** `prevention`. Drop the candidate if you cannot infer both — a "fix:" commit alone is insufficient.
- `severity`: 5 = data loss / security / outage; 3 = feature bug; 1 = cosmetic.
- Prefer failures likely to recur or with non-obvious causes.
- Cite the fix commit hash inside `incident`; cap: **≤20**.

### `workflow` → `create_procedural`
Source signals: CI pipelines (`.github/workflows/`, `.gitlab-ci.yml`, …), Makefile/justfile/scripts, lint & format config, CONTRIBUTING.

- `steps` must be ordered and executable. One procedural per real workflow (CI, release, local setup, test command, commit convention).
- List tools in `related_tools`. Cap: **≤10**.

### `git` → `create_episodic`
Source signals: themed `milestones` and `migrations` from the git collection.

- Collapse each **milestone** into ONE episodic memory, not one per commit. Summarize the theme, time span, and outcome; list representative files and commit hashes.
- Set `importance` ≥ 0.8 for milestones flagged `has_breaking` or listed in `migrations`.
- Skip pure churn (formatting, dependency bumps, typo fixes) unless it reveals something durable.
- Use `related_commits` + `files_touched`. Cap: **~15–25** episodic memories even for large histories.

## When you stop

Stop when you have processed all evidence. Do not invent items to reach a cap — the caps are upper bounds, not targets. If a dimension produced no usable material, say so in the report rather than padding it.
