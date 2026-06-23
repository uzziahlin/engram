# engram shared install/configure library — POSIX sh (no shebang: meant to be sourced)
#
# Single source of truth for the "what to do after we have a binary" logic:
# database init, PATH hint, and MCP-client (un)configuration. Sourced by both
#   - install.sh  (download path; fetched at runtime when run via curl|sh), and
#   - Makefile    (build-from-source path).
# so this logic is never duplicated.
#
# Callers may pre-set / override via env: ENGRAM_INSTALL_DIR, BIN_NAME, REPO,
# ENGRAM_CLIENTS. This file intentionally does NOT `set -e`/`set -u` — that is
# the caller's choice (install.sh runs under `set -eu`).

# --- variable defaults (overridable via env so every caller can reuse) ---
INSTALL_DIR="${ENGRAM_INSTALL_DIR:-$HOME/.engram/bin}"
BIN_NAME="${BIN_NAME:-engram}"
REPO="${REPO:-uzziahlin/engram}"

# --- output helpers ---
info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$1" >&2; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$1"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$1" >&2; exit 1; }

abs_bin() { echo "$INSTALL_DIR/$BIN_NAME"; }

# --- atomic binary install ---
# Install SRC into $INSTALL_DIR/$BIN_NAME via a temp file + mv (rename), never
# truncating a binary a running process still holds open. Overwriting a live
# inode in place (plain `cp`) corrupts its code pages and the next exec gets
# SIGKILL'd on macOS; an atomic rename swaps in a fresh inode instead.
install_binary() {
  _src="$1"
  mkdir -p "$INSTALL_DIR"
  _dst="$INSTALL_DIR/$BIN_NAME"
  _tmp="$_dst.tmp.$$"
  cp "$_src" "$_tmp" || die "failed to copy $_src to $_tmp"
  chmod +x "$_tmp" 2>/dev/null || true
  # macOS: strip the quarantine xattr before the file becomes the target.
  if [ "$(uname -s)" = "Darwin" ]; then
    xattr -d com.apple.quarantine "$_tmp" 2>/dev/null || true
  fi
  mv -f "$_tmp" "$_dst"
}

# --- initialize the database ---
run_init() {
  if "$(abs_bin)" init >/dev/null 2>&1; then
    ok "Initialized database at ~/.engram/"
  else
    warn "engram init did not complete cleanly (fine if already initialized)"
  fi
}

# --- PATH hint ---
ensure_path() {
  case ":$PATH:" in
    *":$INSTALL_DIR:"*) ;;
    *)
      warn "$INSTALL_DIR is not on your PATH. Add it:"
      case "$(basename "${SHELL:-}")" in
        zsh)  printf '    echo '\''export PATH="$HOME/.engram/bin:$PATH"'\'' >> ~/.zshrc\n' ;;
        bash) printf '    echo '\''export PATH="$HOME/.engram/bin:$PATH"'\'' >> ~/.bashrc\n' ;;
        fish) printf '    fish_add_path ~/.engram/bin\n' ;;
        *)    printf '    add %s to your PATH\n' "$INSTALL_DIR" ;;
      esac ;;
  esac
}

# --- MCP client config write (idempotent, absolute path) ---
# Writes {"mcpServers":{"engram":{"command":"<abs>","args":[]}}} into a config
# file, merging with python3 (ubiquitous on macOS/Linux) so existing entries
# are preserved and re-running never duplicates engram.
write_mcp_json() {
  CFG="$1"
  BIN="$(abs_bin)"
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$CFG" "$BIN" <<'PY'
import json, os, sys
path, bin_path = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(path):
    try:
        with open(path) as f: data = json.load(f)
    except Exception: data = {}
servers = data.setdefault("mcpServers", {})
if "engram" in servers:
    print(f"  engram already present in {path}")
else:
    servers["engram"] = {"command": bin_path, "args": []}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(data, f, indent=2)
    print(f"  added engram to {path}")
PY
  else
    warn "python3 not found; cannot safely edit $CFG — add engram manually"
  fi
}

# --- MCP client config removal (idempotent) ---
remove_mcp_json() {
  CFG="$1"
  [ -f "$CFG" ] || return 0
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$CFG" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as f: data = json.load(f)
except Exception:
    sys.exit(0)
servers = data.get("mcpServers")
if isinstance(servers, dict) and "engram" in servers:
    del servers["engram"]
    with open(path, "w") as f: json.dump(data, f, indent=2)
    print(f"  removed engram from {path}")
else:
    print(f"  engram not present in {path}")
PY
  else
    warn "python3 not found; cannot edit $CFG — remove engram manually"
  fi
}

# --- per-client helpers ---
# Claude Code is the primary target: prefer the CLI when available, then ensure
# the global config file regardless.
_configure_claude() {
  BIN="$(abs_bin)"
  if command -v claude >/dev/null 2>&1; then
    if claude mcp list 2>/dev/null | grep -q '^engram'; then
      ok "engram already registered with Claude Code"
    else
      if claude mcp add engram -- "$BIN" >/dev/null 2>&1; then
        ok "Registered engram with Claude Code (claude mcp add)"
      else
        warn "claude mcp add failed; writing config file instead"
      fi
    fi
  fi
  write_mcp_json "$HOME/.claude.json"
}

_cursor_cfg()   { echo "$HOME/.cursor/mcp.json"; }
_windsurf_cfg() { echo "$HOME/.codeium/windsurf/mcp_config.json"; }
_codex_cfg()    { echo "$HOME/.codex/mcp.json"; }

# --- configure MCP clients ---
# ENGRAM_CLIENTS controls scope (default "auto"):
#   auto                       -> Claude Code always; cursor/windsurf/codex only
#                                 when their config dir exists (install.sh behavior)
#   space-separated token list -> configure exactly those, unconditionally
#                                 (known tokens: claude cursor windsurf codex)
configure_clients() {
  info "Configuring MCP clients (idempotent)..."
  clients="${ENGRAM_CLIENTS:-auto}"
  if [ "$clients" = "auto" ]; then
    _configure_claude
    [ -d "$HOME/.cursor" ]  && write_mcp_json "$(_cursor_cfg)"
    [ -d "$HOME/.codeium" ] && write_mcp_json "$(_windsurf_cfg)"
    [ -d "$HOME/.codex" ]   && write_mcp_json "$(_codex_cfg)"
  else
    for c in $clients; do
      case "$c" in
        claude)   _configure_claude ;;
        cursor)   write_mcp_json "$(_cursor_cfg)" ;;
        windsurf) write_mcp_json "$(_windsurf_cfg)" ;;
        codex)    write_mcp_json "$(_codex_cfg)" ;;
        *)        warn "Unknown client '$c' (known: claude cursor windsurf codex)" ;;
      esac
    done
  fi
  ok "MCP clients configured. Restart your editor(s) to activate engram."
}

# --- unconfigure MCP clients (used by `make uninstall`) ---
# Removal is safe and idempotent, so it always sweeps every known client.
unconfigure_clients() {
  info "Removing engram from MCP clients..."
  if command -v claude >/dev/null 2>&1; then
    if claude mcp remove engram >/dev/null 2>&1; then
      ok "Unregistered engram from Claude Code (claude mcp remove)"
    fi
  fi
  remove_mcp_json "$HOME/.claude.json"
  remove_mcp_json "$(_cursor_cfg)"
  remove_mcp_json "$(_windsurf_cfg)"
  remove_mcp_json "$(_codex_cfg)"
}
