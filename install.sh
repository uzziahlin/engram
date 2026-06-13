#!/bin/sh
# engram installer — POSIX sh (macOS, Linux, WSL, Git Bash)
#
#   curl -fsSL https://raw.githubusercontent.com/uzziahlin/engram/main/install.sh | sh
#
# Downloads the prebuilt binary for your platform, installs it to
# ~/.engram/bin/engram, initializes the database, and auto-configures any
# detected MCP client (Claude Code, Cursor, Windsurf, Codex CLI).
#
# Env overrides:
#   ENGRAM_VERSION=<x.y.z>   install a specific version
#   ENGRAM_INSTALL_DIR=<dir> install location (default ~/.engram/bin)
set -eu

REPO="uzziahlin/engram"
INSTALL_DIR="${ENGRAM_INSTALL_DIR:-$HOME/.engram/bin}"
BIN_NAME="engram"

# --- output helpers ---
info() { printf '\033[1;34m==>\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$1" >&2; }
ok()   { printf '\033[1;32m✓\033[0m %s\n' "$1"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$1" >&2; exit 1; }

# --- platform detection ---
detect() {
  OS="$(uname -s)"
  ARCH="$(uname -m)"
  case "$OS" in
    Darwin)
      case "$ARCH" in
        arm64|aarch64) TARGET="aarch64-apple-darwin"; ARCHIVE="tar.gz" ;;
        x86_64)        TARGET="x86_64-apple-darwin";  ARCHIVE="tar.gz" ;;
        *) die "Unsupported macOS arch: $ARCH" ;;
      esac ;;
    Linux)
      case "$ARCH" in
        x86_64|amd64)  TARGET="x86_64-unknown-linux-gnu"; ARCHIVE="tar.gz" ;;
        aarch64|arm64) TARGET="aarch64-unknown-linux-gnu"; ARCHIVE="tar.gz" ;;
        *) die "Unsupported Linux arch: $ARCH" ;;
      esac ;;
    MINGW*|MSYS*|CYGWIN*)
      case "$ARCH" in
        x86_64) TARGET="x86_64-pc-windows-msvc"; ARCHIVE="zip"; BIN_NAME="engram.exe" ;;
        *) die "Unsupported Windows arch: $ARCH" ;;
      esac ;;
    *) die "Unsupported OS: $OS (use macOS, Linux, or WSL/Git Bash)" ;;
  esac
}

# --- resolve version (default: latest release) ---
resolve_version() {
  VERSION="${ENGRAM_VERSION:-}"
  if [ -z "$VERSION" ]; then
    info "Querying latest release..."
    VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
      | sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' | head -n1)"
    [ -n "$VERSION" ] || die "Could not determine latest version (set ENGRAM_VERSION manually)"
  fi
}

# --- download + verify + install ---
download_and_install() {
  ASSET="engram-${VERSION}-${TARGET}.${ARCHIVE}"
  URL="https://github.com/$REPO/releases/download/v${VERSION}/${ASSET}"
  info "Downloading engram v${VERSION} ($TARGET)..."

  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  curl -fsSL -o "$TMP/$ASSET" "$URL" || die "Download failed: $URL"

  # Optional checksum verification
  if curl -fsSL -o "$TMP/$ASSET.sha256" "$URL.sha256" 2>/dev/null; then
    if command -v sha256sum >/dev/null 2>&1; then
      (cd "$TMP" && sha256sum -c "$ASSET.sha256") >/dev/null 2>&1 || die "Checksum mismatch"
      ok "Checksum verified"
    elif command -v shasum >/dev/null 2>&1; then
      (cd "$TMP" && shasum -a 256 -c "$ASSET.sha256") >/dev/null 2>&1 || die "Checksum mismatch"
      ok "Checksum verified"
    else
      warn "No sha256 tool found; skipping checksum verification"
    fi
  fi

  mkdir -p "$INSTALL_DIR"
  case "$ARCHIVE" in
    tar.gz)
      tar -xzf "$TMP/$ASSET" -C "$TMP"
      find "$TMP" -name "$BIN_NAME" -type f -exec cp {} "$INSTALL_DIR/$BIN_NAME" \;
      ;;
    zip)
      command -v unzip >/dev/null 2>&1 || die "unzip is required to install on Windows"
      (cd "$TMP" && unzip -o "$ASSET" >/dev/null)
      find "$TMP" -name "$BIN_NAME" -type f -exec cp {} "$INSTALL_DIR/$BIN_NAME" \;
      ;;
  esac
  chmod +x "$INSTALL_DIR/$BIN_NAME" 2>/dev/null || true

  # macOS Gatekeeper: curl-downloaded binaries get quarantined and may be
  # blocked on first run. Strip the quarantine extended attribute.
  if [ "$OS" = "Darwin" ]; then
    xattr -d com.apple.quarantine "$INSTALL_DIR/$BIN_NAME" 2>/dev/null || true
  fi
}

# --- initialize the database ---
run_init() {
  if "$INSTALL_DIR/$BIN_NAME" init >/dev/null 2>&1; then
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
      case "$(basename "$SHELL")" in
        zsh)  printf '    echo '\''export PATH="$HOME/.engram/bin:$PATH"'\'' >> ~/.zshrc\n' ;;
        bash) printf '    echo '\''export PATH="$HOME/.engram/bin:$PATH"'\'' >> ~/.bashrc\n' ;;
        fish) printf '    fish_add_path ~/.engram/bin\n' ;;
        *)    printf '    add %s to your PATH\n' "$INSTALL_DIR" ;;
      esac ;;
  esac
}

abs_bin() { echo "$INSTALL_DIR/$BIN_NAME"; }

# --- MCP client auto-configuration (idempotent, absolute path) ---
# Writes {"mcpServers":{"engram":{"command":"<abs>","args":[]}}} into a config
# file, merging with python3 (ubiquitous on macOS/Linux) so existing entries
# are preserved and re-running the installer never duplicates engram.
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

configure_clients() {
  info "Configuring MCP clients (idempotent)..."
  BIN="$(abs_bin)"

  # Claude Code — primary target. Prefer the CLI when available, then ensure
  # the global config file regardless.
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

  # Only touch a client's config if that client appears to be installed.
  [ -d "$HOME/.cursor" ]   && write_mcp_json "$HOME/.cursor/mcp.json"
  [ -d "$HOME/.codeium" ]  && write_mcp_json "$HOME/.codeium/windsurf/mcp_config.json"
  [ -d "$HOME/.codex" ]    && write_mcp_json "$HOME/.codex/mcp.json"

  ok "MCP clients configured. Restart your editor(s) to activate engram."
}

main() {
  info "Installing engram..."
  detect
  resolve_version
  download_and_install
  run_init
  ensure_path
  configure_clients
  echo
  ok "engram v${VERSION} installed at $(abs_bin)"
  echo "    Run '$(abs_bin) --help' or restart your MCP client."
}

main "$@"
