#!/bin/sh
# engram installer — POSIX sh (macOS, Linux, WSL, Git Bash)
#
#   curl -fsSL https://raw.githubusercontent.com/uzziahlin/engram/main/install.sh | sh
#
# Downloads the prebuilt binary for your platform, installs it to
# ~/.engram/bin/engram, initializes the database, and auto-configures any
# detected MCP client (Claude Code, Cursor, Windsurf, Codex CLI).
#
# The post-download logic (init + PATH hint + MCP config) is shared with the
# Makefile via scripts/engram-common.sh. When run via curl|sh that file isn't
# on disk, so it's fetched from the same repo/ref at runtime.
#
# Env overrides:
#   ENGRAM_VERSION=<x.y.z>   install a specific version
#   ENGRAM_INSTALL_DIR=<dir> install location (default ~/.engram/bin)
#   ENGRAM_CLIENTS=<list>    MCP clients to configure (default "auto"; see common lib)
#   ENGRAM_REF=<git-ref>     ref to fetch engram-common.sh from (default "main")
set -eu

REPO="uzziahlin/engram"

# --- load shared library (single source of truth for init + MCP config) ---
# Local checkout: source the file sitting next to this script. Via curl|sh, $0
# is not a repo path so the file is absent — fetch it from raw at the same ref.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" 2>/dev/null && pwd || true)
if [ -n "${SCRIPT_DIR:-}" ] && [ -f "$SCRIPT_DIR/scripts/engram-common.sh" ]; then
  . "$SCRIPT_DIR/scripts/engram-common.sh"
else
  _ref="${ENGRAM_REF:-main}"
  _url="https://raw.githubusercontent.com/$REPO/$_ref/scripts/engram-common.sh"
  _tmp="$(mktemp)"
  curl -fsSL -o "$_tmp" "$_url" || {
    printf '\033[1;31merror:\033[0m %s\n' "Could not fetch shared library: $_url" >&2
    exit 1
  }
  . "$_tmp"
  rm -f "$_tmp"
fi

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
  # Users may pin with or without the leading 'v' (v0.6.0 or 0.6.0); strip it
  # so the download URL is never .../download/vv0.6.0/...
  VERSION="${VERSION#v}"
  if [ -z "$VERSION" ]; then
    info "Querying latest release..."
    VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
      | sed -n 's/.*"tag_name": *"v\{0,1\}\([^"]*\)".*/\1/p' | head -n1)"
    [ -n "$VERSION" ] || die "Could not determine latest version (set ENGRAM_VERSION manually; \
note anonymous GitHub API calls are rate-limited to 60/hour per IP)"
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

  # Checksum verification — fail-closed. For a `curl | sh` installer the
  # checksum file is the only integrity guarantee; a MISSING .sha256 asset
  # previously fell through silently and installed an unverified binary.
  if ! curl -fsSL -o "$TMP/$ASSET.sha256" "$URL.sha256" 2>/dev/null; then
    die "Checksum file not found: $URL.sha256 — refusing to install an unverified binary"
  fi
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$TMP" && sha256sum -c "$ASSET.sha256") >/dev/null 2>&1 || die "Checksum mismatch"
    ok "Checksum verified"
  elif command -v shasum >/dev/null 2>&1; then
    (cd "$TMP" && shasum -a 256 -c "$ASSET.sha256") >/dev/null 2>&1 || die "Checksum mismatch"
    ok "Checksum verified"
  else
    die "No sha256 tool found (need sha256sum or shasum) — cannot verify the download"
  fi

  case "$ARCHIVE" in
    tar.gz)
      tar -xzf "$TMP/$ASSET" -C "$TMP"
      ;;
    zip)
      command -v unzip >/dev/null 2>&1 || die "unzip is required to install on Windows"
      (cd "$TMP" && unzip -o "$ASSET" >/dev/null)
      ;;
  esac

  # Locate the extracted binary and install it atomically (mv, not in-place cp)
  # so we never corrupt a copy a running MCP client still holds open. macOS
  # quarantine stripping happens inside install_binary.
  _bin="$(find "$TMP" -name "$BIN_NAME" -type f | head -n1)"
  [ -n "$_bin" ] || die "binary '$BIN_NAME' not found in downloaded archive"
  install_binary "$_bin"
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
