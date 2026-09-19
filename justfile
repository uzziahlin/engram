# engram — build from source and install, mirroring install.sh.
#
# install.sh downloads a prebuilt binary; this justfile builds it locally with
# cargo and then runs the exact same post-install steps (init + MCP config) by
# sourcing scripts/engram-common.sh — the single source of truth shared with
# install.sh.
#
# Quick start:
#   just                     # build -> install -> init -> configure (full install.sh equivalent)
#   just install             # build + copy the binary into install_dir
#   just configure clients="claude cursor"   # configure only specific MCP clients
#   just uninstall           # remove binary + engram entries from MCP configs
#   just uninstall purge=1   # also delete the ~/.engram database
#
# Overridable variables (via `just name=value` on the command line, or the
# equivalent environment variable):
#   install_dir  install location for the binary   (default ~/.engram/bin)
#   data_dir     engram database / state directory  (default ~/.engram)
#   clients      MCP clients to configure: "auto" or a space-separated subset of
#                {claude cursor windsurf codex}     (default auto)
#   purge        with `uninstall`, set to 1 to also remove data_dir

home        := env_var('HOME')
install_dir := env_var_or_default('INSTALL_DIR', home / '.engram' / 'bin')
data_dir    := env_var_or_default('DATA_DIR', home / '.engram')
clients     := env_var_or_default('CLIENTS', 'auto')
purge       := env_var_or_default('PURGE', '0')
common      := 'scripts/engram-common.sh'

# Resolve cargo's output directory authoritatively (honors CARGO_TARGET_DIR and
# .cargo/config's build.target-dir) rather than assuming ./target; fall back to
# CARGO_TARGET_DIR or ./target if `cargo metadata`/python3 are unavailable.
# Eager (single backtick) evaluation: resolved once when just loads the file.
target_dir := `cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["target_directory"])' 2>/dev/null || echo "${CARGO_TARGET_DIR:-target}"`
target_bin := target_dir / 'release' / 'engram'

# build, install, init the DB, and configure MCP clients (default)
default: build install init configure
    @printf '\033[1;32m✓\033[0m %s\n' "engram installed at {{install_dir}}/engram — restart your MCP client."

# compile the release binary
build:
    cargo build --release --locked

# copy the freshly built binary into install_dir
install: build
    @ENGRAM_INSTALL_DIR="{{install_dir}}" sh -c '. {{common}}; install_binary "{{target_bin}}"; ok "Installed $(abs_bin)"; ensure_path'

# initialize the engram database (engram init)
init:
    @ENGRAM_INSTALL_DIR="{{install_dir}}" sh -c '. {{common}}; run_init'

# register engram with MCP clients (idempotent; honors clients)
configure:
    @ENGRAM_INSTALL_DIR="{{install_dir}}" ENGRAM_CLIENTS="{{clients}}" sh -c '. {{common}}; configure_clients'

# remove the binary and engram entries from MCP configs (purge=1 also drops data_dir)
uninstall:
    #!/usr/bin/env sh
    set -eu
    rm -f "{{install_dir}}/engram"
    printf '\033[1;32m✓\033[0m %s\n' "Removed {{install_dir}}/engram"
    ENGRAM_INSTALL_DIR="{{install_dir}}" sh -c '. {{common}}; unconfigure_clients'
    if [ "{{purge}}" = "1" ]; then
      rm -rf "{{data_dir}}"
      printf '\033[1;33m!!\033[0m %s\n' "Purged database at {{data_dir}}"
    fi

# remove cargo build artifacts
clean:
    cargo clean

# list available recipes
help:
    @just --list
