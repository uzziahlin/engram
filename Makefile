# engram — build from source and install, mirroring install.sh.
#
# install.sh downloads a prebuilt binary; this Makefile builds it locally with
# cargo and then runs the exact same post-install steps (init + MCP config) by
# sourcing scripts/engram-common.sh — the single source of truth shared with
# install.sh.
#
# Quick start:
#   make            # build -> install -> init -> configure (full install.sh equivalent)
#   make install    # just build + copy the binary into INSTALL_DIR
#   make configure CLIENTS="claude cursor"   # configure only specific MCP clients
#   make uninstall  # remove binary + engram entries from MCP configs
#   make uninstall PURGE=1                    # also delete the ~/.engram database
#
# Overridable variables:
#   INSTALL_DIR  install location for the binary   (default ~/.engram/bin)
#   DATA_DIR     engram database / state directory  (default ~/.engram)
#   CLIENTS      MCP clients to configure: "auto" or a space-separated subset of
#                {claude cursor windsurf codex}     (default auto)
#   PURGE        with `uninstall`, set to 1 to also remove DATA_DIR
#   TARGET_DIR   cargo output dir (default: auto-detected via `cargo metadata`)

SHELL       := /bin/sh
INSTALL_DIR ?= $(HOME)/.engram/bin
DATA_DIR    ?= $(HOME)/.engram
CLIENTS     ?= auto
PURGE       ?= 0
COMMON      := scripts/engram-common.sh

# Resolve cargo's output directory authoritatively (honors CARGO_TARGET_DIR and
# .cargo/config's build.target-dir) rather than assuming ./target; fall back to
# CARGO_TARGET_DIR or ./target if `cargo metadata`/python3 are unavailable.
# Deferred (=) so only targets that actually need the binary pay the cost.
TARGET_DIR   = $(shell cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin)["target_directory"])' 2>/dev/null || echo "$${CARGO_TARGET_DIR:-target}")
TARGET_BIN   = $(TARGET_DIR)/release/engram

# Run targets serially so `make` (= all) executes build->install->init->configure
# in order even under `make -j`. cargo handles its own internal parallelism.
.NOTPARALLEL:
.DEFAULT_GOAL := all
.PHONY: all build install init configure uninstall clean help

## all: build, install, init the DB, and configure MCP clients (default)
all: build install init configure
	@printf '\033[1;32m✓\033[0m %s\n' "engram installed at $(INSTALL_DIR)/engram — restart your MCP client."

## build: compile the release binary
build:
	cargo build --release --locked

## install: copy the freshly built binary into INSTALL_DIR
install: build
	@ENGRAM_INSTALL_DIR="$(INSTALL_DIR)" sh -c '. $(COMMON); install_binary "$(TARGET_BIN)"; ok "Installed $$(abs_bin)"; ensure_path'

## init: initialize the engram database (engram init)
init:
	@ENGRAM_INSTALL_DIR="$(INSTALL_DIR)" sh -c '. $(COMMON); run_init'

## configure: register engram with MCP clients (idempotent; honors CLIENTS)
configure:
	@ENGRAM_INSTALL_DIR="$(INSTALL_DIR)" ENGRAM_CLIENTS="$(CLIENTS)" sh -c '. $(COMMON); configure_clients'

## uninstall: remove the binary and engram entries from MCP configs (PURGE=1 also drops DATA_DIR)
uninstall:
	@rm -f "$(INSTALL_DIR)/engram"
	@printf '\033[1;32m✓\033[0m %s\n' "Removed $(INSTALL_DIR)/engram"
	@ENGRAM_INSTALL_DIR="$(INSTALL_DIR)" sh -c '. $(COMMON); unconfigure_clients'
	@if [ "$(PURGE)" = "1" ]; then \
	  rm -rf "$(DATA_DIR)"; \
	  printf '\033[1;33m!!\033[0m %s\n' "Purged database at $(DATA_DIR)"; \
	fi

## clean: remove cargo build artifacts
clean:
	cargo clean

## help: list available targets
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed -e 's/^## /  /'
