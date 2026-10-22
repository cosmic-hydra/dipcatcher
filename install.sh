#!/bin/sh
# fxi installer — interactive CLI for fx-1.
#
#   curl -fsSL https://raw.githubusercontent.com/cosmic-hydra/dipcatcher/main/install.sh | sh
#
# What it does, in order:
#   1. finds a Python >= 3.12 (FXI_PYTHON to override)
#   2. creates (or reuses) a venv at ${FXI_HOME:-~/.local/share/fxi}
#   3. pip-installs the fx-1 wheel — from $FXI_INSTALL_URL, --wheel, or the
#      default GitHub release asset for $FXI_VERSION
#   4. symlinks fxi/fx1/dipcatcher/quant/verify-ledger/mc-engine into
#      ${FXI_BIN_DIR:-~/.local/bin}
#   5. smoke-runs `fxi --version`
#
# It never uses sudo and never writes outside FXI_HOME, FXI_BIN_DIR, and the
# pip cache. `--uninstall` removes what it installed.
#
# Optional integrity pin: set FXI_WHEEL_SHA256=<hex> to require the wheel's
# sha256 to match before it is installed (works for --wheel and remote URLs).
set -eu

FXI_VERSION="${FXI_VERSION:-0.4.0}"
FXI_HOME="${FXI_HOME:-$HOME/.local/share/fxi}"
FXI_BIN_DIR="${FXI_BIN_DIR:-$HOME/.local/bin}"
DEFAULT_WHEEL_URL="https://github.com/cosmic-hydra/dipcatcher/releases/download/v${FXI_VERSION}/fx_1-${FXI_VERSION}-py3-none-any.whl"
ENTRY_POINTS="fxi fx1 dipcatcher quant verify-ledger mc-engine"

log()  { printf '\033[1;34mfxi>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mfxi warn>\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mfxi error>\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    # Self-contained on purpose: under `curl | sh`, "$0" is the shell, not
    # this script, so the help text cannot be read back from "$0".
    cat <<'EOF'
fxi installer — interactive CLI for fx-1.

  curl -fsSL https://raw.githubusercontent.com/cosmic-hydra/dipcatcher/main/install.sh | sh

What it does, in order:
  1. finds a Python >= 3.12 (FXI_PYTHON to override)
  2. creates (or reuses) a venv at ${FXI_HOME:-~/.local/share/fxi}
  3. pip-installs the fx-1 wheel — from $FXI_INSTALL_URL, --wheel, or the
     default GitHub release asset for $FXI_VERSION
  4. symlinks fxi/fx1/dipcatcher/quant/verify-ledger/mc-engine into
     ${FXI_BIN_DIR:-~/.local/bin}
  5. smoke-runs `fxi --version`

It never uses sudo and never writes outside FXI_HOME, FXI_BIN_DIR, and the
pip cache. `--uninstall` removes what it installed.

Optional integrity pin: set FXI_WHEEL_SHA256=<hex> to require the wheel's
sha256 to match before it is installed (works for --wheel and remote URLs).

Usage: sh install.sh [--wheel PATH] [--version X.Y.Z] [--home DIR]
                     [--bin-dir DIR] [--uninstall]
EOF
    exit "${1:-0}"
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        die "no sha256 tool (sha256sum/shasum/openssl) to verify FXI_WHEEL_SHA256"
    fi
}

# Refuse destructive or surprising destinations before any rm -rf / mkdir.
safe_home() {
    case "$1" in
        ""|"/"|"$HOME") die "FXI_HOME='$1' is not a safe directory (refusing)" ;;
    esac
}

WHEEL=""
UNINSTALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --wheel)    WHEEL="${2:?--wheel needs a path}"; shift 2 ;;
        --version)  FXI_VERSION="${2:?--version needs a value}"; shift 2 ;;
        --home)     FXI_HOME="${2:?--home needs a path}"; shift 2 ;;
        --bin-dir)  FXI_BIN_DIR="${2:?--bin-dir needs a path}"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help)  usage 0 ;;
        *) die "unknown argument $1 (try --help)" ;;
    esac
done

find_python() {
    if [ -n "${FXI_PYTHON:-}" ]; then
        command -v "$FXI_PYTHON" >/dev/null 2>&1 || die "FXI_PYTHON=$FXI_PYTHON not found"
        printf '%s' "$FXI_PYTHON"
        return
    fi
    for cand in python3.13 python3.12 python3 python; do
        if command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
                printf '%s' "$cand"
                return
            fi
        fi
    done
    die "no Python >= 3.12 found. Install python.org 3.12+, or set FXI_PYTHON=/path/to/python"
}

if [ "$UNINSTALL" = 1 ]; then
    safe_home "$FXI_HOME"
    for name in $ENTRY_POINTS; do
        link="$FXI_BIN_DIR/$name"
        if [ -L "$link" ]; then
            case "$(readlink "$link" 2>/dev/null || true)" in
                "$FXI_HOME"/*) rm -f "$link"; log "removed $link" ;;
            esac
        fi
    done
    rm -rf "$FXI_HOME"
    log "removed $FXI_HOME"
    exit 0
fi

PY="$(find_python)"
log "using $($PY --version 2>&1) at $PY"

safe_home "$FXI_HOME"
if ! mkdir -p "$FXI_HOME" 2>/dev/null; then
    die "cannot write FXI_HOME=$FXI_HOME (no sudo by design; pick another with --home)"
fi

WHEEL_SOURCE="${WHEEL:-${FXI_INSTALL_URL:-$DEFAULT_WHEEL_URL}}"
case "$WHEEL_SOURCE" in
    http://*|https://*) log "installing fx-1 $FXI_VERSION from $WHEEL_SOURCE" ;;
    *) [ -f "$WHEEL_SOURCE" ] || die "wheel not found: $WHEEL_SOURCE" ;
       log "installing fx-1 from local wheel $WHEEL_SOURCE" ;;
esac

INSTALL_FILE="$WHEEL_SOURCE"
if [ -n "${FXI_WHEEL_SHA256:-}" ]; then
    case "$WHEEL_SOURCE" in
        http://*|https://*)
            INSTALL_FILE="$FXI_HOME/fx_1-${FXI_VERSION}-py3-none-any.whl"
            if command -v curl >/dev/null 2>&1; then
                curl -fsSL "$WHEEL_SOURCE" -o "$INSTALL_FILE"
            elif command -v wget >/dev/null 2>&1; then
                wget -qO "$INSTALL_FILE" "$WHEEL_SOURCE"
            else
                die "FXI_WHEEL_SHA256 verification needs curl or wget to stage the wheel"
            fi ;;
    esac
    actual="$(sha256_of "$INSTALL_FILE")"
    [ "$actual" = "$FXI_WHEEL_SHA256" ] \
        || die "sha256 mismatch for $INSTALL_FILE (expected $FXI_WHEEL_SHA256, got $actual)"
    log "sha256 verified: $actual"
else
    case "$WHEEL_SOURCE" in
        http://*|https://*)
            warn "unpinned download; set FXI_WHEEL_SHA256=<hex> to verify the wheel" ;;
    esac
fi

log "venv: $FXI_HOME/venv"
"$PY" -m venv "$FXI_HOME/venv"
"$FXI_HOME/venv/bin/python" -m pip install --quiet --upgrade pip
"$FXI_HOME/venv/bin/python" -m pip install --quiet "$INSTALL_FILE"

if ! mkdir -p "$FXI_BIN_DIR" 2>/dev/null; then
    die "cannot write FXI_BIN_DIR=$FXI_BIN_DIR (pick another with --bin-dir)"
fi
for name in $ENTRY_POINTS; do
    if [ -x "$FXI_HOME/venv/bin/$name" ]; then
        ln -sf "$FXI_HOME/venv/bin/$name" "$FXI_BIN_DIR/$name"
    else
        warn "$name not present in this wheel; skipping"
    fi
done

if ! "$FXI_BIN_DIR/fxi" --version >/dev/null 2>&1; then
    die "smoke test failed: $FXI_BIN_DIR/fxi --version did not run"
fi
log "installed: $("$FXI_BIN_DIR/fxi" --version)"

case ":$PATH:" in
    *":$FXI_BIN_DIR:"*) ;;
    *) warn "$FXI_BIN_DIR is not on PATH; add 'export PATH=\"$FXI_BIN_DIR:\$PATH\"' to your shell rc" ;;
esac

log "next: fxi keys set fx1   # store your fx-1 API key (prompted, not echoed)"
log "      fxi                # open the interactive shell"
