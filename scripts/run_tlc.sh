#!/usr/bin/env bash
# Model-check spec/tla/OrderLifecycle.tla with TLC.
# Safety (crashes, amends, duplicate and late fills) and liveness
# (no crash, no amend, weak fairness) are separate configs.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Stable v1.7.4 (TLC 2.19), not the rolling v1.8.0 prerelease.
# Asset IDs do not follow replacement uploads. Deletion must fail closed.
# See spec/tla/README.md for upstream provenance and upgrade procedure.
JAR="${TLA_TOOLS_JAR:-$ROOT/.tla/tla2tools-1.7.4.jar}"
URL="https://api.github.com/repos/tlaplus/tlaplus/releases/assets/184694200"
SHA="936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
SIZE="2274532"

DOWNLOAD=""
META=""
HEADERS=""
cleanup() {
  [[ -z "$DOWNLOAD" ]] || rm -f -- "$DOWNLOAD"
  [[ -z "$META" ]] || rm -rf -- "$META"
  [[ -z "$HEADERS" ]] || rm -f -- "$HEADERS"
  return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

verify_jar() {
  local file="$1" actual_sha actual_size
  actual_sha="$(sha256sum < "$file")"
  actual_sha="${actual_sha%% *}"
  actual_size="$(wc -c < "$file" | tr -d '[:space:]')"
  if [[ "$actual_sha" != "$SHA" || "$actual_size" != "$SIZE" ]]; then
    printf 'TLC artifact verification failed: %s\n' "$file" >&2
    printf 'Expected SHA-256: %s; bytes: %s\n' "$SHA" "$SIZE" >&2
    printf 'Actual   SHA-256: %s; bytes: %s\n' "$actual_sha" "$actual_size" >&2
    return 1
  fi
}

if [[ ! -f "$JAR" ]]; then
  mkdir -p "$(dirname "$JAR")"
  DOWNLOAD="$(mktemp "${JAR}.download.XXXXXX")"
  # Anonymous api.github.com calls share 60/hour per IP; busy CI runners hit
  # 403. With GITHUB_TOKEN set, authenticate from a 0600 header file so the
  # token stays out of argv. curl drops it on the cross-host redirect.
  auth=()
  if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    HEADERS="$(mktemp)"
    printf 'Authorization: Bearer %s\n' "$GITHUB_TOKEN" > "$HEADERS"
    auth=(-H "@$HEADERS")
  fi
  curl --fail --silent --show-error --location \
    --proto '=https' --proto-redir '=https' \
    --connect-timeout 30 --max-time 180 \
    -H 'Accept: application/octet-stream' \
    ${auth[@]+"${auth[@]}"} \
    -o "$DOWNLOAD" "$URL"
  if [[ -n "$HEADERS" ]]; then
    rm -f -- "$HEADERS"
    HEADERS=""
  fi
  verify_jar "$DOWNLOAD"
  mv -T -- "$DOWNLOAD" "$JAR"
  DOWNLOAD=""
fi
# Overrides and cached files must satisfy exactly the same integrity checks.
verify_jar "$JAR"

META="$(mktemp -d)"

run_tlc() {
  local cfg="$1"
  local dir="$2"
  mkdir -p "$dir"
  java -Xmx1g -cp "$JAR" tlc2.TLC \
    -workers 2 \
    -cleanup \
    -metadir "$dir" \
    -config "$cfg" \
    "$ROOT/spec/tla/OrderLifecycle.tla"
}

echo "== TLC safety =="
run_tlc "$ROOT/spec/tla/OrderLifecycle.cfg" "$META/safety"
echo "== TLC liveness =="
run_tlc "$ROOT/spec/tla/OrderLifecycleLiveness.cfg" "$META/liveness"
