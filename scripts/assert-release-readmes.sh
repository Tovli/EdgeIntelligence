#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_PUBLISH_CRATES="el-core el-memory el-telemetry el-provenance el-safety el-runtime el-grammar el-provenance-ed25519 el-engine-candle el-cloud el-ffi"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

assert_file_readme() {
  local dir="$1"
  local label="$2"

  [ -s "$dir/README.md" ] || fail "${label} is missing a non-empty README.md"
}

assert_cargo_readmes() {
  local crates="${PUBLISH_CRATES:-$DEFAULT_PUBLISH_CRATES}"
  local tmp
  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' RETURN

  cd "$ROOT"
  for crate in $crates; do
    cargo package --list --allow-dirty -p "$crate" > "$tmp"
    grep -Fxq "README.md" "$tmp" || fail "crates.io package ${crate} does not include README.md"
    echo "OK: crates.io package ${crate} includes README.md"
  done
}

assert_npm_readme() {
  local dir="${1:?npm package directory is required}"
  local tmp
  assert_file_readme "$dir" "npm package at $dir"
  tmp="$(mktemp)"

  cd "$dir"
  npm pack --dry-run --json > "$tmp"
  grep -Eq '"path": ?"README\.md"' "$tmp" \
    || fail "npm package at $dir would not ship README.md"
  rm -f "$tmp"
  echo "OK: npm package at $dir ships README.md"
}

assert_pub_readme() {
  local dir="${1:?pub package directory is required}"
  assert_file_readme "$dir" "pub.dev package at $dir"
  echo "OK: pub.dev package at $dir includes README.md"
}

case "${1:-cargo}" in
  cargo)
    assert_cargo_readmes
    ;;
  npm)
    assert_npm_readme "${2:-}"
    ;;
  pub)
    assert_pub_readme "${2:-}"
    ;;
  *)
    fail "usage: $0 {cargo|npm <package-dir>|pub <package-dir>}"
    ;;
esac
