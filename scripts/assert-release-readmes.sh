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
  local package_list
  local metadata
  package_list="$(mktemp)"
  metadata="$(mktemp)"
  trap 'rm -f "$package_list" "$metadata"' RETURN

  cd "$ROOT"
  cargo metadata --no-deps --format-version 1 > "$metadata"
  for crate in $crates; do
    if ! CRATE="$crate" METADATA="$metadata" python3 <<'PYEOF'
import json
import os
from pathlib import Path

crate = os.environ["CRATE"]
metadata = Path(os.environ["METADATA"])
packages = json.loads(metadata.read_text(encoding="utf-8"))["packages"]

for package in packages:
    if package["name"] == crate:
        readme = Path(package["manifest_path"]).parent / "README.md"
        raise SystemExit(0 if readme.is_file() and readme.stat().st_size > 0 else 1)

raise SystemExit(1)
PYEOF
    then
      fail "crates.io package ${crate} source README.md is missing or empty"
    fi

    cargo package --list --allow-dirty -p "$crate" > "$package_list"
    grep -Fxq "README.md" "$package_list" || fail "crates.io package ${crate} does not include README.md"
    echo "OK: crates.io package ${crate} includes README.md"
  done
}

assert_npm_readme() {
  local dir="${1:?npm package directory is required}"
  local pack_json
  assert_file_readme "$dir" "npm package at $dir"

  cd "$dir"
  pack_json="$(npm pack --dry-run --json)"
  grep -Eq '"path": ?"README\.md"' <<<"$pack_json" \
    || fail "npm package at $dir would not ship README.md"
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
