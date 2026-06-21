#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin"

cat > "$TMP/bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [ "$1" != "install" ]; then
  echo "unexpected cargo command: $*" >&2
  exit 64
fi

attempts_file="$FAKE_STATE/attempts"
attempts=0
[ -f "$attempts_file" ] && attempts="$(cat "$attempts_file")"
attempts=$((attempts + 1))
echo "$attempts" > "$attempts_file"
printf '%s\n' "$*" >> "$FAKE_STATE/cargo-args"

if [ "$attempts" -eq 1 ]; then
  echo "transient registry failure" >&2
  exit 101
fi

cat > "$FAKE_BIN/cargo-expand" <<'BIN'
#!/usr/bin/env bash
printf 'cargo-expand 1.2.3\n'
BIN
chmod +x "$FAKE_BIN/cargo-expand"
SH
chmod +x "$TMP/bin/cargo"

OUTPUT="$TMP/output"
PATH="$TMP/bin:$PATH" \
FAKE_BIN="$TMP/bin" \
FAKE_STATE="$TMP" \
CARGO_INSTALL_ATTEMPTS=2 \
CARGO_INSTALL_RETRY_SECONDS=0 \
bash "$ROOT/scripts/install-cargo-tool.sh" cargo-expand cargo-expand 1.2.3 > "$OUTPUT"

test "$(cat "$TMP/attempts")" = "2"
grep -q -- "install cargo-expand --version 1.2.3 --locked --force" "$TMP/cargo-args"
grep -q -- "cargo-expand 1.2.3" "$OUTPUT"
