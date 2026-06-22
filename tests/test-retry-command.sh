#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin"

cat > "$TMP/bin/flaky-command" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

attempts_file="$FAKE_STATE/attempts"
attempts=0
[ -f "$attempts_file" ] && attempts="$(cat "$attempts_file")"
attempts=$((attempts + 1))
echo "$attempts" > "$attempts_file"
printf '%s\n' "$*" >> "$FAKE_STATE/args"

if [ "$attempts" -eq 1 ]; then
  echo "transient network failure" >&2
  exit 56
fi

printf 'ok\n'
SH
chmod +x "$TMP/bin/flaky-command"

OUTPUT="$TMP/output"
PATH="$TMP/bin:$PATH" \
FAKE_STATE="$TMP" \
COMMAND_ATTEMPTS=2 \
COMMAND_RETRY_SECONDS=0 \
bash "$ROOT/scripts/retry-command.sh" flaky-command alpha beta > "$OUTPUT"

test "$(cat "$TMP/attempts")" = "2"
grep -q -- "alpha beta" "$TMP/args"
grep -q -- "ok" "$OUTPUT"
