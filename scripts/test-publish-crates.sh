#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

FAKE_STATE="$TMP/state"
mkdir -p "$TMP/bin" "$FAKE_STATE"

cat > "$TMP/bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

crate=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -p)
      shift
      crate="$1"
      ;;
  esac
  shift || true
done

case "$crate" in
  el-core)
    echo "el-core should have been skipped" >&2
    exit 64
    ;;
  el-ffi)
    attempts_file="$FAKE_STATE/el_ffi_attempts"
    attempts=0
    [ -f "$attempts_file" ] && attempts="$(cat "$attempts_file")"
    attempts=$((attempts + 1))
    echo "$attempts" > "$attempts_file"
    if [ "$attempts" -eq 1 ]; then
      echo "status 429 Too Many Requests" >&2
      exit 101
    fi
    touch "$FAKE_STATE/el_ffi_published"
    echo "Published el-ffi"
    ;;
  *)
    echo "unexpected crate: $crate" >&2
    exit 65
    ;;
esac
SH
chmod +x "$TMP/bin/cargo"

cat > "$TMP/bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

url="${*: -1}"
case "$url" in
  */crates/el-core/0.3.5)
    printf '{"version":{"num":"0.3.5"}}'
    ;;
  */crates/el-ffi/0.3.5)
    if [ -f "$FAKE_STATE/el_ffi_published" ]; then
      printf '{"version":{"num":"0.3.5"}}'
    else
      exit 22
    fi
    ;;
  *)
    echo "unexpected url: $url" >&2
    exit 66
    ;;
esac
SH
chmod +x "$TMP/bin/curl"

cat > "$TMP/bin/jq" <<'SH'
#!/usr/bin/env bash
cat >/dev/null
exit 0
SH
chmod +x "$TMP/bin/jq"

OUTPUT="$TMP/output"
PATH="$TMP/bin:$PATH" \
FAKE_STATE="$FAKE_STATE" \
VERSION="0.3.5" \
PUBLISH_CRATES="el-core el-ffi" \
PUBLISH_MAX_ATTEMPTS="2" \
RATE_LIMIT_SLEEP_SECONDS="0" \
INDEX_POLL_ATTEMPTS="2" \
INDEX_POLL_SECONDS="0" \
"$ROOT/scripts/publish-crates.sh" > "$OUTPUT"

grep -q "el-core 0.3.5 already published; skipping" "$OUTPUT"
grep -q "Rate limited while publishing el-ffi" "$OUTPUT"
grep -q "el-ffi 0.3.5 indexed" "$OUTPUT"
test "$(cat "$FAKE_STATE/el_ffi_attempts")" = "2"
