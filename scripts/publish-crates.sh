#!/usr/bin/env bash
set -euo pipefail

: "${VERSION:?VERSION is required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UA="${CRATES_IO_USER_AGENT:-edge-intelligence-release/ci}"
PUBLISH_CRATES="${PUBLISH_CRATES:-el-core el-memory el-telemetry el-provenance el-safety el-runtime el-grammar el-provenance-ed25519 el-engine-candle el-cloud el-ffi}"
PUBLISH_MAX_ATTEMPTS="${PUBLISH_MAX_ATTEMPTS:-3}"
RATE_LIMIT_SLEEP_SECONDS="${RATE_LIMIT_SLEEP_SECONDS:-600}"
INDEX_POLL_ATTEMPTS="${INDEX_POLL_ATTEMPTS:-24}"
INDEX_POLL_SECONDS="${INDEX_POLL_SECONDS:-15}"

PUBLISH_CRATES="$PUBLISH_CRATES" bash "$SCRIPT_DIR/assert-release-readmes.sh" cargo

crate_url() {
  printf 'https://crates.io/api/v1/crates/%s/%s' "$1" "$VERSION"
}

crate_version_exists() {
  local crate="$1"
  curl -sf -A "$UA" "$(crate_url "$crate")" | jq -e '.version.num' >/dev/null 2>&1
}

wait_indexed() {
  local crate="$1"

  echo "  Waiting for ${crate} ${VERSION} in crates.io index..."
  for i in $(seq 1 "$INDEX_POLL_ATTEMPTS"); do
    if crate_version_exists "$crate"; then
      echo "  OK: ${crate} ${VERSION} indexed"
      return 0
    fi

    echo "    not yet (${i}/${INDEX_POLL_ATTEMPTS}), retrying in ${INDEX_POLL_SECONDS}s..."
    sleep "$INDEX_POLL_SECONDS"
  done

  echo "ERROR: ${crate} ${VERSION} not indexed after polling"
  return 1
}

is_rate_limited() {
  grep -Eiq '429|Too Many Requests|rate limit' "$1"
}

publish_crate() {
  local crate="$1"

  if crate_version_exists "$crate"; then
    echo "  OK: ${crate} ${VERSION} already published; skipping"
    return 0
  fi

  local attempt=1
  while [ "$attempt" -le "$PUBLISH_MAX_ATTEMPTS" ]; do
    local log
    log="$(mktemp)"

    echo "Publishing ${crate} ${VERSION} (attempt ${attempt}/${PUBLISH_MAX_ATTEMPTS})..."
    set +e
    cargo publish -p "$crate" --no-verify 2>&1 | tee "$log"
    local status="${PIPESTATUS[0]}"
    set -e

    if [ "$status" -eq 0 ]; then
      rm -f "$log"
      wait_indexed "$crate"
      return 0
    fi

    if crate_version_exists "$crate"; then
      echo "  OK: ${crate} ${VERSION} appeared after publish error; skipping"
      rm -f "$log"
      return 0
    fi

    if is_rate_limited "$log"; then
      if [ "$attempt" -lt "$PUBLISH_MAX_ATTEMPTS" ]; then
        echo "Rate limited while publishing ${crate}; sleeping ${RATE_LIMIT_SLEEP_SECONDS}s before retry..."
        rm -f "$log"
        sleep "$RATE_LIMIT_SLEEP_SECONDS"
        attempt=$((attempt + 1))
        continue
      fi

      echo "ERROR: rate limited while publishing ${crate} after ${attempt} attempts"
    fi

    rm -f "$log"
    return "$status"
  done
}

for crate in $PUBLISH_CRATES; do
  publish_crate "$crate"
done
