#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 64
fi

attempts="${COMMAND_ATTEMPTS:-3}"
retry_seconds="${COMMAND_RETRY_SECONDS:-15}"
status=0

for attempt in $(seq 1 "$attempts"); do
  echo "Running command (attempt ${attempt}/${attempts}): $*"

  set +e
  "$@"
  status="$?"
  set -e

  if [ "$status" -eq 0 ]; then
    exit 0
  fi

  if [ "$attempt" -lt "$attempts" ]; then
    echo "Command failed with status ${status}; retrying in ${retry_seconds}s..."
    sleep "$retry_seconds"
  fi
done

echo "ERROR: command failed after ${attempts} attempts: $*" >&2
exit "$status"
