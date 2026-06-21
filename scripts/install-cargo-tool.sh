#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <crate> <binary> [version]" >&2
  exit 64
fi

crate="$1"
binary="$2"
version="${3:-}"
attempts="${CARGO_INSTALL_ATTEMPTS:-3}"
retry_seconds="${CARGO_INSTALL_RETRY_SECONDS:-15}"

installed_version_matches() {
  command -v "$binary" >/dev/null 2>&1 || return 1

  if [ -z "$version" ]; then
    return 0
  fi

  "$binary" --version 2>/dev/null | grep -F -- "$version" >/dev/null
}

if installed_version_matches; then
  "$binary" --version
  exit 0
fi

for attempt in $(seq 1 "$attempts"); do
  echo "Installing ${crate}${version:+ ${version}} (attempt ${attempt}/${attempts})..."

  args=(install "$crate" --locked --force)
  if [ -n "$version" ]; then
    args=(install "$crate" --version "$version" --locked --force)
  fi

  if cargo "${args[@]}"; then
    "$binary" --version
    exit 0
  fi

  if [ "$attempt" -lt "$attempts" ]; then
    echo "Retrying ${crate} install in ${retry_seconds}s..."
    sleep "$retry_seconds"
  fi
done

echo "ERROR: failed to install ${crate}${version:+ ${version}} after ${attempts} attempts" >&2
exit 1
