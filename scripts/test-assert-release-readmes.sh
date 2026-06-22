#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$TMP/bin" "$TMP/npm/src" "$TMP/pub"

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
  crate-a|crate-b)
    printf '%s\n' Cargo.toml README.md src/lib.rs
    ;;
  *)
    echo "unexpected crate: $crate" >&2
    exit 65
    ;;
esac
SH
chmod +x "$TMP/bin/cargo"

cat > "$TMP/bin/npm" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" != "pack" ]; then
  echo "unexpected npm command: $*" >&2
  exit 66
fi

cat <<'JSON'
[
  {
    "files": [
      { "path": "README.md" },
      { "path": "package.json" }
    ]
  }
]
JSON
SH
chmod +x "$TMP/bin/npm"

printf '%s\n' '{"name":"readme-check","version":"1.0.0","files":["src/","README.md"]}' > "$TMP/npm/package.json"
printf '%s\n' '# readme-check' > "$TMP/npm/README.md"
printf '%s\n' 'console.log(1);' > "$TMP/npm/src/index.js"
printf '%s\n' '# pub-check' > "$TMP/pub/README.md"

PATH="$TMP/bin:$PATH" \
PUBLISH_CRATES="crate-a crate-b" \
bash "$ROOT/scripts/assert-release-readmes.sh" cargo

PATH="$TMP/bin:$PATH" bash "$ROOT/scripts/assert-release-readmes.sh" npm "$TMP/npm"
bash "$ROOT/scripts/assert-release-readmes.sh" pub "$TMP/pub"
