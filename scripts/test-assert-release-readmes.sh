#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p \
  "$TMP/bin" \
  "$TMP/crate-a/src" \
  "$TMP/crate-b/src" \
  "$TMP/crate-empty/src" \
  "$TMP/crate-no-readme-list/src" \
  "$TMP/npm/src" \
  "$TMP/npm-no-readme/src" \
  "$TMP/pub"

cat > "$TMP/bin/cargo" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "metadata" ]; then
  cat "${FAKE_METADATA:?}"
  exit 0
fi

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
  crate-empty)
    printf '%s\n' Cargo.toml README.md src/lib.rs
    ;;
  crate-no-readme-list)
    printf '%s\n' Cargo.toml src/lib.rs
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

if [ "${NPM_PACK_WITHOUT_README:-}" = "1" ]; then
  cat <<'JSON'
[
  {
    "files": [
      { "path": "package.json" }
    ]
  }
]
JSON
  exit 0
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

cat > "$TMP/metadata.json" <<JSON
{
  "packages": [
    { "name": "crate-a", "manifest_path": "$TMP/crate-a/Cargo.toml" },
    { "name": "crate-b", "manifest_path": "$TMP/crate-b/Cargo.toml" },
    { "name": "crate-empty", "manifest_path": "$TMP/crate-empty/Cargo.toml" },
    { "name": "crate-no-readme-list", "manifest_path": "$TMP/crate-no-readme-list/Cargo.toml" }
  ]
}
JSON

for crate in crate-a crate-b crate-empty crate-no-readme-list; do
  printf '%s\n' '[package]' "name = \"$crate\"" > "$TMP/$crate/Cargo.toml"
  printf '%s\n' 'pub fn placeholder() {}' > "$TMP/$crate/src/lib.rs"
done
printf '%s\n' '# crate-a' > "$TMP/crate-a/README.md"
printf '%s\n' '# crate-b' > "$TMP/crate-b/README.md"
: > "$TMP/crate-empty/README.md"
printf '%s\n' '# crate-no-readme-list' > "$TMP/crate-no-readme-list/README.md"

printf '%s\n' '{"name":"readme-check","version":"1.0.0","files":["src/","README.md"]}' > "$TMP/npm/package.json"
printf '%s\n' '# readme-check' > "$TMP/npm/README.md"
printf '%s\n' 'console.log(1);' > "$TMP/npm/src/index.js"
printf '%s\n' '{"name":"readme-check","version":"1.0.0","files":["src/"]}' > "$TMP/npm-no-readme/package.json"
printf '%s\n' '# readme-check' > "$TMP/npm-no-readme/README.md"
printf '%s\n' 'console.log(1);' > "$TMP/npm-no-readme/src/index.js"
printf '%s\n' '# pub-check' > "$TMP/pub/README.md"

PATH="$TMP/bin:$PATH" \
FAKE_METADATA="$TMP/metadata.json" \
PUBLISH_CRATES="crate-a crate-b" \
bash "$ROOT/scripts/assert-release-readmes.sh" cargo

if PATH="$TMP/bin:$PATH" \
  FAKE_METADATA="$TMP/metadata.json" \
  PUBLISH_CRATES="crate-empty" \
  bash "$ROOT/scripts/assert-release-readmes.sh" cargo 2>/dev/null; then
  echo "empty crate README should fail" >&2
  exit 1
fi

if PATH="$TMP/bin:$PATH" \
  FAKE_METADATA="$TMP/metadata.json" \
  PUBLISH_CRATES="crate-no-readme-list" \
  bash "$ROOT/scripts/assert-release-readmes.sh" cargo 2>/dev/null; then
  echo "crate package missing README.md should fail" >&2
  exit 1
fi

PATH="$TMP/bin:$PATH" bash "$ROOT/scripts/assert-release-readmes.sh" npm "$TMP/npm"
if NPM_PACK_WITHOUT_README=1 \
  PATH="$TMP/bin:$PATH" \
  bash "$ROOT/scripts/assert-release-readmes.sh" npm "$TMP/npm-no-readme" 2>/dev/null; then
  echo "npm package excluding README.md should fail" >&2
  exit 1
fi

bash "$ROOT/scripts/assert-release-readmes.sh" pub "$TMP/pub"
