#!/bin/sh
# Download the ADR-026 integration fixture from immutable revisions and verify
# its manifest before it is handed to a native or mobile test.
set -eu

readonly script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
readonly fixture_dir=${1:-qwen-fixture}
readonly model_url='https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/9217f5db79a29953eb74d5343926648285ec7e67/qwen2.5-0.5b-instruct-q4_k_m.gguf'
readonly tokenizer_url='https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer.json'

verify_fixture() {
  if command -v sha256sum >/dev/null 2>&1; then
    (
      cd "$fixture_dir"
      sha256sum -c "$script_dir/qwen-fixture.sha256"
    )
    return
  fi

  (
    cd "$fixture_dir"
    while read -r expected filename; do
      actual=$(shasum -a 256 "$filename" | awk '{print $1}')
      test "$actual" = "$expected"
    done < "$script_dir/qwen-fixture.sha256"
  )
}

mkdir -p "$fixture_dir"
if [ -f "$fixture_dir/qwen.gguf" ] \
  && [ -f "$fixture_dir/tokenizer.json" ] \
  && verify_fixture; then
  echo "Using verified Qwen integration fixture in $fixture_dir"
  exit 0
fi

echo "Downloading pinned Qwen integration fixture into $fixture_dir"
bash "$script_dir/retry-command.sh" curl --fail --location \
  --output "$fixture_dir/qwen.gguf" "$model_url"
bash "$script_dir/retry-command.sh" curl --fail --location \
  --output "$fixture_dir/tokenizer.json" "$tokenizer_url"
verify_fixture
