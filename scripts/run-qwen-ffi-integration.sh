#!/bin/sh
set -eu

readonly test_name="native_qwen_integration_decodes_and_streams_english_text"
readonly output_file="$(mktemp)"
trap 'rm -f "$output_file"' EXIT

# Do not use Cargo's module-qualified --exact filter here: moving this test
# into an integration target would otherwise turn the command into a green
# no-op. The assertions below require Cargo to run exactly the one named test.
cargo test -p el-ffi "$test_name" -- --ignored 2>&1 | tee "$output_file"

grep -F "running 1 test" "$output_file"
grep -F "$test_name ... ok" "$output_file"
grep -F "test result: ok. 1 passed;" "$output_file"
