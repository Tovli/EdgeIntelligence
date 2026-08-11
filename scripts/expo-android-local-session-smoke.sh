#!/bin/sh
set -eu

readonly apk_path="expo-smoke/android/app/build/outputs/apk/release/app-release.apk"
readonly package_name="com.tovli.edgeintelligence.example"
readonly model_path="${EDGE_INTELLIGENCE_QWEN_GGUF:?set EDGE_INTELLIGENCE_QWEN_GGUF to the Qwen GGUF}"
readonly tokenizer_path="${EDGE_INTELLIGENCE_QWEN_TOKENIZER:?set EDGE_INTELLIGENCE_QWEN_TOKENIZER to tokenizer.json}"
readonly app_assets_dir="/sdcard/Android/data/$package_name/files/models"
readonly app_model_path="$app_assets_dir/qwen.gguf"
readonly app_tokenizer_path="$app_assets_dir/tokenizer.json"
readonly success_text="Edge Intelligence Qwen local session passed."
readonly failure_text="Edge Intelligence Qwen local session failed:"

test -s "$model_path"
test -s "$tokenizer_path"

adb install -r "$apk_path"
# `adb push` runs as the shell user, while Android grants this app access to
# its own scoped external-files directory without a storage permission. Keeping
# the release APK preserves its embedded JavaScript bundle for an offline smoke.
adb shell mkdir -p "$app_assets_dir"
adb push "$model_path" "$app_model_path"
adb push "$tokenizer_path" "$app_tokenizer_path"
adb shell test -s "$app_model_path"
adb shell test -s "$app_tokenizer_path"
adb logcat -c
adb shell monkey -p "$package_name" 1 || true

dump_ui() {
  adb shell uiautomator dump /sdcard/window.xml >/dev/null 2>&1 || true
  adb shell cat /sdcard/window.xml 2>/dev/null || true
}

dump_failure_logs() {
  echo "Local-session logcat follows." >&2
  adb logcat -d -v threadtime >&2 || true
}

for _ in $(seq 1 180); do
  ui_dump=$(dump_ui)
  if printf '%s\n' "$ui_dump" | grep -qF "$success_text"; then
    exit 0
  fi
  if printf '%s\n' "$ui_dump" | grep -qF "$failure_text"; then
    echo "Local-session failure UI follows." >&2
    printf '%s\n' "$ui_dump" >&2
    dump_failure_logs
    exit 1
  fi
  sleep 2
done

echo "Local-session success UI was not found." >&2
dump_ui >&2
dump_failure_logs
exit 1
