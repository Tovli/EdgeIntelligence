#!/bin/sh
set -eu

readonly apk_path="expo-smoke/android/app/build/outputs/apk/release/app-release.apk"
readonly package_name="com.tovli.edgeintelligence.example"
readonly success_text="Edge Intelligence native bridge loaded."

adb install -r "$apk_path"
adb logcat -c
adb shell monkey -p "$package_name" 1 || true

for _ in $(seq 1 60); do
  adb shell uiautomator dump /sdcard/window.xml >/dev/null 2>&1 || true
  if adb shell cat /sdcard/window.xml 2>/dev/null | grep -qF "$success_text"; then
    exit 0
  fi
  sleep 2
done

echo "Local-session success UI was not found; logcat follows." >&2
adb logcat -d
exit 1
