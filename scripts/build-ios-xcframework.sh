#!/usr/bin/env bash
set -euo pipefail

crate_manifest="crates/adapters/el-ffi/Cargo.toml"
output_root="target/ios-xcframework"
device_framework="$output_root/device/el_ffi.framework"
simulator_framework="$output_root/simulator/el_ffi.framework"

rm -rf "$output_root"

cargo build --manifest-path "$crate_manifest" --target aarch64-apple-ios --release
cargo build --manifest-path "$crate_manifest" --target aarch64-apple-ios-sim --release
cargo build --manifest-path "$crate_manifest" --target x86_64-apple-ios --release

create_framework() {
  local framework_path="$1"
  local library_path="$2"

  mkdir -p "$framework_path/Headers" "$framework_path/Modules"
  cp "$library_path" "$framework_path/el_ffi"
  chmod +x "$framework_path/el_ffi"
  install_name_tool -id '@rpath/el_ffi.framework/el_ffi' "$framework_path/el_ffi"

  cat > "$framework_path/Headers/el_ffi.h" <<'HEADER'
#ifndef EL_FFI_H
#define EL_FFI_H

// Dart FFI resolves the exported bridge symbols dynamically.

#endif
HEADER

  cat > "$framework_path/Modules/module.modulemap" <<'MODULEMAP'
framework module el_ffi {
  umbrella header "el_ffi.h"
  export *
}
MODULEMAP

  cat > "$framework_path/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>el_ffi</string>
  <key>CFBundleIdentifier</key>
  <string>com.tovli.edge-intelligence.el-ffi</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>el_ffi</string>
  <key>CFBundlePackageType</key>
  <string>FMWK</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>MinimumOSVersion</key>
  <string>13.0</string>
</dict>
</plist>
PLIST
}

create_framework \
  "$device_framework" \
  "target/aarch64-apple-ios/release/libel_ffi.dylib"

mkdir -p "$output_root/simulator"
lipo -create \
  target/aarch64-apple-ios-sim/release/libel_ffi.dylib \
  target/x86_64-apple-ios/release/libel_ffi.dylib \
  -output "$output_root/simulator/libel_ffi.dylib"
create_framework \
  "$simulator_framework" \
  "$output_root/simulator/libel_ffi.dylib"

xcodebuild -create-xcframework \
  -framework "$device_framework" \
  -framework "$simulator_framework" \
  -output "$output_root/el_ffi.xcframework"
