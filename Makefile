SHELL := bash

# ─── Targets ─────────────────────────────────────────────────────────────────
ANDROID_TARGET := aarch64-linux-android
IOS_TARGET     := aarch64-apple-ios
FFI            := --manifest-path crates/adapters/el-ffi/Cargo.toml
OUT            := out
FRB_VERSION    := 2.12.0
FRB_RUST_OUTPUT := crates/adapters/el-ffi/src/frb_generated.rs

# FRB 2.12 canonicalizes the crate root with the Windows extended-path prefix
# but does not do the same for a relative rust-output path.
ifeq ($(OS),Windows_NT)
FRB_RUST_OUTPUT := $(shell python -c "from pathlib import Path; print('\\\\?\\' + str(Path('crates/adapters/el-ffi/src/frb_generated.rs').resolve()))")
endif

ifneq ($(strip $(CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER)),)
ANDROID_TOOLCHAIN_BIN := $(dir $(CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER))
CC_aarch64_linux_android ?= $(CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER)
AR_aarch64_linux_android ?= $(ANDROID_TOOLCHAIN_BIN)llvm-ar
RANLIB_aarch64_linux_android ?= $(ANDROID_TOOLCHAIN_BIN)llvm-ranlib
export CC_aarch64_linux_android
export AR_aarch64_linux_android
export RANLIB_aarch64_linux_android
endif

.PHONY: check build-android build-ios build-ios-xcframework build-wasm codegen-rn codegen-dart codegen-flutter codegen-web bindings

# ─── Workspace ───────────────────────────────────────────────────────────────

check:
	cargo test
	cargo fmt --check
	cargo clippy --all-targets -- -D warnings

# ─── Cross-compile ───────────────────────────────────────────────────────────
#
# Prerequisites
#   Android:  rustup target add aarch64-linux-android
#             set CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER to the NDK clang path
#             (Make exports CC/AR/RANLIB for C build scripts)
#             (see .cargo/config.toml for the exact variable name)
#   iOS:      rustup target add aarch64-apple-ios  (macOS + Xcode required)
#   wasm:     cargo install wasm-pack

## Cross-compile el-ffi as a shared library for Android (aarch64; linker selects the API).
build-android:
	cargo build $(FFI) --target $(ANDROID_TARGET) --release

## Cross-compile el-ffi as a static library for iOS (aarch64).
build-ios:
	cargo build $(FFI) --target $(IOS_TARGET) --release

## Build a dynamic iOS XCFramework for Flutter device and simulator hosts.
build-ios-xcframework:
	bash scripts/build-ios-xcframework.sh

## Build el-ffi as a WASM + wasm-bindgen ESM package.
build-wasm:
	wasm-pack build crates/adapters/el-ffi \
		--target web \
		--out-dir ../../../$(OUT)/web

# ─── Binding codegen ─────────────────────────────────────────────────────────
#
# Prerequisites (install once)
#   RN:      npm install --global uniffi-bindgen-react-native@0.31.0-3
#   Dart:    cargo install flutter_rust_bridge_codegen --version $(FRB_VERSION) --locked
#   Web:     (wasm-pack, covered by build-wasm)

## Generate React Native JSI bindings (TypeScript + C++).
## Requires: build-android
codegen-rn: build-android
	@mkdir -p $(OUT)/rn $(OUT)/rn/cpp
	uniffi-bindgen-react-native generate jsi bindings \
		--library \
		--crate el-ffi \
		--ts-dir $(OUT)/rn \
		--cpp-dir $(OUT)/rn/cpp \
		--no-format \
		target/$(ANDROID_TARGET)/release/libel_ffi.so

## Generate Dart bindings via flutter_rust_bridge v2 codegen.
codegen-dart:
	@mkdir -p $(OUT)/dart/lib/src
	@rm -rf $(OUT)/dart/android $(OUT)/dart/ios $(OUT)/dart/example
	@cp packaging/dart/pubspec.yaml $(OUT)/dart/pubspec.yaml
	@cp LICENSE $(OUT)/dart/LICENSE
	@cp packaging/dart/README.md $(OUT)/dart/README.md
	@cp packaging/dart/lib/edge_intelligence.dart $(OUT)/dart/lib/edge_intelligence.dart
	@cp -R packaging/dart/android $(OUT)/dart/android
	@cp -R packaging/dart/ios $(OUT)/dart/ios
	@cp -R packaging/dart/example $(OUT)/dart/example
	@cp -R packaging/dart/test $(OUT)/dart/test
	@printf '%s\n' \
		'# Changelog' \
		'' \
		'## 0.1.0' \
		'' \
		'Release notes are tracked in the Edge Intelligence repository tags and GitHub releases.' \
		> $(OUT)/dart/CHANGELOG.md
	flutter_rust_bridge_codegen generate \
		--rust-root crates/adapters/el-ffi \
		--rust-input crate::dart_api \
		--rust-output "$(FRB_RUST_OUTPUT)" \
		--dart-root $(OUT)/dart \
		--dart-output $(OUT)/dart/lib/src \
		--no-add-mod-to-lib \
		--no-auto-upgrade-dependency \
		--no-deps-check \
		--no-dart-format
	@cp packaging/dart/lib/src/runtime_loader*.dart $(OUT)/dart/lib/src/
	@dart format $(OUT)/dart/lib $(OUT)/dart/example/lib $(OUT)/dart/example/dart_cli.dart >/dev/null

## Compatibility alias for older automation; prefer codegen-dart.
codegen-flutter: codegen-dart
	@rm -rf $(OUT)/flutter
	@cp -R $(OUT)/dart $(OUT)/flutter

## Build WASM output (identical to build-wasm; alias for consistency).
codegen-web: build-wasm

## Run all three codegen surfaces.
bindings: codegen-rn codegen-dart codegen-web
