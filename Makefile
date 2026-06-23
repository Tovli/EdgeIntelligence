SHELL := bash

# ─── Targets ─────────────────────────────────────────────────────────────────
ANDROID_TARGET := aarch64-linux-android
IOS_TARGET     := aarch64-apple-ios
FFI            := --manifest-path crates/adapters/el-ffi/Cargo.toml
OUT            := out
FRB_VERSION    := 2.12.0

ifneq ($(strip $(CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER)),)
ANDROID_TOOLCHAIN_BIN := $(dir $(CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER))
CC_aarch64_linux_android ?= $(CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER)
AR_aarch64_linux_android ?= $(ANDROID_TOOLCHAIN_BIN)llvm-ar
RANLIB_aarch64_linux_android ?= $(ANDROID_TOOLCHAIN_BIN)llvm-ranlib
export CC_aarch64_linux_android
export AR_aarch64_linux_android
export RANLIB_aarch64_linux_android
endif

.PHONY: check build-android build-ios build-wasm codegen-rn codegen-flutter codegen-web bindings

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

## Cross-compile el-ffi as a shared library for Android (aarch64, API 35).
build-android:
	cargo build $(FFI) --target $(ANDROID_TARGET) --release

## Cross-compile el-ffi as a static library for iOS (aarch64).
build-ios:
	cargo build $(FFI) --target $(IOS_TARGET) --release

## Build el-ffi as a WASM + wasm-bindgen ESM package.
build-wasm:
	wasm-pack build crates/adapters/el-ffi \
		--target web \
		--out-dir ../../../$(OUT)/web

# ─── Binding codegen ─────────────────────────────────────────────────────────
#
# Prerequisites (install once)
#   RN:      npm install --global uniffi-bindgen-react-native@0.31.0-3
#   Flutter: cargo install flutter_rust_bridge_codegen --version $(FRB_VERSION) --locked
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

## Generate Flutter/Dart bindings via flutter_rust_bridge v2 codegen.
codegen-flutter:
	@mkdir -p $(OUT)/flutter/lib
	@printf '%s\n' \
		'name: edge_intelligence' \
		'version: 0.1.0' \
		'description: Generated Flutter bindings for Edge Intelligence.' \
		'repository: https://github.com/Tovli/EdgeIntelligence' \
		'environment:' \
		'  sdk: ">=3.3.0 <4.0.0"' \
		'dependencies:' \
		'  flutter_rust_bridge: ^$(FRB_VERSION)' \
		> $(OUT)/flutter/pubspec.yaml
	@cp LICENSE $(OUT)/flutter/LICENSE
	@cp packaging/flutter/README.md $(OUT)/flutter/README.md
	@printf '%s\n' \
		'# Changelog' \
		'' \
		'## 0.1.0' \
		'' \
		'Release notes are tracked in the Edge Intelligence repository tags and GitHub releases.' \
		> $(OUT)/flutter/CHANGELOG.md
	flutter_rust_bridge_codegen generate \
		--rust-root crates/adapters/el-ffi \
		--rust-input crate:: \
		--dart-root $(OUT)/flutter \
		--dart-output $(OUT)/flutter/lib \
		--no-add-mod-to-lib \
		--no-auto-upgrade-dependency \
		--no-deps-check \
		--no-dart-format

## Build WASM output (identical to build-wasm; alias for consistency).
codegen-web: build-wasm

## Run all three codegen surfaces.
bindings: codegen-rn codegen-flutter codegen-web
