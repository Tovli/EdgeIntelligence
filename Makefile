SHELL := bash

# ─── Targets ─────────────────────────────────────────────────────────────────
ANDROID_TARGET := aarch64-linux-android
IOS_TARGET     := aarch64-apple-ios
FFI            := --manifest-path crates/adapters/el-ffi/Cargo.toml
OUT            := out

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
		--out-dir ../../$(OUT)/web

# ─── Binding codegen ─────────────────────────────────────────────────────────
#
# Prerequisites (install once)
#   RN:      cargo install uniffi-bindgen-react-native --locked
#   Flutter: cargo install flutter_rust_bridge_codegen --locked
#   Web:     (wasm-pack, covered by build-wasm)

## Generate React Native (TypeScript + JSI + Turbo Module) bindings.
## Requires: build-android
codegen-rn: build-android
	@mkdir -p $(OUT)/rn
	uniffi-bindgen-react-native generate \
		--library target/$(ANDROID_TARGET)/release/libel_ffi.so \
		--out-dir $(OUT)/rn \
		--crate el-ffi

## Generate Flutter/Dart bindings via flutter_rust_bridge v2 codegen.
codegen-flutter:
	@mkdir -p $(OUT)/flutter
	flutter_rust_bridge_codegen generate \
		--rust-root crates/adapters/el-ffi \
		--dart-output $(OUT)/flutter

## Build WASM output (identical to build-wasm; alias for consistency).
codegen-web: build-wasm

## Run all three codegen surfaces.
bindings: codegen-rn codegen-flutter codegen-web
