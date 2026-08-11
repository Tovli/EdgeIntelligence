#!/usr/bin/env python3
"""Validate CI dependencies that are easy to accidentally leave implicit."""

from __future__ import annotations

from pathlib import Path
import re
import sys


WORKFLOWS = [
    Path(".github/workflows/release.yml"),
    Path(".github/workflows/bindings.yml"),
]

RN_RETRY_COMMAND = "bash scripts/retry-command.sh make codegen-rn"
UBRN_OCCURRENCE_PATTERN = re.compile(r"uniffi-bindgen-react-native@")
RETRY_WRAPPED_UBRN_PREFIX = re.compile(
    r"bash\s+(?:\.\./)?scripts/retry-command\.sh\s+"
    r"(?:npm\s+(?:install|i)\s+(?:--global|-g)|npx\s+(?:--yes|-y))\s*$"
)
UBRN_NPX_PREFIX = re.compile(r"npx\s+(?:--yes|-y)\s*$")
WASM_RETRY_COMMAND = "bash scripts/retry-command.sh curl -fsSL"
BINDINGS_UPLOAD_IF = "if: github.event_name != 'pull_request'"
DART_FRB_STALENESS_CHECK = (
    "git diff --exit-code -- crates/adapters/el-ffi/src/frb_generated.rs"
)
PUB_DESKTOP_ARTIFACTS = [
    "linux-x64-lib",
    "macos-universal-lib",
    "windows-x64-lib",
]
PUB_MOBILE_ARTIFACTS = [
    "android-libs",
    "ios-xcframework",
]
QWEN_FIXTURE_SCRIPT = Path("scripts/download-qwen-fixture.sh")
QWEN_FIXTURE_MANIFEST = Path("scripts/qwen-fixture.sha256")
QWEN_FFI_TEST_SCRIPT = Path("scripts/run-qwen-ffi-integration.sh")
QWEN_FFI_TEST_NAME = "native_qwen_integration_decodes_and_streams_english_text"
ADR_026_RELEASE_GUARD = "Assert ADR-026 migration has a minor release"
ADR_026_VERSION_GUARD = "scripts/assert-adr-026-release-version.py"

REQUIRED_INSTALLS = [
    (
        "flutter_rust_bridge_codegen",
        "scripts/install-cargo-tool.sh flutter_rust_bridge_codegen flutter_rust_bridge_codegen",
    ),
    (
        "cargo-expand",
        "scripts/install-cargo-tool.sh cargo-expand cargo-expand",
    ),
]


def workflow_job_body(text: str, job_name: str) -> str | None:
    """Return one top-level workflow job without depending on its successor."""
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\s*\n"
        rf"(.*?)(?=^  [A-Za-z0-9_-]+:\s*$|\Z)",
        text,
    )
    return match.group(1) if match else None


def check_workflow(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    codegen_pos = text.find("make codegen-dart")
    rn_codegen_pos = text.find("make codegen-rn")

    if codegen_pos < 0:
        return [f"{path}: missing make codegen-dart step"]

    if DART_FRB_STALENESS_CHECK not in text[codegen_pos:]:
        errors.append(
            f"{path}: missing Dart FRB Rust glue staleness check after codegen step"
        )

    for label, needle in REQUIRED_INSTALLS:
        install_pos = text.find(needle)
        if install_pos < 0:
            errors.append(f"{path}: missing explicit {label} install before Dart codegen")
        elif install_pos > codegen_pos:
            errors.append(f"{path}: installs {label} after Dart codegen")

    if rn_codegen_pos >= 0 and RN_RETRY_COMMAND not in text:
        errors.append(f"{path}: React Native codegen must run through retry wrapper")
    for invocation in UBRN_OCCURRENCE_PATTERN.finditer(text):
        prefix = text[max(0, invocation.start() - 512) : invocation.start()]
        if RETRY_WRAPPED_UBRN_PREFIX.search(prefix):
            continue
        if UBRN_NPX_PREFIX.search(prefix):
            errors.append(f"{path}: UBRN npx invocation must run through retry wrapper")
        else:
            errors.append(f"{path}: UBRN installation must run through retry wrapper")

    if "aarch64-linux-android" in text:
        for mobile_build_requirement in (
            "armv7-linux-androideabi",
            "x86_64-linux-android",
            "ndk-version: r28c",
            "armv7a-linux-androideabi24-clang",
            "aarch64-linux-android24-clang",
            "x86_64-linux-android24-clang",
            "bash scripts/build-ios-xcframework.sh",
        ):
            if mobile_build_requirement not in text:
                errors.append(
                    f"{path}: mobile validation missing {mobile_build_requirement!r}"
                )
        if "Verify Android 16 KB page alignment" not in text:
            errors.append(f"{path}: missing Android 16 KB page-alignment gate")

    if "Install wasm-pack" in text and WASM_RETRY_COMMAND not in text:
        errors.append(f"{path}: wasm-pack download must run through retry wrapper")

    if path.name == "bindings.yml":
        if "npm run test:rn-factory" not in text:
            errors.append(f"{path}: React Native factory runtime test must run in PR validation")
        for trigger_path in (
            "packaging/dart/**",
            "scripts/build-ios-xcframework.sh",
            "scripts/download-qwen-fixture.sh",
            "scripts/qwen-fixture.sha256",
            "scripts/run-qwen-ffi-integration.sh",
            "scripts/assert-adr-026-release-version.py",
            "scripts/expo-android-local-session-smoke.sh",
            "scripts/verify-android-page-alignment.py",
        ):
            if text.count(trigger_path) < 2:
                errors.append(
                    f"{path}: push and pull_request filters must include {trigger_path}"
                )
        step_blocks = text.split("\n      - ")
        for block in step_blocks:
            if "uses: actions/upload-artifact@v4" in block and BINDINGS_UPLOAD_IF not in block:
                errors.append(
                    f"{path}: PR validation artifact uploads must be gated with "
                    f"{BINDINGS_UPLOAD_IF!r}"
                )
        for job_name in ("host", "ios"):
            job = workflow_job_body(text, job_name)
            if job is None:
                errors.append(f"{path}: missing {job_name} FFI validation job")
                continue
            for required_qwen_fixture_part in (
                "actions/cache@v4",
                "hashFiles('scripts/qwen-fixture.sha256')",
                "scripts/download-qwen-fixture.sh qwen-fixture",
                "EDGE_INTELLIGENCE_QWEN_GGUF",
                "bash scripts/run-qwen-ffi-integration.sh",
            ):
                if required_qwen_fixture_part not in job:
                    errors.append(
                        f"{path}: {job_name} FFI validation must restore, verify, and run the real Qwen fixture ({required_qwen_fixture_part!r})"
                    )

    if path.name == "release.yml":
        if ADR_026_RELEASE_GUARD not in text:
            errors.append(
                f"{path}: missing ADR-026 minor-release guard for the one-path React Native migration"
            )
        if f"python3 {ADR_026_VERSION_GUARD}" not in text:
            errors.append(
                f"{path}: ADR-026 release guard must run {ADR_026_VERSION_GUARD}"
            )
        if "npm run test:rn-factory" not in text:
            errors.append(f"{path}: React Native factory runtime test must run before publishing")
        if "Validate Dart package resolves without Flutter SDK" in text:
            errors.append(
                f"{path}: Flutter plugin packages cannot claim standalone Dart resolution"
            )
        if text.count("flutter pub get --no-example") < 2:
            errors.append(f"{path}: Flutter plugin resolution must use flutter pub get")
        if "flutter pub publish --dry-run" not in text:
            errors.append(f"{path}: pub.dev validation must run in Flutter context")
        if "flutter pub publish --force" not in text:
            errors.append(f"{path}: pub.dev publishing must run in Flutter context")
        if "verify-flutter-ios:" not in text:
            errors.append(f"{path}: missing Flutter iOS package verification job")
        verify = workflow_job_body(text, "verify")
        if verify is None:
            errors.append(f"{path}: missing release verification job")
        else:
            for required_qwen_fixture_part in (
                "actions/cache@v4",
                "hashFiles('scripts/qwen-fixture.sha256')",
                "scripts/download-qwen-fixture.sh qwen-fixture",
                "EDGE_INTELLIGENCE_QWEN_GGUF",
                "bash scripts/run-qwen-ffi-integration.sh",
            ):
                if required_qwen_fixture_part not in verify:
                    errors.append(
                        f"{path}: release verification must provision and run the real Qwen fixture ({required_qwen_fixture_part!r})"
                    )
        for smoke_name in (
            "Run Flutter iOS CocoaPods package smoke build",
            "Run Flutter iOS SwiftPM package smoke build",
        ):
            if smoke_name not in text:
                errors.append(f"{path}: missing {smoke_name}")
        assemble_npm = workflow_job_body(text, "assemble-npm")
        if assemble_npm is None:
            errors.append(f"{path}: missing assemble-npm job")
        else:
            generation_pos = assemble_npm.find("generate jsi turbo-module")
            if generation_pos < 0:
                errors.append(f"{path}: npm assembly missing turbo-module generation step")
            else:
                for required_input in (
                    "test -s src/rn/el_ffi.ts",
                    "test -s src/rn/cpp/el_ffi.cpp",
                    "test -s src/rn/cpp/el_ffi.hpp",
                    "test ! -f android/proguard-rules.pro",
                    "node-version: '22.13'",
                    "uses: dtolnay/rust-toolchain@stable",
                    "uses: Swatinem/rust-cache@v2",
                ):
                    if required_input not in assemble_npm:
                        errors.append(
                            f"{path}: npm assembly missing {required_input!r}"
                        )
                    elif required_input.startswith("test -s src/rn/"):
                        if assemble_npm.find(required_input) > generation_pos:
                            errors.append(
                                f"{path}: npm assembly must validate downloaded "
                                f"{required_input.removeprefix('test -s ')} before turbo-module generation"
                            )
                for generated_output in (
                    "test -f android/src/main/java/com/edgeintelligence/EdgeIntelligenceSdkModule.java",
                    "test -f ios/EdgeIntelligenceSdk.mm",
                    "test -f src/rn/native.ts",
                    "test -f src/rn/NativeEdgeIntelligenceSdk.ts",
                ):
                    if generated_output not in assemble_npm:
                        errors.append(
                            f"{path}: npm assembly missing generated output "
                            f"{generated_output.removeprefix('test -f ')}"
                        )
                    elif assemble_npm.find(generated_output) < generation_pos:
                        errors.append(
                            f"{path}: npm assembly validates generated output "
                            f"{generated_output.removeprefix('test -f ')} before generation"
                        )
        expo_smoke = workflow_job_body(text, "smoke-expo-native-module")
        if expo_smoke is None:
            errors.append(f"{path}: missing smoke-expo-native-module job")
        elif "lib/x86_64/libel_ffi.so" not in expo_smoke:
            errors.append(f"{path}: Expo Android smoke must assert the x86_64 Rust library")
        else:
            for required_qwen_smoke_part in (
                "Restore Qwen Android smoke fixture cache",
                "hashFiles('scripts/qwen-fixture.sha256')",
                "scripts/download-qwen-fixture.sh qwen-fixture",
                "EDGE_INTELLIGENCE_QWEN_GGUF",
                ":app:assembleRelease",
                "expo-android-local-session-smoke.sh",
            ):
                if required_qwen_smoke_part not in expo_smoke:
                    errors.append(
                        f"{path}: Expo Android smoke must exercise packaged Qwen inference ({required_qwen_smoke_part!r})"
                    )
        if "huggingface.co/Qwen/" in text:
            errors.append(
                f"{path}: Qwen fixture URLs belong only in the pinned downloader script"
            )
        assemble_dart = workflow_job_body(text, "assemble-dart")
        if assemble_dart is None:
            errors.append(f"{path}: missing assemble-dart job")
            return errors
        for artifact in PUB_DESKTOP_ARTIFACTS + PUB_MOBILE_ARTIFACTS:
            if artifact not in assemble_dart:
                errors.append(f"{path}: pub.dev assembly must include {artifact}")
        for artifact in ("wasm-output",):
            if artifact in assemble_dart:
                errors.append(
                    f"{path}: pub.dev assembly must not include unsupported {artifact}"
                )
        if "Check pub.dev package size" not in assemble_dart:
            errors.append(f"{path}: pub.dev assembly must check package size")
        if "Run Dart package runtime smoke tests" not in assemble_dart:
            errors.append(
                f"{path}: pub.dev assembly must run packaged Dart runtime smoke tests"
            )
        if "Run Flutter Android package smoke build" not in assemble_dart:
            errors.append(
                f"{path}: pub.dev assembly must smoke-build the Flutter Android example"
            )

        required_mobile_paths = (
            "assembly/android/src/main/jniLibs/armeabi-v7a/libel_ffi.so",
            "assembly/android/src/main/jniLibs/arm64-v8a/libel_ffi.so",
            "assembly/android/src/main/jniLibs/x86_64/libel_ffi.so",
            "assembly/ios/edge_intelligence/Frameworks/el_ffi.xcframework",
        )
        for mobile_path in required_mobile_paths:
            if mobile_path not in assemble_dart:
                errors.append(
                    f"{path}: pub.dev assembly must package {mobile_path}"
                )

        for android_build_requirement in (
            "ndk-version: r28c",
            "armv7a-linux-androideabi24-clang",
            "aarch64-linux-android24-clang",
            "x86_64-linux-android24-clang",
        ):
            if android_build_requirement not in text:
                errors.append(
                    f"{path}: Android build missing {android_build_requirement!r}"
                )

    return errors


def check_dart_mobile_package() -> list[str]:
    errors: list[str] = []
    pubspec = Path("packaging/dart/pubspec.yaml").read_text(encoding="utf-8")
    loader = Path("packaging/dart/lib/src/runtime_loader_io.dart").read_text(
        encoding="utf-8"
    )
    makefile = Path("Makefile").read_text(encoding="utf-8")

    required_pubspec_fragments = (
        "android:",
        "ios:",
        "flutter:",
        "ffiPlugin: true",
        'flutter_rust_bridge: ">=2.12.0 <2.12.1"',
    )
    for fragment in required_pubspec_fragments:
        if fragment not in pubspec:
            errors.append(
                f"packaging/dart/pubspec.yaml: missing Flutter mobile declaration {fragment!r}"
            )

    required_files = (
        Path("packaging/dart/android/build.gradle"),
        Path("packaging/dart/android/src/main/AndroidManifest.xml"),
        Path("packaging/dart/ios/edge_intelligence.podspec"),
        Path("packaging/dart/ios/edge_intelligence/Package.swift"),
        Path("packaging/dart/example/pubspec.yaml"),
        Path("packaging/dart/example/lib/main.dart"),
    )
    for path in required_files:
        if not path.is_file():
            errors.append(f"{path}: required Flutter mobile package file is missing")

    required_mobile_loaders = (
        r"if \(Platform\.isAndroid\).*?ExternalLibrary\.open\(\s*'libel_ffi\.so'",
        r"if \(Platform\.isIOS\).*?ExternalLibrary\.open\(\s*'el_ffi\.framework/el_ffi'",
    )
    for loader_pattern in required_mobile_loaders:
        if re.search(loader_pattern, loader, re.DOTALL) is None:
            errors.append(
                "packaging/dart/lib/src/runtime_loader_io.dart: "
                f"missing automatic mobile loader matching {loader_pattern!r}"
            )

    for package_dir in ("android", "ios"):
        copy_command = f"packaging/dart/{package_dir}"
        if copy_command not in makefile:
            errors.append(
                f"Makefile: codegen-dart must copy the Flutter {package_dir} plugin layout"
            )

    return errors


def check_qwen_fixture() -> list[str]:
    errors: list[str] = []
    if not QWEN_FIXTURE_SCRIPT.is_file():
        return [f"{QWEN_FIXTURE_SCRIPT}: missing pinned Qwen fixture downloader"]
    if not QWEN_FIXTURE_MANIFEST.is_file():
        return [f"{QWEN_FIXTURE_MANIFEST}: missing Qwen fixture SHA-256 manifest"]

    downloader = QWEN_FIXTURE_SCRIPT.read_text(encoding="utf-8")
    manifest = QWEN_FIXTURE_MANIFEST.read_text(encoding="utf-8")
    if "/resolve/main/" in downloader:
        errors.append(
            f"{QWEN_FIXTURE_SCRIPT}: fixture downloads must pin immutable revisions, not main"
        )
    for required_downloader_part in (
        "sha256sum -c",
        "qwen-fixture.sha256",
        "9217f5db79a29953eb74d5343926648285ec7e67",
        "7ae557604adf67be50417f59c2c2f167def9a775",
    ):
        if required_downloader_part not in downloader:
            errors.append(
                f"{QWEN_FIXTURE_SCRIPT}: missing immutable fixture verification {required_downloader_part!r}"
            )
    for required_manifest_part in (
        "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db  qwen.gguf",
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539  tokenizer.json",
    ):
        if required_manifest_part not in manifest:
            errors.append(
                f"{QWEN_FIXTURE_MANIFEST}: missing expected SHA-256 entry {required_manifest_part!r}"
            )
    return errors


def check_qwen_ffi_test_runner() -> list[str]:
    if not QWEN_FFI_TEST_SCRIPT.is_file():
        return [f"{QWEN_FFI_TEST_SCRIPT}: missing real Qwen FFI test runner"]

    runner = QWEN_FFI_TEST_SCRIPT.read_text(encoding="utf-8")
    errors: list[str] = []
    for required_part in (
        f'readonly test_name="{QWEN_FFI_TEST_NAME}"',
        'cargo test -p el-ffi "$test_name" -- --ignored',
        'grep -F "running 1 test"',
        'grep -F "$test_name ... ok"',
        'grep -F "test result: ok. 1 passed;"',
    ):
        if required_part not in runner:
            errors.append(
                f"{QWEN_FFI_TEST_SCRIPT}: must prove exactly one named real-Qwen test ran ({required_part!r})"
            )
    return errors


def main() -> int:
    errors: list[str] = []
    for workflow in WORKFLOWS:
        errors.extend(check_workflow(workflow))
    errors.extend(check_dart_mobile_package())
    errors.extend(check_qwen_fixture())
    errors.extend(check_qwen_ffi_test_runner())

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print("OK: CI workflows guard tool dependencies and retry external downloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
