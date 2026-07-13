#!/usr/bin/env python3
"""Validate CI dependencies that are easy to accidentally leave implicit."""

from pathlib import Path
import re
import sys


WORKFLOWS = [
    Path(".github/workflows/release.yml"),
    Path(".github/workflows/bindings.yml"),
]

RN_RETRY_COMMAND = "bash scripts/retry-command.sh make codegen-rn"
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
        for trigger_path in (
            "packaging/dart/**",
            "scripts/build-ios-xcframework.sh",
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

    if path.name == "release.yml":
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
        if "Run Flutter iOS package smoke build" not in text:
            errors.append(f"{path}: missing Flutter iOS simulator smoke build")
        parts = text.split("  assemble-dart:", 1)
        if len(parts) < 2:
            errors.append(f"{path}: missing assemble-dart job")
            return errors
        assemble_dart = parts[1].split("\n  publish-crates:", 1)[0]
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
            "assembly/ios/Frameworks/el_ffi.xcframework",
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


def main() -> int:
    errors: list[str] = []
    for workflow in WORKFLOWS:
        errors.extend(check_workflow(workflow))
    errors.extend(check_dart_mobile_package())

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print("OK: CI workflows guard Dart codegen tools and retry external codegen downloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
