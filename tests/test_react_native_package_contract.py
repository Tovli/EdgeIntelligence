"""Static release-contract checks for the React Native/Expo package template."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging" / "npm"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
BINDINGS_WORKFLOW = ROOT / ".github" / "workflows" / "bindings.yml"
ADR = ROOT / "docs" / "adr" / "ADR-025-react-native-expo-autolink-ready-native-distribution.md"
ADR_026 = ROOT / "docs" / "adr" / "ADR-026-tokenizer-aware-qwen-local-sessions-for-react-native.md"
ADR_026_RELEASE_VERSION_GUARD = ROOT / "scripts" / "assert-adr-026-release-version.py"
UBRN_CMAKE_PATCH = ROOT / "scripts" / "patch-ubrn-cmake.py"
UBRN_VERSION = "0.31.0-3"


def load_ci_checker():
    spec = importlib.util.spec_from_file_location(
        "check_ci_workflows", ROOT / "tests" / "check_ci_workflows.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ubrn_cmake_patch():
    spec = importlib.util.spec_from_file_location("patch_ubrn_cmake", UBRN_CMAKE_PATCH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_declares_generated_runtime_dependencies() -> None:
    package = json.loads((PACKAGE / "package.json").read_text(encoding="utf-8"))

    assert package["repository"]["url"] == "https://github.com/Tovli/EdgeIntelligence.git"
    assert package["dependencies"]["@ubjs/core"] == UBRN_VERSION
    assert package["dependencies"]["uniffi-bindgen-react-native"] == UBRN_VERSION
    assert package["peerDependencies"]["react"] == ">=19.2.0 <20"
    assert package["peerDependencies"]["react-native"] == ">=0.85.0 <0.87.0"
    assert package["peerDependenciesMeta"]["react"]["optional"] is True
    assert package["peerDependenciesMeta"]["react-native"]["optional"] is True
    assert "expo" not in package["peerDependencies"]
    assert "expo" not in package["peerDependenciesMeta"]
    assert package["main"] == "./src/unsupported-node.cjs"

    for path in (
        RELEASE_WORKFLOW,
        BINDINGS_WORKFLOW,
    ):
        assert f"uniffi-bindgen-react-native@{UBRN_VERSION}" in path.read_text(
            encoding="utf-8"
        )


def test_rust_uniffi_matches_ubrn_codegen_abi() -> None:
    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert metadata.returncode == 0, metadata.stderr

    packages = json.loads(metadata.stdout)["packages"]
    ffi = next(package for package in packages if package["name"] == "el-ffi")
    uniffi = next(
        dependency for dependency in ffi["dependencies"] if dependency["name"] == "uniffi"
    )
    expected_uniffi = UBRN_VERSION.rsplit("-", 1)[0]

    assert uniffi["req"] == f"={expected_uniffi}"


def test_adr_matches_the_expo_autolinking_dependency_contract() -> None:
    adr = ADR.read_text(encoding="utf-8")
    normalized = " ".join(adr.split())

    assert "React Native remains a peer dependency" in normalized
    assert "Expo is not a peer dependency" in normalized
    assert "verified host for this decision is Expo SDK 57 / React Native 0.86" in normalized
    assert "React Native and Expo\n   remain peer dependencies" not in adr


def test_adr_026_matches_the_tokenizer_aware_qwen_contract() -> None:
    adr = ADR_026.read_text(encoding="utf-8")
    normalized = " ".join(adr.split())

    assert "localEdgeLlm(modelUri, tokenizerUri)" in normalized
    assert "localEdgeLlm(modelUri)" in normalized
    assert "QwenChatProvider::from_paths" in adr
    assert "all-`?`" in adr
    assert "64 generated tokens" in adr
    assert "deferred to a separately scoped mobile" in adr
    assert "0.4.0-or-later minor release" in adr


def test_adr_026_release_version_guard_requires_a_minor_bump() -> None:
    allowed = subprocess.run(
        [sys.executable, str(ADR_026_RELEASE_VERSION_GUARD), "0.4.0"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    rejected = subprocess.run(
        [sys.executable, str(ADR_026_RELEASE_VERSION_GUARD), "0.3.15"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert allowed.returncode == 0, allowed.stderr
    assert "at or above 0.4.0" in allowed.stdout
    assert rejected.returncode == 1
    assert "below the required 0.4.0 minor release" in rejected.stderr


def test_package_codegen_names_match_ubrn_turbo_module() -> None:
    package = json.loads((PACKAGE / "package.json").read_text(encoding="utf-8"))
    config = (PACKAGE / "ubrn.config.yaml").read_text(encoding="utf-8")
    react_native_config = (PACKAGE / "react-native.config.js").read_text(
        encoding="utf-8"
    )

    codegen_name = package["codegenConfig"]["name"]
    configured_spec = re.search(r"(?m)^\s+spec:\s+(\S+)\s*$", config)
    assert configured_spec is not None
    assert codegen_name == "EdgeIntelligenceSdkSpec"
    module_name = codegen_name.removesuffix("Spec")
    assert module_name == "EdgeIntelligenceSdk"
    assert configured_spec.group(1) == codegen_name
    assert "outputDir" not in package["codegenConfig"]
    assert "cmakeListsPath" not in react_native_config
    assert "entrypoint: src/rn/native.ts" in config
    assert "android/build.gradle" in config
    assert "android/src/main/AndroidManifest.xml" in config
    assert "android/proguard-rules.pro" in config
    assert "- \"*.podspec\"" in config

    gradle = (PACKAGE / "android" / "build.gradle").read_text(encoding="utf-8")
    assert 'apply plugin: "com.facebook.react"' in gradle
    assert 'path "CMakeLists.txt"' in gradle
    assert f'libraryName = "{module_name}"' in gradle
    assert "JavaVersion.VERSION_17" in gradle
    assert "crates/ubrn_cli/src/jsi/android/codegen.rs:32-40" in gradle
    assert "kotlin" not in gradle.lower()


def test_package_exposes_react_native_and_expo_native_metadata() -> None:
    assert (PACKAGE / "react-native.config.js").is_file()
    assert not (PACKAGE / "app.plugin.js").exists()
    assert (PACKAGE / "edge-intelligence-sdk.podspec").is_file()
    assert (PACKAGE / "android" / "build.gradle").is_file()
    assert (PACKAGE / "android" / "src" / "main" / "AndroidManifest.xml").is_file()
    assert (PACKAGE / "android" / "consumer-rules.pro").is_file()
    assert not (PACKAGE / "android" / "CMakeLists.txt").exists()
    assert (PACKAGE / "ubrn.config.yaml").is_file()
    assert (PACKAGE / "src" / "rn" / "index.ts").is_file()
    assert (PACKAGE / "example-expo" / "app.json").is_file()
    assert (PACKAGE / "example-expo" / "index.js").is_file()
    assert (PACKAGE / "android" / ".npmignore").read_text(
        encoding="utf-8"
    ).splitlines() == [
        ".gradle/"
    ]
    manifest = (PACKAGE / "android" / "src" / "main" / "AndroidManifest.xml").read_text(
        encoding="utf-8"
    )
    assert "package=" not in manifest
    consumer_rules = (PACKAGE / "android" / "consumer-rules.pro").read_text(
        encoding="utf-8"
    )
    assert "-keep class com.sun.jna.*" in consumer_rules
    assert "-keepclassmembers class * extends com.sun.jna.*" in consumer_rules

    podspec = (PACKAGE / "edge-intelligence-sdk.podspec").read_text(
        encoding="utf-8"
    )
    assert "install_modules_dependencies(s)" in podspec
    assert "s.dependency 'uniffi-bindgen-react-native', '0.31.0-3'" in podspec
    assert "respond_to?(:min_ios_version_supported, true)" in podspec
    assert "ios/EdgeIntelligenceSdk.{h,mm}" in podspec
    assert "ios/generated/**/*.{h,hpp,cpp,m,mm}" in podspec
    assert "ios/**/*" not in podspec


def test_published_package_excludes_its_source_only_expo_fixture() -> None:
    package = json.loads((PACKAGE / "package.json").read_text(encoding="utf-8"))

    assert "example-expo/" not in package["files"]
    assert "app.plugin.js" not in package["files"]


def test_runtime_guard_explains_how_to_load_the_native_module() -> None:
    entrypoint = (PACKAGE / "src" / "rn" / "index.ts").read_text(encoding="utf-8")
    package = json.loads((PACKAGE / "package.json").read_text(encoding="utf-8"))

    assert "require('./native')" in entrypoint
    assert "Expo Go" in entrypoint
    assert "rebuild" in entrypoint
    assert "const native = loadNativeBindings();" in entrypoint
    assert "requireNativeElFfi();\n  return loadNativeBindings()" not in entrypoint
    assert "import type { EdgeLlm, EdgeLlmLike }" in entrypoint
    assert "): EdgeLlmLike" in entrypoint
    assert "export type { EdgeLlm, EdgeLlmLike, SdkError }" in entrypoint
    assert "cloudEdgeLlm" in entrypoint
    assert package["types"] == "./src/web/el_ffi.d.ts"
    assert list(package["exports"]["."]) == [
        "react-native",
        "browser",
        "types",
        "default",
    ]
    assert package["exports"]["."]["types"] == "./src/web/el_ffi.d.ts"
    assert package["exports"]["."]["react-native"]["types"] == "./src/rn/index.d.ts"
    assert package["exports"]["."]["browser"]["types"] == "./src/web/el_ffi.d.ts"
    assert package["exports"]["."]["default"] == "./src/unsupported-node.cjs"
    assert (PACKAGE / "src" / "unsupported-node.cjs").is_file()
    assert (PACKAGE / "tsconfig.rn.json").is_file()
    assert (PACKAGE / "typecheck" / "rn-public-api.ts").is_file()
    public_typecheck = (PACKAGE / "typecheck" / "rn-public-api.ts").read_text(
        encoding="utf-8"
    )
    assert "IsExact" in public_typecheck
    assert "ReturnType<typeof localEdgeLlm>" in public_typecheck
    tsconfig = (PACKAGE / "tsconfig.rn.json").read_text(encoding="utf-8")
    assert '"customConditions": ["react-native"]' in tsconfig
    assert '"typecheck/rn-public-api.ts"' in tsconfig

    native_types = (PACKAGE / "src" / "rn" / "index.d.ts").read_text(
        encoding="utf-8"
    )
    assert not (PACKAGE / "src" / "index.d.ts").exists()
    assert "localEdgeLlm" in native_types
    assert "cloudEdgeLlm" in native_types
    assert "EdgeLlmLike" in native_types
    assert "react-native" not in native_types
    assert "typeof bindings.EdgeLlm.localQwen" in entrypoint
    assert "generated native entrypoint did not export EdgeLlm" in entrypoint


def test_react_native_qwen_factory_requires_both_local_asset_paths() -> None:
    ffi = (ROOT / "crates" / "adapters" / "el-ffi" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    entrypoint = (PACKAGE / "src" / "rn" / "index.ts").read_text(encoding="utf-8")
    native_types = (PACKAGE / "src" / "rn" / "index.d.ts").read_text(encoding="utf-8")
    public_typecheck = (PACKAGE / "typecheck" / "rn-public-api.ts").read_text(
        encoding="utf-8"
    )

    assert "pub fn local_qwen(model_uri: String, tokenizer_uri: String)" in ffi
    assert "QwenChatProvider::from_paths(&model_uri, &tokenizer_uri)" in ffi
    assert "Qwen model path must not be empty" in ffi
    assert "Qwen tokenizer path must not be empty" in ffi
    assert "native_qwen_integration_decodes_and_streams_english_text" in ffi
    qwen_runner = ROOT / "scripts" / "run-qwen-ffi-integration.sh"
    assert qwen_runner.is_file()
    qwen_runner_source = qwen_runner.read_text(encoding="utf-8")
    assert 'readonly test_name="native_qwen_integration_decodes_and_streams_english_text"' in qwen_runner_source
    assert 'grep -F "running 1 test"' in qwen_runner_source
    assert 'grep -F "$test_name ... ok"' in qwen_runner_source
    assert "ask_stream_cb" in ffi
    assert "max_tokens: Some(QWEN_FFI_DEFAULT_MAX_TOKENS)" in ffi
    assert "QWEN_FFI_DEFAULT_MAX_TOKENS: u32 = 64" in ffi
    assert "fn reset(&self) -> Result<(), SdkError>" in ffi
    assert "self.provider.end_session()" in ffi
    assert "model: {model_uri}, tokenizer: {tokenizer_uri}" in ffi
    assert "qwen_ffi_cap_is_applied_to_ask_and_stream_requests" in ffi
    assert "native.EdgeLlm.localQwen(modelUri, tokenizerUri)" in entrypoint
    assert "requireLocalAssetPath(modelUri, 'modelUri')" in entrypoint
    assert "requireLocalAssetPath(tokenizerUri, 'tokenizerUri')" in entrypoint
    assert (
        "localEdgeLlm(modelUri: string, tokenizerUri: string): EdgeLlmLike" in native_types
    )
    assert "localEdgeLlm(modelUri: string): EdgeLlmLike" in native_types
    assert "Parameters<typeof localEdgeLlm>" in public_typecheck
    assert "[modelUri: string, tokenizerUri: string]" in public_typecheck
    assert "const legacyLocal: (modelUri: string)" in public_typecheck
    assert "Migrate to localEdgeLlm(modelUri, tokenizerUri)." in entrypoint
    runtime_test = PACKAGE / "typecheck" / "rn-factory-runtime-test.cjs"
    assert runtime_test.is_file()
    assert "localQwen" in runtime_test.read_text(encoding="utf-8")


def test_expo_fixture_targets_current_supported_host() -> None:
    app = (PACKAGE / "example-expo" / "App.tsx").read_text(encoding="utf-8")
    app_config = (PACKAGE / "example-expo" / "app.json").read_text(encoding="utf-8")
    fixture = json.loads(
        (PACKAGE / "example-expo" / "package.json").read_text(encoding="utf-8")
    )

    assert fixture["main"] == "index.js"
    assert fixture["dependencies"]["expo"] == "~57.0.8"
    assert fixture["dependencies"]["react"] == "19.2.3"
    assert fixture["dependencies"]["react-native"] == "0.86.0"
    lock = json.loads(
        (PACKAGE / "example-expo" / "package-lock.json").read_text(encoding="utf-8")
    )
    assert lock["packages"][""]["dependencies"] == fixture["dependencies"]
    assert lock["packages"]["node_modules/expo"]["version"] == "57.0.9"
    assert lock["packages"]["node_modules/react-native"]["version"] == "0.86.0"
    assert "Edge Intelligence Qwen local session passed." in app
    assert "function describeError(error: unknown): string" in app
    assert "error.inner.message" in app
    smoke_script = (
        ROOT / "scripts" / "expo-android-local-session-smoke.sh"
    ).read_text(encoding="utf-8")
    assert "Edge Intelligence Qwen local session passed." in smoke_script
    assert "localEdgeLlm(modelUri, tokenizerUri)" in app
    assert "askStreamCb" in app
    assert "sdk.reset();" in app
    assert "EDGE_INTELLIGENCE_QWEN_GGUF" in smoke_script
    assert "Edge Intelligence Qwen local session failed:" in smoke_script
    assert "Local-session failure UI follows." in smoke_script
    assert "dump_ui >&2" in smoke_script
    assert "adb push" in smoke_script
    assert 'chmod 755 "$app_external_dir" "$app_files_dir" "$app_assets_dir"' in smoke_script
    assert 'chmod 644 "$app_model_path" "$app_tokenizer_path"' in smoke_script
    assert 'readonly app_external_dir="/sdcard/Android/data/$package_name"' in smoke_script
    assert 'readonly app_assets_dir="$app_files_dir/models"' in smoke_script
    assert "app-release.apk" in smoke_script
    assert "app-debug.apk" not in smoke_script
    assert "localEdgeLlm('')" not in app
    assert "export const sdk" not in app
    assert '"plugins"' not in app_config
    assert (PACKAGE / "example-expo" / "package-lock.json").is_file()
    fixture_readme = (PACKAGE / "example-expo" / "README.md").read_text(
        encoding="utf-8"
    )
    assert "release CI fixture" in fixture_readme

    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "Assert ADR-026 migration has a minor release" in workflow
    assert "python3 scripts/assert-adr-026-release-version.py" in workflow
    assert "npm install --package-lock-only --ignore-scripts" in workflow
    assert "bash ../scripts/retry-command.sh npm ci" in workflow
    assert "Run React Native package type check" in workflow
    assert "npm run typecheck:rn" in workflow
    assert "npm run test:rn-factory" in workflow
    assert "scripts/download-qwen-fixture.sh qwen-fixture" in workflow
    assert ":app:assembleRelease" in workflow
    assert ":app:assembleDebug" not in workflow


def test_expo_ios_smoke_selects_only_top_level_workspace() -> None:
    checker = load_ci_checker()
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    expo_smoke = checker.workflow_job_body(workflow, "smoke-expo-native-module")
    assert expo_smoke is not None
    assert "ios/*.xcworkspace" in expo_smoke
    assert "find ios -name '*.xcworkspace'" not in expo_smoke


def test_android_emulator_runner_receives_a_single_command() -> None:
    checker = load_ci_checker()
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    expo_smoke = checker.workflow_job_body(workflow, "smoke-expo-native-module")
    assert expo_smoke is not None
    emulator_step = expo_smoke.split(
        "uses: reactivecircus/android-emulator-runner@v2", 1
    )[1]
    scalar = re.search(r"(?m)^\s+script:\s*(\S.*?)\s*$", emulator_step)
    assert scalar is not None
    command = scalar.group(1)
    assert command == "sh scripts/expo-android-local-session-smoke.sh"
    assert "\r" not in command and "\n" not in command
    assert "ram-size: 4096M" in emulator_step
    assert "heap-size: 1024M" in emulator_step

    smoke_script = ROOT / "scripts" / "expo-android-local-session-smoke.sh"
    assert smoke_script.is_file()
    smoke_source = smoke_script.read_text(encoding="utf-8")
    assert "dump_failure_logs" in smoke_source
    assert "adb logcat -d -v threadtime" in smoke_source
    shell = shutil.which("sh")
    if shell:
        syntax_check = subprocess.run(
            [shell, "-n", str(smoke_script)],
            capture_output=True,
            check=False,
            text=True,
        )
        assert syntax_check.returncode == 0, syntax_check.stderr


def test_documentation_uses_generated_typescript_names() -> None:
    readme = (PACKAGE / "README.md").read_text(encoding="utf-8")
    ffi_readme = (ROOT / "crates" / "adapters" / "el-ffi" / "README.md").read_text(
        encoding="utf-8"
    )
    ffi_source = (ROOT / "crates" / "adapters" / "el-ffi" / "src" / "lib.rs").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(readme.split())

    assert "askStreamCb" in readme
    assert "onToken" in readme
    assert "ask_stream_cb" not in readme
    assert "localEdgeLlm(qwen05b, qwenTokenizer)" in readme
    assert "matching official `tokenizer.json`" in readme
    assert "currently synchronous native calls" in readme
    assert "Every Qwen reply is capped at 64 generated" in readme
    assert "EdgeLlm.localQwen" in ffi_readme
    assert "askStreamCb" in ffi_readme
    assert "native log" not in ffi_source
    assert "caller must stop or rebuild" in ffi_source
    assert "self.provider.end_session().map_err(SdkError::from)" in ffi_source
    assert '"plugins": ["edge-intelligence-sdk"]' not in readme
    assert "Android x86 is not supported" in normalized


def test_workflow_job_slicing_uses_the_next_top_level_job() -> None:
    checker = load_ci_checker()
    workflow = """jobs:
  assemble-dart:
    steps:
      - run: assemble
  smoke-expo-native-module:
    steps:
      - run: smoke
"""

    body = checker.workflow_job_body(workflow, "assemble-dart")
    assert body is not None
    assert "run: assemble" in body
    assert "run: smoke" not in body


def test_ci_checker_rejects_unretried_multiline_ubrn_install() -> None:
    checker = load_ci_checker()
    workflow = """jobs:
  codegen:
    steps:
      - run: make codegen-dart
      - run: >-
          bash scripts/retry-command.sh npm install --global
          uniffi-bindgen-react-native@0.31.0-3
      - run: >-
          npm install --global
          uniffi-bindgen-react-native@0.31.0-3
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.yml"
        path.write_text(workflow, encoding="utf-8")
        errors = checker.check_workflow(path)

    assert any("UBRN installation must run through retry wrapper" in error for error in errors)


def test_ci_checker_rejects_unretried_ubrn_npx() -> None:
    checker = load_ci_checker()
    workflow = """jobs:
  codegen:
    steps:
      - run: make codegen-dart
      - run: npx --yes uniffi-bindgen-react-native@0.31.0-3 --help
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.yml"
        path.write_text(workflow, encoding="utf-8")
        errors = checker.check_workflow(path)

    assert any("UBRN npx invocation must run through retry wrapper" in error for error in errors)


def test_ci_checker_rejects_reworded_unretried_ubrn_commands() -> None:
    checker = load_ci_checker()
    workflow = """jobs:
  codegen:
    steps:
      - run: make codegen-dart
      - run: npm i -g uniffi-bindgen-react-native@0.31.0-3
      - run: npx -y uniffi-bindgen-react-native@0.31.0-3 --help
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.yml"
        path.write_text(workflow, encoding="utf-8")
        errors = checker.check_workflow(path)

    assert any("UBRN installation must run through retry wrapper" in error for error in errors)
    assert any("UBRN npx invocation must run through retry wrapper" in error for error in errors)


def test_ci_checker_reports_a_missing_turbo_module_generator() -> None:
    checker = load_ci_checker()
    workflow = """jobs:
  codegen:
    steps:
      - run: make codegen-dart
  assemble-npm:
    steps:
      - run: assemble
"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "release.yml"
        path.write_text(workflow, encoding="utf-8")
        errors = checker.check_workflow(path)

    assert any("missing turbo-module generation step" in error for error in errors)


def test_repository_hygiene_contract_is_explicit() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "*.png binary" in attributes
    assert "*.bat text eol=crlf" in attributes
    assert "*.ps1 text eol=crlf" in attributes
    assert "__pycache__/" in gitignore
    assert "packaging/npm/example-expo/android/" in gitignore
    assert "packaging/npm/example-expo/ios/" in gitignore
    assert "packaging/npm/example-expo/node_modules/" in gitignore
    assert "packaging/npm/example-expo/.expo/" in gitignore
    assert "packaging/npm/node_modules/" in gitignore
    assert "packaging/npm/example-expo/package-lock.json" not in gitignore


def test_contract_runner_discovers_tests() -> None:
    calls: list[str] = []
    namespace = {
        "test_first": lambda: calls.append("first"),
        "test_second": lambda: calls.append("second"),
        "helper": lambda: calls.append("helper"),
    }

    assert run_contract_tests(namespace) == 2
    assert calls == ["first", "second"]


def test_contract_runs_for_pull_requests() -> None:
    workflow = BINDINGS_WORKFLOW.read_text(encoding="utf-8")

    assert "'packaging/npm/**'" in workflow
    assert "'tests/test_react_native_package_contract.py'" in workflow
    assert workflow.count("'scripts/assert-adr-026-release-version.py'") == 2
    assert workflow.count("'.github/workflows/release.yml'") == 2
    assert "python3 tests/test_react_native_package_contract.py" in workflow
    assert "Assemble React Native package for Android smoke" in workflow
    assert "Run Expo Android native build smoke" in workflow


def test_premerge_expo_smoke_installs_the_packed_sdk() -> None:
    """Keep CMake's Node resolution identical to a consumer installation."""
    workflow = BINDINGS_WORKFLOW.read_text(encoding="utf-8")

    assert 'tarball="$(npm pack .. --pack-destination . --silent)"' in workflow
    assert 'npm pkg set "dependencies.edge-intelligence-sdk=file:./$tarball"' in workflow
    assert "bash ../../scripts/retry-command.sh npm ci" in workflow


def test_ubrn_cmake_patch_uses_the_exported_runtime_entrypoint() -> None:
    patch = load_ubrn_cmake_patch()
    generated = """execute_process(
    COMMAND node -p \"require.resolve('uniffi-bindgen-react-native/package.json')\"
)
"""

    with tempfile.TemporaryDirectory() as directory:
        cmake = Path(directory) / "CMakeLists.txt"
        cmake.write_text(generated, encoding="utf-8")
        patch.patch_cmake(cmake)
        patch.patch_cmake(cmake)
        patched = cmake.read_text(encoding="utf-8")

    assert "require.resolve('uniffi-bindgen-react-native/package.json')" not in patched
    assert "require.resolve('uniffi-bindgen-react-native')" in patched
    assert "'../../../..'" in patched

    for workflow in (BINDINGS_WORKFLOW, RELEASE_WORKFLOW):
        assert "patch-ubrn-cmake.py android/CMakeLists.txt" in workflow.read_text(
            encoding="utf-8"
        )


def run_contract_tests(namespace=None) -> int:
    available = globals() if namespace is None else namespace
    tests = sorted(
        (name, candidate)
        for name, candidate in available.items()
        if name.startswith("test_") and callable(candidate)
    )
    for _, test in tests:
        test()
    return len(tests)


if __name__ == "__main__":
    count = run_contract_tests()
    print(f"OK: {count} React Native package contract tests passed")
