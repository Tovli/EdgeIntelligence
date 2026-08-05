"""Static release-contract checks for the Flutter/Dart package."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DART_IOS = ROOT / "packaging" / "dart" / "ios"
MAKEFILE = ROOT / "Makefile"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
BINDINGS_WORKFLOW = ROOT / ".github" / "workflows" / "bindings.yml"
STAGE_DART_EXAMPLE = ROOT / "scripts" / "stage_dart_example.py"


def _workflow_job_body(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\s*\n"
        rf"(.*?)(?=^  [A-Za-z0-9_-]+:\s*$|\Z)",
        workflow,
    )
    assert match is not None, f"missing workflow job {job_name!r}"
    return match.group(1)


def _step_blocks(job: str) -> list[str]:
    return re.findall(r"(?ms)^      - (.*?)(?=^      - |\Z)", job)


def _make_target_body(makefile: str, target_name: str) -> str:
    match = re.search(
        rf"(?ms)^{re.escape(target_name)}:[^\n]*\n"
        rf"(.*?)(?=^[A-Za-z0-9_.%-]+:[^\n]*$|\Z)",
        makefile,
    )
    assert match is not None, f"missing Make target {target_name!r}"
    return match.group(1)


def _required_step(
    steps: list[str], description: str, predicate: Callable[[str], bool]
) -> str:
    step = next((step for step in steps if predicate(step)), None)
    assert step is not None, f"missing workflow step {description!r}"
    return step


def _required_step_index(
    steps: list[str], description: str, predicate: Callable[[str], bool]
) -> int:
    matches = [
        index for index, step in enumerate(steps) if predicate(step)
    ]
    assert len(matches) == 1, (
        f"expected exactly one workflow step {description!r}, "
        f"found {len(matches)}"
    )
    return matches[0]


def _write_file(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _run_command(*args: str, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {' '.join(args)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _stage_dart_example(source: Path, destination: Path) -> None:
    _run_command(
        sys.executable,
        str(STAGE_DART_EXAMPLE),
        str(source),
        str(destination),
    )


def _artifact_upload_step(job: str, artifact_name: str) -> str:
    return _required_step(
        _step_blocks(job),
        f"upload artifact {artifact_name}",
        lambda step: "uses: actions/upload-artifact@v4" in step
        and re.search(
            rf"(?m)^\s+name:\s*{re.escape(artifact_name)}\s*$", step
        )
        is not None,
    )


def _assert_hidden_artifact_upload(
    job: str, artifact_name: str, expected_path: str
) -> None:
    upload = _artifact_upload_step(job, artifact_name)
    assert re.search(
        rf"(?m)^\s+path:\s*{re.escape(expected_path)}/?\s*$", upload
    ), f"artifact {artifact_name!r} does not upload {expected_path!r}"
    assert re.search(
        r"(?m)^\s+include-hidden-files:\s*true\s*$", upload
    ), f"artifact {artifact_name!r} drops package ignore metadata"


def test_flutter_ios_binary_target_is_assembled_inside_swift_package() -> None:
    package_swift = (DART_IOS / "edge_intelligence" / "Package.swift").read_text(
        encoding="utf-8"
    )
    binary_target = re.search(
        r'\.binaryTarget\(.*?path:\s*"([^"]+)"', package_swift, re.DOTALL
    )
    assert binary_target is not None
    binary_path = PurePosixPath(binary_target.group(1))
    assert binary_path == PurePosixPath("Frameworks/el_ffi.xcframework")

    podspec = (DART_IOS / "edge_intelligence.podspec").read_text(encoding="utf-8")
    assert (
        "s.vendored_frameworks = "
        "'edge_intelligence/Frameworks/el_ffi.xcframework'"
    ) in podspec

    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert "path: assembly/ios/edge_intelligence/Frameworks/" in workflow
    assert (
        "assembly/ios/edge_intelligence/Frameworks/el_ffi.xcframework"
        in workflow
    )
    assert "test -d ios/edge_intelligence/Frameworks/el_ffi.xcframework" in workflow
    embedded_framework = (
        "test -f "
        "build/ios/iphonesimulator/Runner.app/Frameworks/el_ffi.framework/el_ffi"
    )
    assert workflow.count(embedded_framework) >= 2
    cocoapods = workflow.index("flutter config --no-enable-swift-package-manager")
    swiftpm = workflow.index("flutter config --enable-swift-package-manager")
    assert cocoapods < swiftpm


def test_dart_contract_runs_in_bindings_and_release_ci() -> None:
    bindings = BINDINGS_WORKFLOW.read_text(encoding="utf-8")
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    contract = "tests/test_dart_package_contract.py"

    assert bindings.count(f"'{contract}'") == 2
    assert f"python3 {contract}" in bindings
    assert f"python3 {contract}" in release


def test_codegen_sources_and_artifact_handoffs_retain_package_ignore_metadata() -> None:
    makefile = MAKEFILE.read_text(encoding="utf-8")
    codegen_dart = _make_target_body(makefile, "codegen-dart")
    release = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    bindings = BINDINGS_WORKFLOW.read_text(encoding="utf-8")
    required_ignore_metadata = (
        ".gitignore",
        "example/.gitignore",
        "example/android/.gitignore",
        "example/ios/.gitignore",
    )
    for relative_path in required_ignore_metadata:
        source = ROOT / "packaging" / "dart" / relative_path
        assert source.is_file(), f"missing Dart package ignore file {relative_path!r}"

    assert "cp packaging/dart/.gitignore $(OUT)/dart/.gitignore" in codegen_dart
    assert "scripts/stage_dart_example.py" in codegen_dart, (
        "codegen-dart does not use the behavior-tested example stager"
    )

    _assert_hidden_artifact_upload(
        _workflow_job_body(release, "build-dart"), "dart-bindings", "out/dart"
    )
    _assert_hidden_artifact_upload(
        _workflow_job_body(bindings, "dart"), "dart-bindings", "out/dart"
    )
    _assert_hidden_artifact_upload(
        _workflow_job_body(release, "assemble-dart"), "pub-package", "assembly"
    )


def test_dart_example_stager_uses_only_package_gitignore_rules() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        repository = temporary / "repository"
        source = repository / "packaging" / "dart"
        destination = temporary / "output"
        source.mkdir(parents=True)
        _run_command("git", "init", "--quiet", str(repository))

        fixture_files = {
            repository / ".gitignore": (
                "/packaging/dart/example/outer-only.txt\n"
            ),
            source / ".gitignore": "build/\n",
            source / "example" / ".gitignore": "# retained package metadata\n",
            source / "example" / "android" / ".gitignore": "/gradlew\n",
            source / "example" / "ios" / ".gitignore": "# retained iOS metadata\n",
            source / "example" / "tracked.txt": "tracked\n",
            source / "example" / "deleted.txt": "delete after staging in Git\n",
            source / "example" / "untracked.txt": "untracked\n",
            source / "example" / "build" / "generated.bin": "build residue\n",
            source / "example" / "build" / "tracked.bin": (
                "force-tracked build input\n"
            ),
            source / "example" / "android" / "gradlew": "Gradle residue\n",
            source / "example" / "outer-only.txt": "keep ancestor-ignored file\n",
            source / "example" / "global-only.txt": "keep global-ignored file\n",
            source / "example" / "info-only.txt": "keep info-ignored file\n",
        }
        for path, contents in fixture_files.items():
            _write_file(path, contents)

        global_excludes = temporary / "global-excludes"
        _write_file(global_excludes, "global-only.txt\n")
        _write_file(repository / ".git" / "info" / "exclude", "info-only.txt\n")
        _run_command(
            "git",
            "-C",
            str(repository),
            "config",
            "core.excludesFile",
            str(global_excludes),
        )
        _run_command(
            "git",
            "-C",
            str(repository),
            "add",
            "--",
            ".gitignore",
            "packaging/dart/.gitignore",
            "packaging/dart/example/.gitignore",
            "packaging/dart/example/android/.gitignore",
            "packaging/dart/example/ios/.gitignore",
            "packaging/dart/example/tracked.txt",
            "packaging/dart/example/deleted.txt",
        )
        _run_command(
            "git",
            "-C",
            str(repository),
            "add",
            "--force",
            "--",
            "packaging/dart/example/build/tracked.bin",
        )
        (source / "example" / "deleted.txt").unlink()

        _stage_dart_example(source, destination)

        expected_files = {
            "example/.gitignore": "# retained package metadata\n",
            "example/android/.gitignore": "/gradlew\n",
            "example/ios/.gitignore": "# retained iOS metadata\n",
            "example/tracked.txt": "tracked\n",
            "example/build/tracked.bin": "force-tracked build input\n",
            "example/untracked.txt": "untracked\n",
            "example/outer-only.txt": "keep ancestor-ignored file\n",
            "example/global-only.txt": "keep global-ignored file\n",
            "example/info-only.txt": "keep info-ignored file\n",
        }
        for relative_path, contents in expected_files.items():
            staged = destination / relative_path
            assert staged.read_text(encoding="utf-8") == contents

        excluded_files = (
            "example/build/generated.bin",
            "example/android/gradlew",
            "example/deleted.txt",
        )
        for relative_path in excluded_files:
            assert not (destination / relative_path).exists(), (
                f"unexpected staged file {relative_path!r}"
            )


def test_dart_example_stager_falls_back_to_recursive_copy_without_git() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        source = temporary / "dart-package"
        destination = temporary / "output"
        fixture_files = {
            "example/.gitignore": "build/\n",
            "example/main.dart": "void main() {}\n",
            "example/build/local-output": "copied without Git\n",
            "example/android/gradlew": "copied without Git\n",
        }
        for relative_path, contents in fixture_files.items():
            _write_file(source / relative_path, contents)

        _stage_dart_example(source, destination)

        for relative_path, contents in fixture_files.items():
            staged = destination / relative_path
            assert staged.read_text(encoding="utf-8") == contents


def test_pub_validation_isolated_from_uploaded_assembly() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    build_dart = _workflow_job_body(workflow, "build-dart")
    assemble_dart = _workflow_job_body(workflow, "assemble-dart")
    steps = _step_blocks(assemble_dart)

    build_steps = _step_blocks(build_dart)
    dart_validation = _required_step_index(
        build_steps,
        "stage build-dart validation workspace",
        lambda step: "name: Stage Dart validation workspace" in step
        and "cp -R out/dart validation" in step,
    )
    dependency_resolution = _required_step_index(
        build_steps,
        "resolve build-dart dependencies",
        lambda step: "name: Resolve Flutter plugin dependencies" in step,
    )
    dart_upload = _required_step_index(
        build_steps,
        "upload dart-bindings artifact",
        lambda step: "uses: actions/upload-artifact@v4" in step
        and re.search(r"(?m)^\s+name:\s*dart-bindings\s*$", step) is not None,
    )
    assert dart_validation < dependency_resolution < dart_upload
    assert re.search(
        r"(?m)^\s+working-directory:\s*validation/?\s*$",
        build_steps[dependency_resolution],
    )
    assert re.search(r"(?m)^\s+path:\s*out/dart/?\s*$", build_steps[dart_upload])

    native_artifact_check = _required_step_index(
        steps,
        "validate native library artifacts",
        lambda step: "name: Assert native library artifacts are non-empty" in step,
    )
    size_gate = _required_step_index(
        steps,
        "check pub.dev package size",
        lambda step: "name: Check pub.dev package size" in step,
    )
    flutter_setup = _required_step_index(
        steps,
        "set up Flutter",
        lambda step: "uses: subosito/flutter-action@v2" in step,
    )
    validation = _required_step_index(
        steps,
        "stage pub.dev validation workspace",
        lambda step: "name: Stage pub.dev validation workspace" in step
        and "cp -R assembly validation" in step,
    )
    smoke = _required_step_index(
        steps,
        "run Flutter Android package smoke build",
        lambda step: "name: Run Flutter Android package smoke build" in step,
    )
    dry_run_staging = _required_step_index(
        steps,
        "stage pristine pub.dev dry-run workspace",
        lambda step: "name: Stage pub.dev dry-run workspace" in step
        and "cp -R assembly dry-run" in step,
    )
    dry_run_dependency_resolution = _required_step_index(
        steps,
        "resolve pristine pub.dev dry-run dependencies",
        lambda step: "flutter pub get --no-example" in step
        and re.search(
            r"(?m)^\s+working-directory:\s*dry-run/?\s*$", step
        )
        is not None,
    )
    dry_run = _required_step_index(
        steps,
        "run flutter pub publish dry-run",
        lambda step: "flutter pub publish --dry-run" in step,
    )
    upload = _required_step_index(
        steps,
        "upload pub-package artifact",
        lambda step: "uses: actions/upload-artifact@v4" in step
        and re.search(r"(?m)^\s+name:\s*pub-package\s*$", step) is not None,
    )
    mutable_validation_steps = (
        "Resolve Flutter plugin dependencies",
        "Run Dart package runtime smoke tests",
        "Run Flutter example tests",
        "Run Flutter Android package smoke build",
    )
    for step_name in mutable_validation_steps:
        step = _required_step(
            steps,
            step_name,
            lambda candidate, name=step_name: f"name: {name}" in candidate,
        )
        assert re.search(
            r"(?m)^\s+working-directory:\s*validation(?:/example)?/?\s*$", step
        ), f"{step_name} mutates the publish assembly"

    assert re.search(
        r"(?m)^\s+working-directory:\s*assembly/?\s*$", steps[size_gate]
    )
    assert re.search(
        r"(?m)^\s+working-directory:\s*dry-run/?\s*$", steps[dry_run]
    )
    assert re.search(r"(?m)^\s+path:\s*assembly/?\s*$", steps[upload])
    assert native_artifact_check < size_gate < flutter_setup < validation
    assert (
        validation
        < smoke
        < dry_run_staging
        < dry_run_dependency_resolution
        < dry_run
        < upload
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
    print(f"OK: {count} Dart package contract tests passed")
