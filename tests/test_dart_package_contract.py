"""Static release-contract checks for the Flutter/Dart package."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
DART_IOS = ROOT / "packaging" / "dart" / "ios"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
BINDINGS_WORKFLOW = ROOT / ".github" / "workflows" / "bindings.yml"


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
