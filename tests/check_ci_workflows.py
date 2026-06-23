#!/usr/bin/env python3
"""Validate CI dependencies that are easy to accidentally leave implicit."""

from pathlib import Path
import sys


WORKFLOWS = [
    Path(".github/workflows/release.yml"),
    Path(".github/workflows/bindings.yml"),
]

RN_RETRY_COMMAND = "bash scripts/retry-command.sh make codegen-rn"
WASM_RETRY_COMMAND = "bash scripts/retry-command.sh curl -fsSL"
BINDINGS_UPLOAD_IF = "if: github.event_name != 'pull_request'"

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
    codegen_pos = text.find("make codegen-flutter")
    rn_codegen_pos = text.find("make codegen-rn")

    if codegen_pos < 0:
        return [f"{path}: missing make codegen-flutter step"]

    for label, needle in REQUIRED_INSTALLS:
        install_pos = text.find(needle)
        if install_pos < 0:
            errors.append(f"{path}: missing explicit {label} install before Flutter codegen")
        elif install_pos > codegen_pos:
            errors.append(f"{path}: installs {label} after Flutter codegen")

    if rn_codegen_pos >= 0 and RN_RETRY_COMMAND not in text:
        errors.append(f"{path}: React Native codegen must run through retry wrapper")

    if "Install wasm-pack" in text and WASM_RETRY_COMMAND not in text:
        errors.append(f"{path}: wasm-pack download must run through retry wrapper")

    if path.name == "bindings.yml":
        step_blocks = text.split("\n      - ")
        for block in step_blocks:
            if "uses: actions/upload-artifact@v4" in block and BINDINGS_UPLOAD_IF not in block:
                errors.append(
                    f"{path}: PR validation artifact uploads must be gated with "
                    f"{BINDINGS_UPLOAD_IF!r}"
                )

    return errors


def main() -> int:
    errors: list[str] = []
    for workflow in WORKFLOWS:
        errors.extend(check_workflow(workflow))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print("OK: CI workflows guard Flutter tools and retry external codegen downloads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
