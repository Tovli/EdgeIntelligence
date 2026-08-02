#!/usr/bin/env python3
"""Patch UBRN 0.31 Android CMake for Node's package ``exports`` rules."""

from __future__ import annotations

from pathlib import Path
import sys


UBRN_PACKAGE_JSON_LOOKUP = (
    "require.resolve('uniffi-bindgen-react-native/package.json')"
)
EXPORTED_ENTRYPOINT_LOOKUP = (
    "require('path').join("
    "require('path').resolve("
    "require.resolve('uniffi-bindgen-react-native'), '../../../..'"
    "), 'package.json')"
)


def patch_cmake(path: Path) -> None:
    """Replace UBRN's non-exported package.json lookup exactly once."""
    content = path.read_text(encoding="utf-8")
    if EXPORTED_ENTRYPOINT_LOOKUP in content:
        return
    if content.count(UBRN_PACKAGE_JSON_LOOKUP) != 1:
        raise ValueError(
            f"{path}: expected exactly one UBRN package.json lookup to patch"
        )
    path.write_text(
        content.replace(UBRN_PACKAGE_JSON_LOOKUP, EXPORTED_ENTRYPOINT_LOOKUP),
        encoding="utf-8",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} ANDROID_CMAKE_LISTS")
    patch_cmake(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
