#!/usr/bin/env python3
"""Fail when a 64-bit Android ELF has a LOAD segment below 16 KiB alignment."""

from pathlib import Path
import subprocess
import sys


MIN_ALIGNMENT = 16 * 1024


def load_alignments(readelf: Path, library: Path) -> list[int]:
    output = subprocess.check_output(
        [str(readelf), "-lW", str(library)],
        text=True,
    )
    return [
        int(line.split()[-1], 16)
        for line in output.splitlines()
        if line.lstrip().startswith("LOAD")
    ]


def main(arguments: list[str]) -> int:
    if len(arguments) < 2:
        print(
            "usage: verify-android-page-alignment.py <llvm-readelf> <library>...",
            file=sys.stderr,
        )
        return 2

    readelf = Path(arguments[0])
    errors: list[str] = []
    for library_arg in arguments[1:]:
        library = Path(library_arg)
        alignments = load_alignments(readelf, library)
        if not alignments:
            errors.append(f"{library}: no ELF LOAD segments found")
            continue
        minimum = min(alignments)
        print(f"{library}: minimum LOAD alignment = {minimum} bytes")
        if minimum < MIN_ALIGNMENT:
            errors.append(
                f"{library}: {minimum}-byte LOAD alignment is below 16 KiB"
            )

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
