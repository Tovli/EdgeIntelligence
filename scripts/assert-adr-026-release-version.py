#!/usr/bin/env python3
"""Reject release tags that would ship the ADR-026 runtime migration as a patch."""

from __future__ import annotations

import re
import sys


MINIMUM_VERSION = (0, 4, 0)
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def parse_version(value: str) -> tuple[int, int, int]:
    match = SEMVER.fullmatch(value)
    if match is None:
        raise ValueError(f"expected a release version in MAJOR.MINOR.PATCH form, got {value!r}")
    return tuple(int(part) for part in match.groups())


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <MAJOR.MINOR.PATCH>", file=sys.stderr)
        return 2

    try:
        version = parse_version(argv[1])
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    if version < MINIMUM_VERSION:
        minimum = ".".join(map(str, MINIMUM_VERSION))
        print(
            "ERROR: ADR-026 changes localEdgeLlm(modelUri) runtime behavior; "
            f"{argv[1]} is below the required {minimum} minor release.",
            file=sys.stderr,
        )
        print(
            f"Run: cargo set-version --workspace {minimum} "
            "(or a later compatible version) before tagging.",
            file=sys.stderr,
        )
        return 1

    print(f"OK: ADR-026 migration release {argv[1]} is at or above 0.4.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
