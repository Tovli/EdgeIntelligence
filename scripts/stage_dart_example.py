#!/usr/bin/env python3
"""Stage a Dart example without copying package-ignored build residue."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


EXAMPLE_SUBTREE = PurePosixPath("example")


def _run_git(*args: str, allow_failure: bool = False) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        if allow_failure:
            return subprocess.CompletedProcess(["git", *args], 127, b"", b"")
        raise RuntimeError("git is required while staging from a Git work tree")

    if not allow_failure and completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return completed


def _is_git_work_tree(source_root: Path) -> bool:
    completed = _run_git(
        "-C",
        str(source_root),
        "rev-parse",
        "--is-inside-work-tree",
        allow_failure=True,
    )
    return completed.returncode == 0 and completed.stdout.strip() == b"true"


def _decode_paths(output: bytes) -> set[PurePosixPath]:
    paths: set[PurePosixPath] = set()
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        decoded_path = os.fsdecode(raw_path)
        raw_parts = decoded_path.split("/")
        relative_path = PurePosixPath(decoded_path)
        if (
            any(part in {"", ".", ".."} for part in raw_parts)
            or relative_path.is_absolute()
            or relative_path.parts[0] != EXAMPLE_SUBTREE.name
        ):
            raise RuntimeError(f"git returned unsafe example path: {relative_path}")
        paths.add(relative_path)
    return paths


def _git_selected_paths(source_root: Path) -> set[PurePosixPath]:
    tracked = _run_git(
        "-C",
        str(source_root),
        "ls-files",
        "-z",
        "--cached",
        "--",
        EXAMPLE_SUBTREE.as_posix(),
    )

    with tempfile.TemporaryDirectory(prefix="dart-example-git-") as git_directory:
        _run_git("init", "--bare", "--quiet", git_directory)
        untracked = _run_git(
            f"--git-dir={git_directory}",
            f"--work-tree={source_root}",
            "ls-files",
            "-z",
            "--others",
            "--exclude-per-directory=.gitignore",
            "--",
            EXAMPLE_SUBTREE.as_posix(),
        )

    selected = _decode_paths(tracked.stdout) | _decode_paths(untracked.stdout)
    return {
        relative_path
        for relative_path in selected
        if (source_root.joinpath(*relative_path.parts).exists())
        or source_root.joinpath(*relative_path.parts).is_symlink()
    }


def _copy_selected(
    source_root: Path,
    destination_root: Path,
    selected_paths: set[PurePosixPath],
) -> None:
    destination_example = destination_root / EXAMPLE_SUBTREE.name
    destination_example.mkdir(parents=True)

    for relative_path in sorted(selected_paths, key=lambda path: path.as_posix()):
        source_path = source_root.joinpath(*relative_path.parts)
        destination_path = destination_root.joinpath(*relative_path.parts)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir() and not source_path.is_symlink():
            destination_path.mkdir(exist_ok=True)
        elif source_path.is_file() or source_path.is_symlink():
            shutil.copy2(source_path, destination_path, follow_symlinks=False)
        else:
            raise RuntimeError(f"unsupported example source path: {source_path}")


def stage_dart_example(source_root: Path, destination_root: Path) -> None:
    source_root = source_root.resolve(strict=True)
    destination_root = destination_root.resolve(strict=False)
    source_example = source_root / EXAMPLE_SUBTREE.name
    destination_example = destination_root / EXAMPLE_SUBTREE.name

    if not source_example.is_dir():
        raise RuntimeError(f"missing Dart example directory: {source_example}")
    if (
        source_root == destination_root
        or source_root in destination_root.parents
        or destination_root in source_root.parents
    ):
        raise RuntimeError("source and destination directories must not overlap")
    if destination_example.exists() or destination_example.is_symlink():
        raise RuntimeError(f"destination already exists: {destination_example}")

    destination_root.mkdir(parents=True, exist_ok=True)
    if _is_git_work_tree(source_root):
        _copy_selected(source_root, destination_root, _git_selected_paths(source_root))
    else:
        shutil.copytree(source_example, destination_example, symlinks=True)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} SOURCE_DIR DEST_DIR", file=sys.stderr)
        return 2

    try:
        stage_dart_example(Path(argv[1]), Path(argv[2]))
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
