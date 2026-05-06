#!/usr/bin/env python3
"""Rename TabPFN precomputed JSON files.

Renames files of the form:
  game_instance_<number>_tabpfn_precomputed.json
into:
  model_name=tabpfn_imputer=tabpfn_<number>.json

By default, only numbers 0..29 (inclusive) are considered.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


SOURCE_RE = re.compile(r"^game_instance_(?P<id>\d+)_tabpfn_precomputed\.json$")


@dataclass(frozen=True)
class RenameAction:
    src: Path
    dst: Path


def _iter_source_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("game_instance_*_tabpfn_precomputed.json"))
    return sorted(root.glob("game_instance_*_tabpfn_precomputed.json"))


def _build_actions(
    root: Path,
    recursive: bool,
    min_id: int,
    max_id: int,
) -> tuple[list[RenameAction], list[str]]:
    actions: list[RenameAction] = []
    problems: list[str] = []

    for src in _iter_source_files(root, recursive=recursive):
        if not src.is_file():
            continue

        match = SOURCE_RE.match(src.name)
        if not match:
            continue

        file_id = int(match.group("id"))
        if not (min_id <= file_id <= max_id):
            continue

        dst = src.with_name(f"model_name=tabpfn_imputer=tabpfn_{file_id}.json")
        actions.append(RenameAction(src=src, dst=dst))

    if not actions:
        problems.append(
            "No matching files found. Looked for 'game_instance_<id>_tabpfn_precomputed.json'."
        )

    return actions, problems


def _apply_actions(actions: list[RenameAction], dry_run: bool, overwrite: bool) -> int:
    had_errors = False

    for action in actions:
        if action.src.resolve() == action.dst.resolve():
            continue

        if action.dst.exists() and not overwrite:
            print(f"SKIP (exists): {action.src} -> {action.dst}")
            had_errors = True
            continue

        if dry_run:
            print(f"DRY-RUN: {action.src} -> {action.dst}")
            continue

        try:
            if overwrite:
                action.src.replace(action.dst)
            else:
                action.src.rename(action.dst)
            print(f"RENAMED: {action.src} -> {action.dst}")
        except OSError as exc:
            print(f"ERROR: {action.src} -> {action.dst}: {exc}", file=sys.stderr)
            had_errors = True

    return 1 if had_errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename 'game_instance_<id>_tabpfn_precomputed.json' files to "
            "'model_name=tabpfn_imputer=tabpfn_<id>.json'."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Directory to scan for files (default: current directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under the given path.",
    )
    parser.add_argument(
        "--min-id",
        type=int,
        default=0,
        help="Minimum <id> to rename (default: 0).",
    )
    parser.add_argument(
        "--max-id",
        type=int,
        default=29,
        help="Maximum <id> to rename (default: 29).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination if it already exists.",
    )

    args = parser.parse_args(argv)

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: path does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"ERROR: path is not a directory: {root}", file=sys.stderr)
        return 2

    actions, problems = _build_actions(
        root=root,
        recursive=args.recursive,
        min_id=args.min_id,
        max_id=args.max_id,
    )
    for problem in problems:
        print(problem, file=sys.stderr)

    if not actions:
        return 1

    return _apply_actions(actions, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
