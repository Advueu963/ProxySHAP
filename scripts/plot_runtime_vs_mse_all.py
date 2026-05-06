#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run scripts/plot_runtime_vs_mse.py for each benchmark CSV and write one plot per CSV. "
            "Output file names mirror the CSV base names."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory to search for CSV files (default: repository root).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="results_benchmark_*.csv",
        help="Glob used to find benchmark CSV files (default: results_benchmark_*.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/runtime_vs_mse"),
        help="Directory where PNGs are written (default: plots/runtime_vs_mse).",
    )
    parser.add_argument(
        "--logx",
        action="store_true",
        help="Pass --logx to the plotting script.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Pass --logy to the plotting script.",
    )
    parser.add_argument(
        "--mse-floor",
        type=str,
        default=None,
        help="Pass --mse-floor=<value> to the plotting script (optional).",
    )
    parser.add_argument(
        "--budget-column",
        type=str,
        default="used_budget",
        help="Pass --budget-column=<col> to the plotting script (default: used_budget).",
    )
    parser.add_argument(
        "--budget-quantile",
        type=str,
        default=None,
        help="Pass --budget-quantile=<p> to the plotting script (optional).",
    )
    parser.add_argument(
        "--no-hull",
        action="store_true",
        help="Pass --no-hull to the plotting script.",
    )
    parser.add_argument(
        "--hull-alpha",
        type=str,
        default=None,
        help="Pass --hull-alpha=<value> to the plotting script (optional).",
    )
    parser.add_argument(
        "--hull-line-alpha",
        type=str,
        default=None,
        help="Pass --hull-line-alpha=<value> to the plotting script (optional).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--without-game-call",
        action="store_true",
        help="Subtract 'evaluations' from runtime when computing total runtime.",
    )
    parser.add_argument(
        "--highlight-topk-players",
        type=str,
        default=None,
        help="Pass --highlight-topk-players=<value> to the plotting script (optional).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    csv_files = sorted(args.input_dir.glob(args.glob))
    if not csv_files:
        raise SystemExit(
            f"No CSV files found in {args.input_dir.resolve()} matching {args.glob!r}."
        )

    script_path = Path(__file__).with_name("plot_runtime_vs_mse.py")
    if not script_path.exists():
        raise SystemExit(f"Missing plotting script: {script_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    for csv_path in csv_files:
        output_path = args.output_dir / f"{csv_path.stem}.png"

        cmd: List[str] = [
            sys.executable,
            str(script_path),
            str(csv_path),
            "--output",
            str(output_path),
        ]

        if args.logx:
            cmd.append("--logx")
        if args.logy:
            cmd.append("--logy")
        if args.mse_floor is not None:
            cmd.append(f"--mse-floor={args.mse_floor}")
        if args.budget_column:
            cmd.append(f"--budget-column={args.budget_column}")
        if args.budget_quantile is not None:
            cmd.append(f"--budget-quantile={args.budget_quantile}")
        if args.no_hull:
            cmd.append("--no-hull")
        if args.hull_alpha is not None:
            cmd.append(f"--hull-alpha={args.hull_alpha}")
        if args.hull_line_alpha is not None:
            cmd.append(f"--hull-line-alpha={args.hull_line_alpha}")
        if args.highlight_topk_players is not None:
            cmd.append(f"--highlight-topk-players={args.highlight_topk_players}")
            cmd.append(f"--highlight-alpha=1")
            cmd.append(f"--other-alpha=0.1")
        if args.without_game_call:
            cmd.append(f"--without-game-call")

        if args.dry_run:
            print(" ".join(cmd))
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            failures.append(str(csv_path))
            sys.stderr.write(f"[FAIL] {csv_path.name}\n")
            if result.stdout:
                sys.stderr.write(result.stdout + "\n")
            if result.stderr:
                sys.stderr.write(result.stderr + "\n")
        else:
            print(f"[OK] {csv_path.name} -> {output_path}")

    if failures:
        raise SystemExit(
            f"Failed for {len(failures)} file(s): {', '.join(Path(f).name for f in failures)}"
        )


if __name__ == "__main__":
    main()
