#!/usr/bin/env python3

from __future__ import annotations
import re
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked runtime bars per dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing benchmark CSV files.",
        default=Path("."),
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="results_benchmark_*.csv",
        help="Glob used to find benchmark CSV files (default: results_benchmark_*.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/runtime_vs_budget"),
        help="Destination for the generated figure.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use logarithmic scale for runtimes.",
    )
    args = parser.parse_args()

    return args
def main() -> None:
    args = _parse_args()

    csv_files = sorted(args.input_dir.glob(args.glob))
    if not csv_files:
        raise SystemExit(
            f"No CSV files found in {args.input_dir.resolve()} matching {args.glob!r}."
        )

    script_path = Path(__file__).with_name("plot_runtime_vs_budget.py")
    if not script_path.exists():
        raise SystemExit(f"Missing plotting script: {script_path}")


    failures: List[str] = []
    for csv_path in csv_files:
        regex = r'results_benchmark_(.+)_(.+)_(.+).csv'
        result = re.match(regex, str(csv_path.name))
        if result:
            index = result.group(1)
            order = result.group(2)
            game_type = result.group(3)
            output_path = args.output / (game_type +"_" + index + "_" + order) 
        else:
            output_path = args.output 

        cmd: List[str] = [
            sys.executable,
            str(script_path),
            "--input",
            str(csv_path),
            "--output-dir",
            str(output_path),
        ]

        if args.logy:
            cmd.append("--logy")

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
