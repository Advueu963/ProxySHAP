"""Generate LaTeX figures for runtime-vs-budget PNGs.

Expected directory layout:
  figures/runtime_vs_budget/<game_type>_<index>_<order>/*.png

Where:
  game_type: exhaustive | interventional | pathdependent | tabpfn | huge | ...
  index:     BV | BII | SV | SII
  order:     integer (e.g., 1, 2, 3)

Output:
  - For each game_type, generate:
      * one (or more, if too many subfigures) figure(s) for BV + BII_* ("banzhaf")
      * one (or more) figure(s) for SV + SII_* ("shapley")

Notes:
  - Captions drop the "LocalXAI" suffix everywhere.
  - To avoid LaTeX "Counter too large" (subfigure uses a,b,c,... limited to 26),
    figures are automatically split so each has <= --max-subfigures subfigures.

Usage:
  python3 generate_runtime_vs_budget_figures.py \
    --root figures/runtime_vs_budget \
    --output generated/latex/runtime_vs_budget

Then in LaTeX:
  \input{generated/latex/runtime_vs_budget/runtime_vs_budget_banzhaf_exhaustive.tex}
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


FOLDER_RE = re.compile(r"^(?P<game>[^_]+)_(?P<index>BV|BII|SV|SII)_(?P<order>\d+)$")


@dataclass(frozen=True)
class FolderSpec:
    game_type: str
    index: str
    order: int
    path: Path


def discover_specs(root: Path) -> list[FolderSpec]:
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    specs: list[FolderSpec] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        m = FOLDER_RE.match(child.name)
        if not m:
            continue
        specs.append(
            FolderSpec(
                game_type=m.group("game"),
                index=m.group("index"),
                order=int(m.group("order")),
                path=child,
            )
        )
    return specs


def group_for_index(index: str) -> str:
    if index in {"BV", "BII"}:
        return "banzhaf"
    if index in {"SV", "SII"}:
        return "shapley"
    raise ValueError(index)


def spec_sort_key(spec: FolderSpec) -> tuple[int, int]:
    # Value first, then interactions by order.
    if spec.index in {"BV", "SV"}:
        return (0, spec.order)
    return (1, spec.order)


def dataset_label_from_png(png: Path) -> str:
    name = png.stem
    name = name.replace("_runtime_vs_budget", "")

    # Drop LocalXAI everywhere.
    if name.endswith("LocalXAI"):
        name = name[: -len("LocalXAI")]

    # Make one concatenated name more readable.
    if name.endswith("Superpixel") and not name.endswith(" Superpixel"):
        name = name[: -len("Superpixel")] + " Superpixel"

    return name


def latex_escape(text: str) -> str:
    # Minimal escaping for captions.
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def rel_figures_path(png: Path) -> str:
    rel = png.as_posix()
    if "plots/" in rel:
        rel = "figures/" + rel.split("plots/", 1)[1]
    return rel


def iter_runtime_budget_pngs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*_runtime_vs_budget.png") if p.is_file())


def chunked(
    items: list[tuple[str, str, str]], max_items: int
) -> Iterator[list[tuple[str, str, str]]]:
    for i in range(0, len(items), max_items):
        yield items[i : i + max_items]


def build_subfig_entries(
    specs: Iterable[FolderSpec],
    *,
    game_title: str,
) -> list[tuple[str, str, str]]:
    """Return list of (comment_header, include_path, caption)."""

    entries: list[tuple[str, str, str]] = []

    for spec in sorted(specs, key=spec_sort_key):
        pngs = iter_runtime_budget_pngs(spec.path)
        if not pngs:
            continue

        if spec.index in {"BV", "SV"}:
            header = f"% {spec.index}"
            idx_label = spec.index
        else:
            header = f"% {spec.index} order {spec.order}"
            idx_label = f"{spec.index} {spec.order}"

        for png in pngs:
            dataset = dataset_label_from_png(png)
            caption = latex_escape(f"({game_title}) {dataset} {idx_label}")
            entries.append((header, rel_figures_path(png), caption))

    return entries


def render_figure_block(
    *,
    float_spec: str,
    width: str,
    label: str,
    caption: str,
    entries: list[tuple[str, str, str]],
) -> str:
    lines: list[str] = []
    lines.append(f"\\begin{{figure}}[{float_spec}]")
    lines.append("    \\centering")

    last_header: str | None = None
    for header, include_path, subcap in entries:
        if header != last_header:
            lines.append(f"    {header}")
            last_header = header

        lines.append(f"    \\begin{{subfigure}}[b]{{{width}}}")
        lines.append(f"        \\includegraphics[width=\\textwidth]{{{include_path}}}")
        lines.append(f"        \\caption{{{subcap}}}")
        lines.append("    \\end{subfigure}")

    lines.append(f"    \\label{{{label}}}")
    lines.append("    \\caption{")
    lines.append(f"     {latex_escape(caption)}")
    lines.append("    }")
    lines.append("\\end{figure}")
    return "\n".join(lines) + "\n"


def title_case_game(game_type: str) -> str:
    # preserve mixed casing (e.g., TabPFN if ever present); otherwise capitalize first char.
    if any(c.isupper() for c in game_type[1:]):
        return game_type
    return game_type[:1].upper() + game_type[1:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("plots/runtime_vs_budget"))
    ap.add_argument(
        "--output", type=Path, default=Path("generated/latex/runtime_vs_budget")
    )
    ap.add_argument("--float", dest="float_spec", default="h")
    ap.add_argument("--width", default="0.25\\textwidth")
    ap.add_argument(
        "--max-subfigures",
        type=int,
        default=26,
        help="Max subfigures per figure block to avoid LaTeX counter overflow (default: 26).",
    )
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    specs = discover_specs(args.root)
    by_game: dict[str, list[FolderSpec]] = {}
    for s in specs:
        by_game.setdefault(s.game_type, []).append(s)

    outputs: list[tuple[str, str]] = []  # (filename, content)

    for game_type, game_specs in sorted(by_game.items()):
        game_title = title_case_game(game_type)

        for group in ("banzhaf", "shapley"):
            group_specs = [s for s in game_specs if group_for_index(s.index) == group]
            if not group_specs:
                continue

            entries = build_subfig_entries(group_specs, game_title=game_title)
            if not entries:
                continue

            if group == "banzhaf":
                base_caption = (
                    f"Comparison of Budget and runtime on various {game_type} datasets "
                    "for the Banzhaf value and Banzhaf Interaction Index."
                )
            else:
                base_caption = (
                    f"Comparison of Budget and runtime on various {game_type} datasets "
                    "for the Shapley value and Shapley Interaction Index."
                )

            chunks = list(chunked(entries, args.max_subfigures))
            for part_idx, part in enumerate(chunks, start=1):
                suffix = "" if len(chunks) == 1 else f"_part{part_idx}"
                label = f"fig:runtime_vs_budget_{group}_{game_type}{suffix}"
                caption = (
                    base_caption
                    if len(chunks) == 1
                    else f"{base_caption} (Part {part_idx}/{len(chunks)})."
                )

                content = render_figure_block(
                    float_spec=args.float_spec,
                    width=args.width,
                    label=label,
                    caption=caption,
                    entries=part,
                )

                filename = f"runtime_vs_budget_{group}_{game_type}{suffix}.tex"
                outputs.append((filename, content))

    if args.stdout:
        for _, content in outputs:
            print(content)
        return 0

    args.output.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs:
        (args.output / filename).write_text(content, encoding="utf-8")

    print(f"Wrote {len(outputs)} snippet(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
