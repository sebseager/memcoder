#!/usr/bin/env python3
"""Plot eval-harness performance restricted to questions where naive scored low.

Useful for stripping out questions Qwen can answer purely from pretraining
knowledge (standard ANSI escape codes, self-documenting function names,
generic editor conventions) so the remaining set actually requires document-
specific information to answer well.

Reads ``judgments.jsonl`` from a run directory, keeps only the QA pairs where
the naive condition scored at or below ``--naive-max`` (default 2), and
renders the same set of plots into a sibling subdirectory.

Usage:
    python scripts/plot_naive_hard.py --run-dir results/kilo_easy_v0_<ts>
    python scripts/plot_naive_hard.py --run-dir results/<run> --naive-max 1

Outputs:
    <run-dir>/plots_naive_le<N>/
      judgments_filtered.jsonl
      scores_per_condition.png
      mean_score_per_condition.png
      failure_modes_per_condition.png
      per_document_heatmap.png
      summary.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.plots import render_naive_filtered_plots  # noqa: E402

LOGGER = logging.getLogger("memcoder.plot_naive_hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--naive-max",
        type=int,
        default=2,
        help=(
            "Keep only questions whose naive judge score is <= this value. "
            "Default 2 (drop questions Qwen got right or partially right "
            "without any context)."
        ),
    )
    parser.add_argument(
        "--source",
        default="judgments.jsonl",
        help="Filename inside --run-dir to read (default judgments.jsonl)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    run_dir: Path = args.run_dir.resolve()
    src = run_dir / args.source
    if not src.exists():
        print(f"error: {src} not found", file=sys.stderr)
        return 1

    try:
        paths = render_naive_filtered_plots(
            run_dir, naive_max=args.naive_max, source=args.source
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not paths:
        print(
            f"error: no questions had a naive score <= {args.naive_max}; nothing to plot",
            file=sys.stderr,
        )
        return 1

    print(run_dir / f"plots_naive_le{args.naive_max}")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
