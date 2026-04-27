#!/usr/bin/env python3
"""Generate result plots for a single eval-harness run.

Reads ``judgments.jsonl`` (or any compatible file) from a run directory and
writes PNGs back into the same directory.

Usage:
    python scripts/plot_eval.py --run-dir results/kilo_easy_v0_20260426T2037
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.plots import render_run_plots  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--source",
        default="judgments.jsonl",
        help="Filename inside --run-dir to read (default judgments.jsonl)",
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Render plots for conditions with zero rows instead of skipping them",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    try:
        out_paths = render_run_plots(
            args.run_dir.resolve(),
            source=args.source,
            show_empty=args.show_empty,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for p in out_paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
