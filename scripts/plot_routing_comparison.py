#!/usr/bin/env python3
"""Render SHINE oracle-vs-embedding routing comparison figures.

Reads ``judgments.jsonl`` from each of two run directories and writes
one or both of:

  --out           grouped score-distribution chart
                  (P(score=s) for s in 1..5, oracle vs embedding)
  --out-per-repo  per-repo mean-score chart (one pair of bars per repo)

At least one of ``--out`` / ``--out-per-repo`` must be provided.

Usage:

    uv run python scripts/plot_routing_comparison.py \\
        --oracle-run    results/all_repos_easy_oracle_20260430T2345 \\
        --embedding-run results/all_repos_easy_embedding_top1_20260430T2345 \\
        --out           results/figures/shine_routing_score_distribution.png \\
        --out-per-repo  results/figures/shine_routing_per_repo.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.compare_plots import render_per_repo_means, render_routing_score_delta  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-run", required=True, type=Path)
    parser.add_argument("--embedding-run", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path,
        help="path for the grouped score-distribution PNG",
    )
    parser.add_argument(
        "--out-per-repo", type=Path,
        help="path for the per-repo mean-score PNG",
    )
    parser.add_argument(
        "--unpaired",
        action="store_true",
        help="compare full SHINE sets instead of (repo, doc, qa)-paired sets",
    )
    parser.add_argument("--oracle-label", default="oracle")
    parser.add_argument("--embedding-label", default="embedding (top-1)")
    parser.add_argument("--naive-label", default="naive")
    parser.add_argument(
        "--show-naive", action="store_true",
        help="add a naive-condition bar to --out-per-repo (no effect on --out)",
    )
    parser.add_argument("--title", default=None,
                        help="overrides the title of --out (distribution plot)")
    parser.add_argument("--per-repo-title", default=None,
                        help="overrides the title of --out-per-repo")
    args = parser.parse_args(argv)

    if args.out is None and args.out_per_repo is None:
        parser.error("at least one of --out / --out-per-repo must be provided")

    common = dict(
        oracle_run_dir=args.oracle_run,
        embedding_run_dir=args.embedding_run,
        paired=not args.unpaired,
        oracle_label=args.oracle_label,
        embedding_label=args.embedding_label,
    )

    if args.out is not None:
        out = render_routing_score_delta(out_path=args.out, title=args.title, **common)
        print(out)
    if args.out_per_repo is not None:
        out = render_per_repo_means(
            out_path=args.out_per_repo,
            title=args.per_repo_title,
            show_naive=args.show_naive,
            naive_label=args.naive_label,
            **common,
        )
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
