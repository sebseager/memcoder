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
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.plots import load_judgments, render_plots_from_rows  # noqa: E402

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

    rows = load_judgments(src)
    if not rows:
        print(f"error: no rows in {src}", file=sys.stderr)
        return 1

    naive_score_by_qa = _naive_scores_by_qa(rows)
    keep_qa_ids = {
        qa for qa, score in naive_score_by_qa.items() if score <= args.naive_max
    }
    if not keep_qa_ids:
        print(
            f"error: no questions had a naive score <= {args.naive_max}; nothing to plot",
            file=sys.stderr,
        )
        return 1

    filtered = [r for r in rows if r.get("qa_id") in keep_qa_ids]
    LOGGER.info(
        "kept %d/%d QA(s) where naive score <= %d -> %d total rows across all conditions",
        len(keep_qa_ids),
        len(naive_score_by_qa),
        args.naive_max,
        len(filtered),
    )

    out_dir = run_dir / f"plots_naive_le{args.naive_max}"
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = out_dir / "judgments_filtered.jsonl"
    filtered_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in filtered) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote %s", filtered_path)

    suffix = f" (naive score ≤ {args.naive_max}; n_qa={len(keep_qa_ids)})"
    plot_paths = render_plots_from_rows(
        filtered, out_dir=out_dir, title_suffix=suffix
    )

    summary_path = _write_summary(
        out_dir=out_dir,
        run_dir=run_dir,
        naive_max=args.naive_max,
        kept_qa_ids=keep_qa_ids,
        all_qa_count=len(naive_score_by_qa),
        rows=filtered,
    )
    LOGGER.info("wrote %s", summary_path)

    print(out_dir)
    for p in plot_paths:
        print(p)
    print(filtered_path)
    print(summary_path)
    return 0


def _naive_scores_by_qa(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return one naive score per qa_id. Picks the first naive row encountered."""
    out: dict[str, int] = {}
    for r in rows:
        if r.get("condition") != "naive":
            continue
        qa = r.get("qa_id")
        if not isinstance(qa, str) or qa in out:
            continue
        score = (r.get("judge") or {}).get("score")
        if isinstance(score, int):
            out[qa] = score
    return out


def _write_summary(
    *,
    out_dir: Path,
    run_dir: Path,
    naive_max: int,
    kept_qa_ids: set[str],
    all_qa_count: int,
    rows: list[dict[str, Any]],
) -> Path:
    by_cond: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        cond = str(r.get("condition") or "")
        score = (r.get("judge") or {}).get("score")
        if isinstance(score, int):
            by_cond[cond].append(score)

    lines: list[str] = []
    lines.append(
        f"# Filtered Eval Report — naive score ≤ {naive_max} — {run_dir.name}"
    )
    lines.append("")
    lines.append(
        f"- Source: `{run_dir.name}/judgments.jsonl`"
    )
    lines.append(
        f"- Kept QAs: **{len(kept_qa_ids)} / {all_qa_count}** "
        f"({100.0 * len(kept_qa_ids) / max(all_qa_count, 1):.1f}%)"
    )
    lines.append("")
    lines.append("## Score summary on filtered set")
    lines.append("")
    lines.append("| Condition | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for cond in sorted(by_cond):
        scores = by_cond[cond]
        n = len(scores)
        mean = sum(scores) / n if n else 0.0
        pct5 = 100.0 * sum(1 for s in scores if s == 5) / n if n else 0.0
        hist = Counter(scores)
        lines.append(
            f"| `{cond}` | {n} | {mean:.2f} | {pct5:.1f}% "
            f"| {hist.get(1, 0)} | {hist.get(2, 0)} | {hist.get(3, 0)} "
            f"| {hist.get(4, 0)} | {hist.get(5, 0)} |"
        )
    lines.append("")
    lines.append(
        f"Filter rule: keep only questions where the **naive** condition "
        f"judge score is ≤ {naive_max}. This isolates questions that require "
        "document-specific knowledge to answer well."
    )

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
