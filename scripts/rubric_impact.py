#!/usr/bin/env python3
"""Compare two judge-rubric versions side-by-side on the same predictions.

Reads two judgments files from each provided run dir (default:
``judgments.jsonl`` for v0 and ``judgments_v1.jsonl`` for v1) and writes a
markdown summary that shows how the rubric change moves the headline
numbers per (repo, condition, prompt-variant) cell.

Use this after ``scripts/rejudge.py`` writes the v1 sidecars to evaluate
whether the rubric change behaves as intended.

Usage:

    python scripts/rubric_impact.py \\
        --repo marimo \\
            results/marimo_easy_v0_20260427T1526 \\
            results/marimo_easy_v0_detail_20260427T1720 \\
            results/marimo_easy_v0_adapted_20260427T1728 \\
        --repo kilo \\
            results/kilo_easy_v0_20260427T0138 \\
            results/kilo_easy_v0_detail_20260427T1731 \\
            results/kilo_easy_v0_adapted_20260427T1747

Outputs:

    results/rubric_impact_<TS>/impact.md
    results/rubric_impact_<TS>/<repo_id>_score_shift.png  # per-repo
    results/rubric_impact_<TS>/distribution_shift.png     # all cells
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("memcoder.rubric_impact")

VARIANT_LABELS = ("baseline", "detail", "adapted")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load(path: Path) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[(row["qa_id"], row["condition"])] = row
    return out


def _scores_by_condition(rows: dict[tuple[str, str], dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for (_, cond), row in rows.items():
        score = (row.get("judge") or {}).get("score")
        if isinstance(score, int):
            out[cond].append(score)
    return out


def _plot_score_shift(
    *,
    repo_id: str,
    cells: list[tuple[str, list[int], list[int]]],  # (label, v0_scores, v1_scores)
    out_path: Path,
) -> None:
    """Per-repo: paired bars showing v0 mean vs v1 mean per cell."""
    labels = [c[0] for c in cells]
    v0_means = [mean(c[1]) if c[1] else 0.0 for c in cells]
    v1_means = [mean(c[2]) if c[2] else 0.0 for c in cells]

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, v0_means, width, label="v0 rubric", color="#888888", edgecolor="white")
    ax.bar(x + width / 2, v1_means, width, label="v1 rubric", color="#4f8cc9", edgecolor="white")

    for i, (v0, v1) in enumerate(zip(v0_means, v1_means)):
        ax.text(i - width / 2, v0 + 0.05, f"{v0:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, v1 + 0.05, f"{v1:.2f}", ha="center", va="bottom", fontsize=8)
        delta = v1 - v0
        ax.text(i, max(v0, v1) + 0.4, f"Δ{delta:+.2f}", ha="center", va="bottom",
                fontsize=8, color="#d97706" if abs(delta) > 0.1 else "#444",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean judge score")
    ax.set_title(f"{repo_id} — v0 vs v1 rubric, per (condition × variant)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_distribution_shift(
    *,
    cells: list[tuple[str, str, list[int], list[int]]],  # (repo, label, v0, v1)
    out_path: Path,
) -> None:
    """Score histogram per cell, v0 vs v1 stacked side-by-side."""
    n_cells = len(cells)
    if n_cells == 0:
        return
    fig, axes = plt.subplots(
        n_cells, 1, figsize=(10.0, 1.6 * n_cells), sharex=True
    )
    if n_cells == 1:
        axes = [axes]

    score_levels = (1, 2, 3, 4, 5)
    for ax, (repo, label, v0, v1) in zip(axes, cells):
        x = np.arange(len(score_levels))
        width = 0.35
        v0_counts = [Counter(v0).get(s, 0) for s in score_levels]
        v1_counts = [Counter(v1).get(s, 0) for s in score_levels]
        ax.bar(x - width / 2, v0_counts, width, color="#888888", label="v0", edgecolor="white")
        ax.bar(x + width / 2, v1_counts, width, color="#4f8cc9", label="v1", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in score_levels])
        ax.set_ylabel(f"{repo}\n{label}", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Judge score")
    fig.suptitle("Score distribution shift, v0 → v1 rubric", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--repo",
        nargs=4,
        action="append",
        required=True,
        metavar=("REPO_LABEL", "BASELINE_DIR", "DETAIL_DIR", "ADAPTED_DIR"),
        help="A repo label (used in output) plus the three run dirs for "
             "baseline / detail / adapted prompts.",
    )
    p.add_argument(
        "--v0-name",
        default="judgments.jsonl",
        help="Filename for v0 judgments (default: judgments.jsonl).",
    )
    p.add_argument(
        "--v1-name",
        default="judgments_v1.jsonl",
        help="Filename for v1 judgments (default: judgments_v1.jsonl).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/rubric_impact_<TS>/).",
    )
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    out_dir = args.out_dir or (REPO_ROOT / "results" / f"rubric_impact_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Judge Rubric Impact — v0 vs v1")
    lines.append("")
    lines.append(f"- v0 judgments: `{args.v0_name}`")
    lines.append(f"- v1 judgments: `{args.v1_name}`")
    lines.append("")

    # Pre-load everything
    repos_data: list[dict] = []
    for repo_args in args.repo:
        label, b_dir, d_dir, a_dir = repo_args
        b_dir, d_dir, a_dir = Path(b_dir), Path(d_dir), Path(a_dir)
        repos_data.append({
            "label": label,
            "dirs": {"baseline": b_dir, "detail": d_dir, "adapted": a_dir},
            "v0": {var: _load(p / args.v0_name) for var, p in
                   (("baseline", b_dir), ("detail", d_dir), ("adapted", a_dir))},
            "v1": {var: _load(p / args.v1_name) for var, p in
                   (("baseline", b_dir), ("detail", d_dir), ("adapted", a_dir))},
        })

    # Cross-repo headline table
    lines.append("## Headline: mean score per cell (v0 → v1)")
    lines.append("")
    lines.append("| Repo | Variant | Condition | n | v0 mean | v1 mean | Δ |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    distribution_cells = []  # for the distribution plot
    for rd in repos_data:
        repo_label = rd["label"]
        for var in VARIANT_LABELS:
            v0_rows = rd["v0"].get(var, {})
            v1_rows = rd["v1"].get(var, {})
            v0_by_cond = _scores_by_condition(v0_rows)
            v1_by_cond = _scores_by_condition(v1_rows)
            for cond in ("naive", "in_context", "shine"):
                v0_s = v0_by_cond.get(cond, [])
                v1_s = v1_by_cond.get(cond, [])
                if not v0_s and not v1_s:
                    continue
                v0_mean = mean(v0_s) if v0_s else None
                v1_mean = mean(v1_s) if v1_s else None
                delta = (v1_mean - v0_mean) if (v0_mean is not None and v1_mean is not None) else None
                v0_str = f"{v0_mean:.2f}" if v0_mean is not None else "—"
                v1_str = f"{v1_mean:.2f}" if v1_mean is not None else "—"
                d_str = f"{delta:+.2f}" if delta is not None else "—"
                # Use whichever is non-empty for n
                n = len(v0_s) if v0_s else len(v1_s)
                lines.append(
                    f"| {repo_label} | {var} | {cond} | {n} | "
                    f"{v0_str} | {v1_str} | {d_str} |"
                )
                # collect for distribution plot (skip in_context to keep plot tractable)
                if cond in ("naive", "shine"):
                    distribution_cells.append((repo_label, f"{var}/{cond}", v0_s, v1_s))
    lines.append("")

    # Per-repo plots
    for rd in repos_data:
        repo_label = rd["label"]
        cells = []
        for var in VARIANT_LABELS:
            v0_rows = rd["v0"].get(var, {})
            v1_rows = rd["v1"].get(var, {})
            v0_by_cond = _scores_by_condition(v0_rows)
            v1_by_cond = _scores_by_condition(v1_rows)
            for cond in ("naive", "shine"):
                v0_s = v0_by_cond.get(cond, [])
                v1_s = v1_by_cond.get(cond, [])
                if not v0_s and not v1_s:
                    continue
                cells.append((f"{var}/{cond}", v0_s, v1_s))
        if cells:
            plot_path = out_dir / f"{repo_label}_score_shift.png"
            _plot_score_shift(repo_id=repo_label, cells=cells, out_path=plot_path)
            lines.append(f"### {repo_label}")
            lines.append("")
            lines.append(f"![{repo_label}_score_shift]({plot_path.name})")
            lines.append("")

    # Distribution shift plot (combined)
    if distribution_cells:
        dist_path = out_dir / "distribution_shift.png"
        _plot_distribution_shift(cells=distribution_cells, out_path=dist_path)
        lines.append("## Score distribution shift")
        lines.append("")
        lines.append(f"![distribution_shift]({dist_path.name})")
        lines.append("")

    # Failure-mode shift, focused on the new no_specifics tag
    lines.append("## Failure-mode taxonomy shift")
    lines.append("")
    lines.append("Counts are over all rows scoring < 5. v1 introduces "
                 "`no_specifics` (split out from `missing_information`).")
    lines.append("")
    lines.append("| Repo | Variant | Condition | tag | v0 count | v1 count |")
    lines.append("|---|---|---|---|---:|---:|")
    for rd in repos_data:
        for var in VARIANT_LABELS:
            for cond in ("naive", "shine"):
                v0_rows = [
                    r for (_, c), r in rd["v0"].get(var, {}).items()
                    if c == cond and (r.get("judge") or {}).get("score", 5) < 5
                ]
                v1_rows = [
                    r for (_, c), r in rd["v1"].get(var, {}).items()
                    if c == cond and (r.get("judge") or {}).get("score", 5) < 5
                ]
                if not v0_rows and not v1_rows:
                    continue
                v0_tags = Counter()
                v1_tags = Counter()
                for r in v0_rows:
                    for t in (r.get("judge") or {}).get("failure_modes", []):
                        v0_tags[t] += 1
                for r in v1_rows:
                    for t in (r.get("judge") or {}).get("failure_modes", []):
                        v1_tags[t] += 1
                all_tags = sorted(set(v0_tags) | set(v1_tags))
                for tag in all_tags:
                    lines.append(
                        f"| {rd['label']} | {var} | {cond} | `{tag}` | "
                        f"{v0_tags.get(tag, 0)} | {v1_tags.get(tag, 0)} |"
                    )
    lines.append("")

    # Per-QA flips: where did the score change between rubrics?
    lines.append("## Per-QA score flips (|Δ| ≥ 1)")
    lines.append("")
    lines.append("Useful for spot-checking — these are the QAs where the "
                 "rubric change moved the score by at least 1 point.")
    lines.append("")
    lines.append("| Repo | Variant | qa_id | cond | v0 | v1 | Δ |")
    lines.append("|---|---|---|---|---:|---:|---:|")
    for rd in repos_data:
        for var in VARIANT_LABELS:
            v0_rows = rd["v0"].get(var, {})
            v1_rows = rd["v1"].get(var, {})
            keys = sorted(set(v0_rows) | set(v1_rows))
            for key in keys:
                qa_id, cond = key
                v0r = v0_rows.get(key)
                v1r = v1_rows.get(key)
                if not v0r or not v1r:
                    continue
                v0s = (v0r.get("judge") or {}).get("score")
                v1s = (v1r.get("judge") or {}).get("score")
                if not isinstance(v0s, int) or not isinstance(v1s, int):
                    continue
                if abs(v1s - v0s) >= 1:
                    lines.append(
                        f"| {rd['label']} | {var} | `{qa_id}` | {cond} | "
                        f"{v0s} | {v1s} | {v1s - v0s:+d} |"
                    )
    lines.append("")

    impact_path = out_dir / "impact.md"
    impact_path.write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"Impact report: {impact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
