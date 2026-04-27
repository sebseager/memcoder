#!/usr/bin/env python3
"""System-prompt A/B test across one or more repos.

Compares three system-prompt variants on the same eval artifacts:

    baseline   (existing run already on disk; pointed at via --repo)
    detail     drops "concisely / final answer"; asks for specifics
    adapted    detail with a leading sentence telling the model its weights
               have been adapted to encode a code-repo document

Conditions per variant:

    detail     -> naive + shine
    adapted    -> shine only       (naive has no adaptation to introspect on)
    in_context not re-run (baseline already established the ceiling)

Usage:

    python scripts/prompt_ab_test.py \\
        --repo config/eval/marimo_easy_v0.yaml:results/marimo_easy_v0_20260427T1526 \\
        --repo config/eval/kilo_easy_v0.yaml:results/kilo_easy_v0_20260427T0138

Outputs (per repo):

    results/<run_name>_detail_<TS>/        predictions, judgments, report
    results/<run_name>_adapted_<TS>/       predictions, judgments, report

Outputs (combined):

    results/prompt_ab_<TS>/summary.md      cross-repo summary, per-repo sections
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

from eval import model as model_module  # noqa: E402
from eval.config import load_run_config, load_snapshot  # noqa: E402
from eval.judge import run_judging  # noqa: E402
from eval.plots import render_naive_filtered_plots, render_run_plots  # noqa: E402
from eval.report import write_report  # noqa: E402
from eval.runner import run_predictions  # noqa: E402

LOGGER = logging.getLogger("memcoder.prompt_ab_test")

# Variant colors — distinct from the eval/plots.py condition palette so the
# A/B comparison plots are visually separable from per-run plots.
VARIANT_ORDER = ("baseline", "detail", "adapted")
VARIANT_COLORS = {
    "baseline": "#888888",
    "detail": "#4f8cc9",
    "adapted": "#d97706",
}

DETAIL_PROMPT = (
    "You are a helpful assistant. Answer with specific detail — name the "
    "identifiers, mechanisms, or steps the question is asking about. If the "
    "question asks for multiple items, list each one."
)

ADAPTED_PREFIX = (
    "Your weights have been adapted to encode a specific document about a "
    "code repository. Draw on that adapted knowledge to answer."
)
ADAPTED_PROMPT = ADAPTED_PREFIX + " " + DETAIL_PROMPT

BASELINE_PROMPT_TEXT = (
    "You are a helpful assistant. Answer the question directly and "
    "concisely. Output only the final answer."
)


# ---------------------------------------------------------------------------
# Variant runner
# ---------------------------------------------------------------------------


def _build_messages_factory(system_prompt: str):
    """Return a build_messages replacement that uses ``system_prompt`` for
    naive/shine and the harness's normal in-context wrapping otherwise.

    Accepts a ``condition`` kwarg for compatibility with the runtime
    signature, but ignores it — the variant prompt overrides whatever
    the condition would otherwise select.
    """

    def build_messages(
        question: str,
        document: str | None = None,
        *,
        condition: str | None = None,  # noqa: ARG001 — accepted for sig compat
    ) -> list[dict[str, str]]:
        if document is not None:
            prompt = (
                "You are a helpful assistant. Answer the question based on the given "
                "context. Do not invent information. Answer directly and concisely.\n\n"
                f"Context:\n{document}"
            )
        else:
            prompt = system_prompt
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]

    return build_messages


def _run_variant(
    *,
    cfg_path: Path,
    variant_name: str,
    system_prompt: str,
    conditions: list[str],
) -> Path:
    """Run predict -> judge -> report for one prompt variant. Returns run_dir."""

    cfg = load_run_config(cfg_path)
    cfg.run_name = f"{cfg.run_name}_{variant_name}"
    cfg.conditions = list(conditions)

    LOGGER.info(
        "Variant=%s run_name=%s conditions=%s",
        variant_name,
        cfg.run_name,
        cfg.conditions,
    )

    original_build_messages = model_module.build_messages
    model_module.build_messages = _build_messages_factory(system_prompt)
    try:
        predictions_path = run_predictions(cfg)
    finally:
        model_module.build_messages = original_build_messages

    run_dir = predictions_path.parent
    (run_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")

    snapshot_cfg = load_snapshot(run_dir)
    run_judging(snapshot_cfg, run_dir)
    write_report(run_dir)
    render_run_plots(run_dir)
    render_naive_filtered_plots(run_dir)

    return run_dir


# ---------------------------------------------------------------------------
# A/B-specific plots
# ---------------------------------------------------------------------------


def _cell_scores_for_repo(
    baseline: dict[tuple[str, str], dict],
    detail: dict[tuple[str, str], dict],
    adapted: dict[tuple[str, str], dict],
) -> dict[tuple[str, str], list[int]]:
    """Build {(condition, variant): [scores]} for one repo."""
    cell_scores: dict[tuple[str, str], list[int]] = defaultdict(list)
    for source, variant in ((baseline, "baseline"), (detail, "detail"), (adapted, "adapted")):
        for (qa_id, cond), row in source.items():
            score = (row.get("judge") or {}).get("score")
            if isinstance(score, int):
                cell_scores[(cond, variant)].append(score)
    return cell_scores


def _plot_repo_mean_scores(
    *,
    repo_id: str,
    cell_scores: dict[tuple[str, str], list[int]],
    out_dir: Path,
    naive_le2: bool = False,
) -> Path:
    """Grouped bar chart: x = condition, bars = variants, y = mean score."""
    conditions = ("naive", "shine")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(conditions))
    width = 0.25

    for i, variant in enumerate(VARIANT_ORDER):
        means = []
        cis = []
        ns = []
        for cond in conditions:
            s = cell_scores.get((cond, variant), [])
            n = len(s)
            ns.append(n)
            if n == 0:
                means.append(0.0)
                cis.append(0.0)
                continue
            m = sum(s) / n
            var = sum((v - m) ** 2 for v in s) / n
            sd = math.sqrt(var)
            cis.append(1.96 * sd / math.sqrt(n))
            means.append(m)

        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=cis,
            label=variant,
            color=VARIANT_COLORS[variant],
            edgecolor="white",
            capsize=4,
        )
        for bar, m, n in zip(bars, means, ns):
            if n == 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{m:.2f}\nn={n}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean judge score")
    suffix = " (naive ≤ 2 filter)" if naive_le2 else ""
    ax.set_title(f"{repo_id}{suffix} — mean score by condition × variant\n"
                 "(error bars: 95% CI)")
    ax.legend(title="Prompt variant")
    ax.grid(axis="y", alpha=0.3)

    name = "mean_score_naive_le2.png" if naive_le2 else "mean_score.png"
    out = out_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_repo_score_distribution(
    *,
    repo_id: str,
    cell_scores: dict[tuple[str, str], list[int]],
    out_dir: Path,
) -> Path:
    """Stacked-percent bars: x = (condition × variant) cell, y = % at each score."""
    score_levels = (1, 2, 3, 4, 5)
    score_colors = ("#cf2e2e", "#e08a3c", "#e2c948", "#7ec466", "#3a8d3a")

    cells = [
        ("naive / baseline", ("naive", "baseline")),
        ("naive / detail", ("naive", "detail")),
        ("shine / baseline", ("shine", "baseline")),
        ("shine / detail", ("shine", "detail")),
        ("shine / adapted", ("shine", "adapted")),
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(cells))
    bottoms = np.zeros(len(cells))

    for s_idx, score in enumerate(score_levels):
        heights = []
        for _, key in cells:
            scores = cell_scores.get(key, [])
            n = len(scores)
            if n == 0:
                heights.append(0.0)
                continue
            heights.append(100.0 * sum(1 for v in scores if v == score) / n)
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=score_colors[s_idx],
            edgecolor="white",
            label=f"score {score}",
        )
        bottoms = bottoms + np.array(heights)

    # Annotate each bar with its mean score
    for i, (_, key) in enumerate(cells):
        scores = cell_scores.get(key, [])
        if not scores:
            continue
        m = sum(scores) / len(scores)
        ax.text(i, 102, f"μ={m:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in cells], rotation=15, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of answers at each score")
    ax.set_title(f"{repo_id} — score distribution by condition × variant")
    ax.legend(title="Judge score", loc="upper right", bbox_to_anchor=(1.18, 1.0))
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "score_distribution.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_per_qa_shine_delta(
    *,
    repo_id: str,
    baseline: dict[tuple[str, str], dict],
    detail: dict[tuple[str, str], dict],
    adapted: dict[tuple[str, str], dict],
    out_dir: Path,
) -> Path:
    """Per-QA SHINE score across the three variants. One line per variant,
    sorted by baseline-shine score so deltas are easy to read."""
    qa_ids = sorted({qa for (qa, _) in baseline})
    qa_shine = []
    for qa_id in qa_ids:
        b = (baseline.get((qa_id, "shine")) or {}).get("judge", {}).get("score")
        d = (detail.get((qa_id, "shine")) or {}).get("judge", {}).get("score")
        a = (adapted.get((qa_id, "shine")) or {}).get("judge", {}).get("score")
        if b is None and d is None and a is None:
            continue
        qa_shine.append((qa_id, b, d, a))

    if not qa_shine:
        # Empty plot rather than crash — keeps the report self-consistent.
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No SHINE rows for this repo", ha="center", va="center")
        ax.axis("off")
        out = out_dir / "per_qa_shine.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    qa_shine.sort(key=lambda r: (r[1] if r[1] is not None else -1))
    qa_labels = [r[0].replace(f"{repo_id}_", "") for r in qa_shine]
    bs = [r[1] if r[1] is not None else np.nan for r in qa_shine]
    ds = [r[2] if r[2] is not None else np.nan for r in qa_shine]
    as_ = [r[3] if r[3] is not None else np.nan for r in qa_shine]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.35 * len(qa_shine)), 5.0))
    x = np.arange(len(qa_shine))
    ax.plot(x, bs, "o-", label="baseline", color=VARIANT_COLORS["baseline"], linewidth=1.5)
    ax.plot(x, ds, "s-", label="detail", color=VARIANT_COLORS["detail"], linewidth=1.5)
    ax.plot(x, as_, "^-", label="adapted", color=VARIANT_COLORS["adapted"], linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(qa_labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_ylabel("SHINE judge score")
    ax.set_xlabel("QA (sorted by baseline shine score)")
    ax.set_title(f"{repo_id} — SHINE score per QA across variants")
    ax.legend(title="Prompt variant", loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "per_qa_shine.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_cross_repo_overview(
    *,
    repo_summaries: list[tuple[str, dict[tuple[str, str], list[int]]]],
    out_dir: Path,
) -> Path:
    """Per-repo deltas: bar chart, x = repo, two bars per cell (Δdetail, Δadapted)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True)

    for ax_idx, cond in enumerate(("naive", "shine")):
        ax = axes[ax_idx]
        repos = [rid for rid, _ in repo_summaries]
        x = np.arange(len(repos))
        width = 0.35

        d_deltas = []
        a_deltas = []
        for _, scores in repo_summaries:
            b = scores.get((cond, "baseline"), [])
            d = scores.get((cond, "detail"), [])
            a = scores.get((cond, "adapted"), [])
            d_deltas.append((mean(d) - mean(b)) if (b and d) else None)
            a_deltas.append((mean(a) - mean(b)) if (b and a) else None)

        d_vals = [v if v is not None else 0.0 for v in d_deltas]
        a_vals = [v if v is not None else 0.0 for v in a_deltas]

        bars_d = ax.bar(
            x - width / 2, d_vals, width,
            label="detail − baseline",
            color=VARIANT_COLORS["detail"], edgecolor="white",
        )
        bars_a = ax.bar(
            x + width / 2, a_vals, width,
            label="adapted − baseline",
            color=VARIANT_COLORS["adapted"], edgecolor="white",
        )

        for bar, v in zip(bars_d, d_deltas):
            if v is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.02 if v >= 0 else -0.06),
                    f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8)
        for bar, v in zip(bars_a, a_deltas):
            if v is None:
                bar.set_alpha(0.2)
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.02 if v >= 0 else -0.06),
                    f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8)

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(repos, rotation=15, ha="right")
        ax.set_title(f"Δ mean score for `{cond}`")
        ax.grid(axis="y", alpha=0.3)
        if ax_idx == 0:
            ax.set_ylabel("Δ mean score (variant − baseline)")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Cross-repo: prompt variant deltas vs. baseline", fontsize=12)
    out = out_dir / "cross_repo_deltas.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def _load_judgments(
    run_dir: Path,
    *,
    judgments_filename: str = "judgments.jsonl",
) -> dict[tuple[str, str], dict]:
    """Return {(qa_id, condition): row} from a run's judgments file.

    The default reads ``judgments.jsonl`` (v0 rubric). Pass
    ``judgments_filename="judgments_v1.jsonl"`` to read the sidecar file
    written by ``scripts/rejudge.py``.
    """
    out: dict[tuple[str, str], dict] = {}
    path = run_dir / judgments_filename
    if not path.exists():
        raise FileNotFoundError(f"missing {judgments_filename} in {run_dir}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[(row["qa_id"], row["condition"])] = row
    return out


def _repo_id_from_judgments(judgments: dict[tuple[str, str], dict]) -> str:
    """Pull the repo_id field from any judgment row."""
    for row in judgments.values():
        rid = row.get("repo_id")
        if rid:
            return str(rid)
    return "unknown"


def _format_score_cell(scores: list[int]) -> str:
    if not scores:
        return "—"
    return f"{mean(scores):.2f} (n={len(scores)})"


def _render_repo_section(
    *,
    baseline_run_dir: Path,
    detail_run_dir: Path,
    adapted_run_dir: Path,
    plot_dir: Path | None = None,
    judgments_filename: str = "judgments.jsonl",
) -> tuple[str, dict[tuple[str, str], list[int]], list[str]]:
    """Render one repo's worth of summary lines.

    Returns
    -------
    repo_id : str
    cell_scores : dict[(condition, variant), list[int]]
        Per-cell score lists. Used by the cross-repo aggregation.
    lines : list[str]
        Markdown lines for this repo's section (without a top-level heading).
    """

    baseline = _load_judgments(baseline_run_dir, judgments_filename=judgments_filename)
    detail = _load_judgments(detail_run_dir, judgments_filename=judgments_filename)
    adapted = _load_judgments(adapted_run_dir, judgments_filename=judgments_filename)
    repo_id = _repo_id_from_judgments(baseline)

    qa_ids = sorted({qa for (qa, _) in baseline})

    cell_scores: dict[tuple[str, str], list[int]] = defaultdict(list)
    rows: list[dict] = []

    # Plots — written before lines are assembled so we can reference them.
    plot_paths: list[Path] = []
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        cs = _cell_scores_for_repo(baseline, detail, adapted)
        plot_paths.append(_plot_repo_mean_scores(
            repo_id=repo_id, cell_scores=cs, out_dir=plot_dir, naive_le2=False,
        ))
        plot_paths.append(_plot_repo_score_distribution(
            repo_id=repo_id, cell_scores=cs, out_dir=plot_dir,
        ))
        plot_paths.append(_plot_per_qa_shine_delta(
            repo_id=repo_id, baseline=baseline, detail=detail, adapted=adapted,
            out_dir=plot_dir,
        ))
        # Naive-le2 filtered mean
        le2_qa = {
            qa_id for qa_id in qa_ids
            if (baseline.get((qa_id, "naive")) or {}).get("judge", {}).get("score", 99) <= 2
        }
        cs_le2: dict[tuple[str, str], list[int]] = defaultdict(list)
        for source, variant in ((baseline, "baseline"), (detail, "detail"), (adapted, "adapted")):
            for (qa_id, cond), row in source.items():
                if qa_id not in le2_qa:
                    continue
                score = (row.get("judge") or {}).get("score")
                if isinstance(score, int):
                    cs_le2[(cond, variant)].append(score)
        if any(cs_le2.values()):
            plot_paths.append(_plot_repo_mean_scores(
                repo_id=repo_id, cell_scores=cs_le2, out_dir=plot_dir, naive_le2=True,
            ))

    for qa_id in qa_ids:
        b_naive = baseline.get((qa_id, "naive"))
        b_shine = baseline.get((qa_id, "shine"))
        d_naive = detail.get((qa_id, "naive"))
        d_shine = detail.get((qa_id, "shine"))
        a_shine = adapted.get((qa_id, "shine"))

        rows.append(
            {
                "qa_id": qa_id,
                "question": (b_naive or b_shine or {}).get("question", ""),
                "expected": (b_naive or b_shine or {}).get("expected_answer", ""),
                "baseline_naive": b_naive,
                "baseline_shine": b_shine,
                "detail_naive": d_naive,
                "detail_shine": d_shine,
                "adapted_shine": a_shine,
            }
        )

        for cell_key, row in (
            (("naive", "baseline"), b_naive),
            (("naive", "detail"), d_naive),
            (("shine", "baseline"), b_shine),
            (("shine", "detail"), d_shine),
            (("shine", "adapted"), a_shine),
        ):
            if row is not None:
                cell_scores[cell_key].append(row["judge"]["score"])

    lines: list[str] = []
    lines.append(f"Repo: **{repo_id}**  (n={len(qa_ids)} QAs)")
    lines.append("")
    lines.append(f"- Baseline run: `{baseline_run_dir.relative_to(REPO_ROOT) if baseline_run_dir.is_relative_to(REPO_ROOT) else baseline_run_dir}`")
    lines.append(f"- Detail run:   `{detail_run_dir.relative_to(REPO_ROOT) if detail_run_dir.is_relative_to(REPO_ROOT) else detail_run_dir}`")
    lines.append(f"- Adapted run:  `{adapted_run_dir.relative_to(REPO_ROOT) if adapted_run_dir.is_relative_to(REPO_ROOT) else adapted_run_dir}`")
    lines.append("")

    # Plot embeds
    if plot_paths:
        lines.append("**Plots:**")
        lines.append("")
        for p in plot_paths:
            try:
                rel = p.relative_to(plot_dir.parent) if plot_dir is not None else p
            except ValueError:
                rel = p
            lines.append(f"![{p.stem}]({rel})")
            lines.append("")

    # Aggregate
    lines.append("**Mean judge score (1–5):**")
    lines.append("")
    lines.append("| Condition | baseline | detail | adapted |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        "| naive | "
        f"{_format_score_cell(cell_scores[('naive', 'baseline')])} | "
        f"{_format_score_cell(cell_scores[('naive', 'detail')])} | "
        "— |"
    )
    lines.append(
        "| shine | "
        f"{_format_score_cell(cell_scores[('shine', 'baseline')])} | "
        f"{_format_score_cell(cell_scores[('shine', 'detail')])} | "
        f"{_format_score_cell(cell_scores[('shine', 'adapted')])} |"
    )
    lines.append("")

    # Distributions
    lines.append("**Score distributions (% at each score):**")
    lines.append("")
    lines.append("| Cell | 1 | 2 | 3 | 4 | 5 | mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cond, var in [
        ("naive", "baseline"),
        ("naive", "detail"),
        ("shine", "baseline"),
        ("shine", "detail"),
        ("shine", "adapted"),
    ]:
        scores = cell_scores[(cond, var)]
        if not scores:
            continue
        n = len(scores)
        dist = [sum(1 for s in scores if s == k) for k in range(1, 6)]
        pct = [f"{100 * c / n:.0f}%" for c in dist]
        lines.append(
            f"| {cond} / {var} | " + " | ".join(pct) + f" | {mean(scores):.2f} |"
        )
    lines.append("")

    # Naive-le2 filtered view
    le2_qa = {
        qa_id
        for qa_id in qa_ids
        if (baseline.get((qa_id, "naive")) or {}).get("judge", {}).get("score", 99) <= 2
    }
    lines.append(f"**Filtered to questions where baseline-naive ≤ 2** (kept {len(le2_qa)}/{len(qa_ids)}):")
    lines.append("")
    lines.append("| Cell | mean | n |")
    lines.append("|---|---:|---:|")
    for cond, var, source in [
        ("naive", "baseline", baseline),
        ("naive", "detail", detail),
        ("shine", "baseline", baseline),
        ("shine", "detail", detail),
        ("shine", "adapted", adapted),
    ]:
        ss = [
            source[(qa_id, cond)]["judge"]["score"]
            for qa_id in le2_qa
            if (qa_id, cond) in source
        ]
        if ss:
            lines.append(f"| {cond} / {var} | {mean(ss):.2f} | {len(ss)} |")
    lines.append("")

    # Per-QA table
    lines.append("**Per-QA scores:**")
    lines.append("")
    lines.append("| qa_id | b_naive | d_naive | b_shine | d_shine | a_shine |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        bn = r["baseline_naive"]["judge"]["score"] if r["baseline_naive"] else "—"
        dn = r["detail_naive"]["judge"]["score"] if r["detail_naive"] else "—"
        bs = r["baseline_shine"]["judge"]["score"] if r["baseline_shine"] else "—"
        ds = r["detail_shine"]["judge"]["score"] if r["detail_shine"] else "—"
        asc = r["adapted_shine"]["judge"]["score"] if r["adapted_shine"] else "—"
        lines.append(f"| {r['qa_id']} | {bn} | {dn} | {bs} | {ds} | {asc} |")
    lines.append("")

    # Shine answer text dump (folded)
    lines.append("<details>")
    lines.append("<summary>SHINE answer text across variants</summary>")
    lines.append("")
    for r in rows:
        bs = r["baseline_shine"]
        ds = r["detail_shine"]
        asc = r["adapted_shine"]
        if not (bs or ds or asc):
            continue
        lines.append(f"#### `{r['qa_id']}`")
        lines.append("")
        lines.append(f"**Q:** {r['question']}")
        lines.append("")
        lines.append(f"**Expected:** {r['expected']}")
        lines.append("")
        if bs:
            lines.append(f"- **shine / baseline** (score {bs['judge']['score']}): {bs['answer']}")
        if ds:
            lines.append(f"- **shine / detail**   (score {ds['judge']['score']}): {ds['answer']}")
        if asc:
            lines.append(f"- **shine / adapted**  (score {asc['judge']['score']}): {asc['answer']}")
        lines.append("")
    lines.append("</details>")

    return repo_id, cell_scores, lines


def _render_combined_overview(
    repo_summaries: list[tuple[str, dict[tuple[str, str], list[int]]]],
) -> list[str]:
    """Cross-repo overview table — one row per (repo, condition), columns are variants."""
    lines: list[str] = []
    lines.append("## Cross-repo overview (mean score per cell)")
    lines.append("")
    lines.append("Per-repo means; do **not** average these across repos — different "
                 "question-style mixes mean cross-repo aggregates can hide flips.")
    lines.append("")
    lines.append("| Repo | Condition | baseline | detail | adapted | Δ detail−baseline | Δ adapted−baseline |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    for repo_id, cell_scores in repo_summaries:
        for cond in ("naive", "shine"):
            b = mean(cell_scores[(cond, "baseline")]) if cell_scores[(cond, "baseline")] else None
            d = mean(cell_scores[(cond, "detail")]) if cell_scores[(cond, "detail")] else None
            a = mean(cell_scores[(cond, "adapted")]) if cell_scores[(cond, "adapted")] else None
            b_str = f"{b:.2f}" if b is not None else "—"
            d_str = f"{d:.2f}" if d is not None else "—"
            a_str = f"{a:.2f}" if a is not None else "—"
            db = f"{d - b:+.2f}" if (b is not None and d is not None) else "—"
            ab = f"{a - b:+.2f}" if (b is not None and a is not None) else "—"
            lines.append(f"| {repo_id} | {cond} | {b_str} | {d_str} | {a_str} | {db} | {ab} |")
    lines.append("")
    return lines


def _summarize_multi(
    *,
    runs: list[tuple[Path, Path, Path]],  # [(baseline, detail, adapted), ...]
    out_dir: Path,
    judgments_filename: str = "judgments.jsonl",
) -> Path:
    """Write one combined summary covering all repos."""

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.md"

    lines: list[str] = []
    lines.append("# Prompt A/B Summary")
    lines.append("")
    lines.append("## Prompts")
    lines.append("")
    lines.append("**baseline** (naive/shine):")
    lines.append(f"> {BASELINE_PROMPT_TEXT}")
    lines.append("")
    lines.append("**detail** (naive/shine):")
    lines.append(f"> {DETAIL_PROMPT}")
    lines.append("")
    lines.append("**adapted** (shine only):")
    lines.append(f"> {ADAPTED_PROMPT}")
    lines.append("")

    repo_summaries: list[tuple[str, dict[tuple[str, str], list[int]]]] = []
    repo_sections: list[tuple[str, list[str]]] = []
    for baseline_dir, detail_dir, adapted_dir in runs:
        # Pre-load to derive repo_id for the plot subdir.
        repo_id_preview = _repo_id_from_judgments(
            _load_judgments(baseline_dir, judgments_filename=judgments_filename)
        )
        plot_dir = out_dir / repo_id_preview
        repo_id, cell_scores, section_lines = _render_repo_section(
            baseline_run_dir=baseline_dir,
            detail_run_dir=detail_dir,
            adapted_run_dir=adapted_dir,
            plot_dir=plot_dir,
            judgments_filename=judgments_filename,
        )
        repo_summaries.append((repo_id, cell_scores))
        repo_sections.append((repo_id, section_lines))

    # Cross-repo overview goes first.
    if len(repo_summaries) > 1:
        cross_plot = _plot_cross_repo_overview(
            repo_summaries=repo_summaries, out_dir=out_dir,
        )
        lines.extend(_render_combined_overview(repo_summaries))
        lines.append(f"![cross_repo_deltas]({cross_plot.name})")
        lines.append("")

    # Then one section per repo.
    for repo_id, section_lines in repo_sections:
        lines.append(f"## {repo_id}")
        lines.append("")
        lines.extend(section_lines)
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_repo_arg(value: str) -> tuple[Path, Path]:
    """Parse ``CONFIG:BASELINE_RUN_DIR`` into (Path, Path)."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"--repo must be CONFIG:BASELINE_RUN_DIR, got {value!r}"
        )
    cfg_str, baseline_str = value.split(":", 1)
    cfg = Path(cfg_str)
    baseline = Path(baseline_str)
    if not cfg.exists():
        raise argparse.ArgumentTypeError(f"config not found: {cfg}")
    if not baseline.exists():
        raise argparse.ArgumentTypeError(f"baseline run dir not found: {baseline}")
    return cfg, baseline


def _parse_summary_only_arg(value: str) -> tuple[Path, Path, Path]:
    """Parse ``BASELINE_DIR:DETAIL_DIR:ADAPTED_DIR`` into a triple."""
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--summary-only repo arg must be "
            f"BASELINE_DIR:DETAIL_DIR:ADAPTED_DIR, got {value!r}"
        )
    paths = [Path(p) for p in parts]
    for p in paths:
        if not p.exists():
            raise argparse.ArgumentTypeError(f"run dir not found: {p}")
    return tuple(paths)  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--repo",
        action="append",
        type=_parse_repo_arg,
        metavar="CONFIG:BASELINE_RUN_DIR",
        help="Repeat once per repo. Each value pairs a run-config YAML with "
             "an existing baseline run directory. Use this for the full "
             "pipeline (predict + judge + report + summary).",
    )
    p.add_argument(
        "--summary-only",
        action="append",
        type=_parse_summary_only_arg,
        metavar="BASELINE_DIR:DETAIL_DIR:ADAPTED_DIR",
        help="Re-render the summary against existing run dirs without "
             "re-running prediction or judging. Repeat once per repo. "
             "Pair with --judgments-filename to use sidecar judgments.",
    )
    p.add_argument(
        "--judgments-filename",
        type=str,
        default="judgments.jsonl",
        help="Filename within each run dir to read judgments from (default: "
             "judgments.jsonl). Use 'judgments_v1.jsonl' to summarize the "
             "v1-rubric rejudge sidecars.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for the combined summary. Defaults to "
             "results/prompt_ab_<TS>/.",
    )
    args = p.parse_args()
    if not args.repo and not args.summary_only:
        p.error("provide either --repo (full pipeline) or --summary-only "
                "(re-render existing run dirs)")
    if args.repo and args.summary_only:
        p.error("--repo and --summary-only are mutually exclusive")
    return args


def main() -> int:
    _setup_logging()
    args = parse_args()

    runs: list[tuple[Path, Path, Path]] = []

    if args.summary_only:
        runs = list(args.summary_only)
        LOGGER.info(
            "Summary-only mode: re-rendering %d repo(s) from %s",
            len(runs),
            args.judgments_filename,
        )
    else:
        for cfg_path, baseline_dir in args.repo:
            LOGGER.info("===== Repo: cfg=%s baseline=%s =====", cfg_path, baseline_dir)

            detail_dir = _run_variant(
                cfg_path=cfg_path,
                variant_name="detail",
                system_prompt=DETAIL_PROMPT,
                conditions=["naive", "shine"],
            )
            LOGGER.info("Detail run -> %s", detail_dir)

            adapted_dir = _run_variant(
                cfg_path=cfg_path,
                variant_name="adapted",
                system_prompt=ADAPTED_PROMPT,
                conditions=["shine"],
            )
            LOGGER.info("Adapted run -> %s", adapted_dir)

            runs.append((baseline_dir, detail_dir, adapted_dir))

    summary_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    suffix = ""
    if args.judgments_filename != "judgments.jsonl":
        # e.g., judgments_v1.jsonl -> _v1
        stem = Path(args.judgments_filename).stem  # judgments_v1
        if stem.startswith("judgments_"):
            suffix = "_" + stem[len("judgments_"):]
    out_dir = args.out_dir or (
        REPO_ROOT / "results" / f"prompt_ab{suffix}_{summary_ts}"
    )
    summary_path = _summarize_multi(
        runs=runs,
        out_dir=out_dir,
        judgments_filename=args.judgments_filename,
    )

    print()
    print("=" * 72)
    print("Per-repo runs:")
    for baseline_dir, detail_dir, adapted_dir in runs:
        print(f"  baseline: {baseline_dir}")
        print(f"  detail:   {detail_dir}")
        print(f"  adapted:  {adapted_dir}")
        print()
    print(f"Judgments file: {args.judgments_filename}")
    print(f"Combined summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
