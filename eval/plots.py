"""Result plotting for an eval-harness run.

Reads ``judgments.jsonl`` from a run directory and writes PNG plots back into
the same directory. The CLI entrypoint at ``scripts/plot_eval.py`` is a thin
wrapper around :func:`render_run_plots` here, and ``run_eval.py report``
calls the same function automatically after writing ``report.md``.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LOGGER = logging.getLogger("memcoder.eval.plots")

FAILURE_MODES = (
    "wrong_specifics",
    "no_specifics",
    "missing_information",
    "off_topic",
    "refusal_or_nonresponse",
    "format_failure",
    "other",
)

CONDITION_ORDER = ("naive", "in_context", "shine")
CONDITION_COLORS = {
    "naive": "#d96f6f",
    "in_context": "#4f8cc9",
    "shine": "#5cb85c",
    "composition": "#5cb85c",
}


def render_run_plots(
    run_dir: Path,
    *,
    source: str = "judgments.jsonl",
    out_dir: Path | None = None,
    show_empty: bool = False,
    title_suffix: str = "",
) -> list[Path]:
    """Render all plots for ``run_dir`` and return their paths.

    Caller-friendly: missing source file raises FileNotFoundError; empty file
    raises ValueError; otherwise the function always succeeds and returns the
    written paths.
    """
    src = run_dir / source
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    rows = _load_judgments(src)
    if not rows:
        raise ValueError(f"no rows in {src}")
    return render_plots_from_rows(
        rows,
        out_dir=out_dir or run_dir,
        show_empty=show_empty,
        title_suffix=title_suffix,
    )


def render_naive_filtered_plots(
    run_dir: Path,
    *,
    naive_max: int = 2,
    source: str = "judgments.jsonl",
) -> list[Path]:
    """Render plots for QAs where the naive condition scored <= ``naive_max``.

    This filters out questions the base model could already answer without
    context/LoRA and writes the filtered rows plus summary under
    ``plots_naive_le<N>/`` inside the run directory.
    """
    src = run_dir / source
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    rows = _load_judgments(src)
    if not rows:
        raise ValueError(f"no rows in {src}")

    naive_score_by_qa = _naive_scores_by_qa(rows)
    keep_qa_ids = {
        qa_id for qa_id, score in naive_score_by_qa.items() if score <= naive_max
    }
    if not keep_qa_ids:
        LOGGER.warning(
            "No questions had naive score <= %d; skipping naive-filtered plots",
            naive_max,
        )
        return []

    filtered = [r for r in rows if r.get("qa_id") in keep_qa_ids]
    out_dir = run_dir / f"plots_naive_le{naive_max}"
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered_path = out_dir / "judgments_filtered.jsonl"
    filtered_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in filtered) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote %s", filtered_path)

    suffix = f" (naive score ≤ {naive_max}; n_qa={len(keep_qa_ids)})"
    plot_paths = render_plots_from_rows(
        filtered, out_dir=out_dir, title_suffix=suffix
    )
    summary_path = _write_naive_filtered_summary(
        out_dir=out_dir,
        run_dir=run_dir,
        naive_max=naive_max,
        kept_qa_ids=keep_qa_ids,
        all_qa_count=len(naive_score_by_qa),
        rows=filtered,
    )
    LOGGER.info("wrote %s", summary_path)
    return [filtered_path, *plot_paths, summary_path]


def render_plots_from_rows(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    show_empty: bool = False,
    title_suffix: str = "",
) -> list[Path]:
    """Render plots from already-loaded judgment rows.

    Used by ad-hoc filtering scripts that want to plot a subset of the run
    without writing a sidecar file at the standard location.
    """
    if not rows:
        raise ValueError("no rows to plot")
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = _present_conditions(rows, include_empty=show_empty)
    LOGGER.info(
        "rendering plots from %d row(s) across %d condition(s): %s",
        len(rows),
        len(conditions),
        conditions,
    )

    out_paths: list[Path] = []
    out_paths.append(_plot_score_distribution(rows, conditions, out_dir, title_suffix))
    out_paths.append(_plot_mean_score(rows, conditions, out_dir, title_suffix))
    out_paths.append(_plot_failure_modes(rows, conditions, out_dir, title_suffix))
    heatmap = _plot_per_document_heatmap(rows, conditions, out_dir, title_suffix)
    if heatmap is not None:
        out_paths.append(heatmap)
    for p in out_paths:
        LOGGER.info("wrote %s", p)
    return out_paths


def load_judgments(path: Path) -> list[dict[str, Any]]:
    """Public wrapper around the internal JSONL reader."""
    return _load_judgments(path)


def naive_scores_by_qa(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Public wrapper returning one naive score per QA ID."""
    return _naive_scores_by_qa(rows)


# ---------------------------------------------------------------------------
# Loading / shaping
# ---------------------------------------------------------------------------

def _load_judgments(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
    return rows


def _present_conditions(rows: list[dict[str, Any]], include_empty: bool) -> list[str]:
    counter = Counter(str(r.get("condition") or "") for r in rows)
    ordered = [c for c in CONDITION_ORDER if include_empty or counter.get(c, 0) > 0]
    individual = sorted(c for c in counter if c == "individual" or c.startswith("individual:"))
    composition = ["composition"] if counter.get("composition", 0) > 0 else []
    extras = sorted(
        c
        for c in counter
        if c not in CONDITION_ORDER and c not in individual and c not in composition
    )
    return ordered + individual + composition + extras


def _condition_color(condition: str) -> str:
    if condition == "individual" or condition.startswith("individual:"):
        return "#6f9fd8"
    return CONDITION_COLORS.get(condition, "#999")


def _scores_by_condition(rows: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        cond = str(r.get("condition") or "")
        score = (r.get("judge") or {}).get("score")
        if isinstance(score, int) and 1 <= score <= 5:
            out[cond].append(score)
    return out


def _naive_scores_by_qa(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Return one naive score per qa_id. Picks the first naive row encountered."""
    out: dict[str, int] = {}
    for r in rows:
        if r.get("condition") != "naive":
            continue
        qa_id = r.get("qa_id")
        if not isinstance(qa_id, str) or qa_id in out:
            continue
        score = (r.get("judge") or {}).get("score")
        if isinstance(score, int):
            out[qa_id] = score
    return out


def _write_naive_filtered_summary(
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
    lines.append(f"# Filtered Eval Report — naive score ≤ {naive_max} — {run_dir.name}")
    lines.append("")
    lines.append(f"- Source: `{run_dir.name}/judgments.jsonl`")
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


# ---------------------------------------------------------------------------
# Plot 1: score distribution per condition (grouped bars)
# ---------------------------------------------------------------------------

def _plot_score_distribution(
    rows: list[dict[str, Any]],
    conditions: list[str],
    out_dir: Path,
    title_suffix: str = "",
) -> Path:
    scores = _scores_by_condition(rows)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    score_levels = [1, 2, 3, 4, 5]
    width = 0.8 / max(len(conditions), 1)
    x = np.arange(len(score_levels))

    for i, cond in enumerate(conditions):
        counts = [Counter(scores.get(cond, [])).get(s, 0) for s in score_levels]
        offset = (i - (len(conditions) - 1) / 2) * width
        ax.bar(
            x + offset,
            counts,
            width=width,
            label=cond,
            color=_condition_color(cond),
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in score_levels])
    ax.set_xlabel("Judge score (1–5)")
    ax.set_ylabel("Count")
    ax.set_title(f"Score distribution per condition{title_suffix}")
    ax.legend(title="Condition")
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "scores_per_condition.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 2: mean score per condition with 95% CI
# ---------------------------------------------------------------------------

def _plot_mean_score(
    rows: list[dict[str, Any]],
    conditions: list[str],
    out_dir: Path,
    title_suffix: str = "",
) -> Path:
    scores = _scores_by_condition(rows)

    means: list[float] = []
    cis: list[float] = []
    ns: list[int] = []
    for cond in conditions:
        s = scores.get(cond, [])
        n = len(s)
        ns.append(n)
        if n == 0:
            means.append(0.0)
            cis.append(0.0)
            continue
        m = sum(s) / n
        var = sum((x - m) ** 2 for x in s) / n
        sd = math.sqrt(var)
        ci = 1.96 * sd / math.sqrt(n) if n > 0 else 0.0
        means.append(m)
        cis.append(ci)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    x = np.arange(len(conditions))
    bars = ax.bar(
        x,
        means,
        yerr=cis,
        color=[_condition_color(c) for c in conditions],
        edgecolor="white",
        capsize=6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean judge score")
    ax.set_title(f"Mean score per condition (error bars: 95% CI){title_suffix}")
    ax.grid(axis="y", alpha=0.3)

    for bar, m, n in zip(bars, means, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{m:.2f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    out = out_dir / "mean_score_per_condition.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 3: failure modes per condition (grouped bars)
# ---------------------------------------------------------------------------

def _plot_failure_modes(
    rows: list[dict[str, Any]],
    conditions: list[str],
    out_dir: Path,
    title_suffix: str = "",
) -> Path:
    counts: dict[str, Counter] = {c: Counter() for c in conditions}
    for r in rows:
        cond = str(r.get("condition") or "")
        if cond not in counts:
            continue
        modes = (r.get("judge") or {}).get("failure_modes") or []
        counts[cond].update(modes)

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    width = 0.8 / max(len(conditions), 1)
    x = np.arange(len(FAILURE_MODES))

    for i, cond in enumerate(conditions):
        bars_y = [counts[cond].get(m, 0) for m in FAILURE_MODES]
        offset = (i - (len(conditions) - 1) / 2) * width
        ax.bar(
            x + offset,
            bars_y,
            width=width,
            label=cond,
            color=_condition_color(cond),
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(FAILURE_MODES, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Failure modes per condition (multi-label, scores < 5 only){title_suffix}"
    )
    ax.legend(title="Condition")
    ax.grid(axis="y", alpha=0.3)

    out = out_dir / "failure_modes_per_condition.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Plot 4: per-document × condition mean score heatmap
# ---------------------------------------------------------------------------

def _plot_per_document_heatmap(
    rows: list[dict[str, Any]],
    conditions: list[str],
    out_dir: Path,
    title_suffix: str = "",
) -> Path | None:
    by_doc_cond: dict[tuple[str, str], list[int]] = defaultdict(list)
    docs: list[str] = []
    seen_docs: set[str] = set()
    for r in rows:
        doc = str(r.get("document_id") or "")
        cond = str(r.get("condition") or "")
        score = (r.get("judge") or {}).get("score")
        if not doc or cond not in conditions or not isinstance(score, int):
            continue
        if doc not in seen_docs:
            seen_docs.add(doc)
            docs.append(doc)
        by_doc_cond[(doc, cond)].append(score)

    if not docs:
        return None

    matrix = np.full((len(docs), len(conditions)), np.nan)
    for i, doc in enumerate(docs):
        for j, cond in enumerate(conditions):
            vals = by_doc_cond.get((doc, cond), [])
            if vals:
                matrix[i, j] = sum(vals) / len(vals)

    fig, ax = plt.subplots(
        figsize=(1.4 + 1.2 * len(conditions), 1.4 + 0.5 * len(docs))
    )
    cmap = plt.get_cmap("RdYlGn")
    im = ax.imshow(matrix, vmin=1, vmax=5, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(docs)))
    ax.set_yticklabels(docs)
    ax.set_title(f"Mean judge score by (document, condition){title_suffix}")

    for i in range(len(docs)):
        for j in range(len(conditions)):
            v = matrix[i, j]
            if np.isnan(v):
                txt, color = "—", "#666"
            else:
                txt = f"{v:.2f}"
                color = "white" if v < 2.5 or v > 4.0 else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, label="Mean score")
    out = out_dir / "per_document_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
