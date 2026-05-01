"""Cross-run comparison plots for the SHINE condition.

Reads ``judgments.jsonl`` from two run directories — typically the
oracle-routed run and an embedding-routed run produced by
``config/run_all_repos_easy_eval.sh`` — and renders a single, restrained
grouped bar chart of the per-score-level proportions on the ``shine``
condition. For each judge score level s in {1, 2, 3, 4, 5}, two adjacent
bars show P_oracle(score=s) and P_embedding(score=s).

Rows are matched by ``qa_id`` by default so the comparison controls for
question difficulty; pass ``paired=False`` to compare the full
distributions instead.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LOGGER = logging.getLogger("memcoder.eval.compare_plots")

SCORE_LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)

# Restrained palette. Oracle / embedding form a desaturated warm/cool
# pair that survives black-and-white printing; the naive accent matches
# the value used elsewhere in eval/plots.py for visual continuity.
COLOR_ORACLE = "#b07242"
COLOR_EMBEDDING = "#3b6e9c"
COLOR_NAIVE = "#7d7d7d"
COLOR_GRID = "#bbbbbb"
COLOR_TEXT_MUTED = "#555555"


def render_routing_score_delta(
    *,
    oracle_run_dir: Path,
    embedding_run_dir: Path,
    out_path: Path,
    paired: bool = True,
    oracle_label: str = "oracle",
    embedding_label: str = "embedding (top-1)",
    title: str | None = None,
) -> Path:
    """Render the SHINE oracle-vs-embedding score-delta figure.

    Args:
        oracle_run_dir: run directory whose ``judgments.jsonl`` is the
            baseline (subtracted).
        embedding_run_dir: run directory whose ``judgments.jsonl`` is
            compared against the baseline.
        out_path: target PNG path; parent directories are created.
        paired: when True (default), restrict to ``qa_id`` values
            present in both runs and compare matched samples; when
            False, use each run's full ``shine`` row set.
        oracle_label, embedding_label: legend / axis text overrides.
        title: optional title override.

    Returns:
        The path written.
    """
    oracle_by_key = _load_shine_scores(oracle_run_dir)
    embed_by_key = _load_shine_scores(embedding_run_dir)

    if paired:
        common = oracle_by_key.keys() & embed_by_key.keys()
        if not common:
            raise ValueError(
                "no overlapping (repo, document, qa) keys between the two runs"
            )
        oracle_scores = [oracle_by_key[k] for k in common]
        embed_scores = [embed_by_key[k] for k in common]
        sample_note = f"paired n = {len(common)}"
    else:
        oracle_scores = list(oracle_by_key.values())
        embed_scores = list(embed_by_key.values())
        sample_note = (
            f"n$_{{{oracle_label}}}$ = {len(oracle_scores)},  "
            f"n$_{{{embedding_label}}}$ = {len(embed_scores)}"
        )

    p_oracle = _proportions(oracle_scores)
    p_embed = _proportions(embed_scores)
    oracle_pp = np.array([100.0 * p_oracle[s] for s in SCORE_LEVELS])
    embed_pp = np.array([100.0 * p_embed[s] for s in SCORE_LEVELS])

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    x = np.arange(len(SCORE_LEVELS))
    width = 0.38

    bars_oracle = ax.bar(
        x - width / 2,
        oracle_pp,
        width=width,
        color=COLOR_ORACLE,
        edgecolor="white",
        linewidth=0.6,
        label=oracle_label,
        zorder=2,
    )
    bars_embed = ax.bar(
        x + width / 2,
        embed_pp,
        width=width,
        color=COLOR_EMBEDDING,
        edgecolor="white",
        linewidth=0.6,
        label=embedding_label,
        zorder=2,
    )

    ymax = float(max(oracle_pp.max(initial=0.0), embed_pp.max(initial=0.0)))
    ax.set_ylim(0, ymax * 1.18 + 2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SCORE_LEVELS])
    ax.set_xlabel("Judge score (1–5)")
    ax.set_ylabel("Proportion of SHINE answers (%)")

    for bars, vals in ((bars_oracle, oracle_pp), (bars_embed, embed_pp)):
        for bar, pp in zip(bars, vals):
            if pp <= 0.0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                pp + ymax * 0.015 + 0.3,
                f"{pp:.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#1a1a1a",
            )

    if title is None:
        title = "SHINE score distribution: oracle vs embedding routing"
    ax.set_title(title, fontsize=11, pad=8)
    ax.text(
        0.99,
        0.97,
        sample_note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLOR_TEXT_MUTED,
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.93),
        frameon=False,
        fontsize=9,
        handlelength=1.3,
        handleheight=0.9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(axis="both", which="major", labelsize=9, length=3, color="#444")
    ax.yaxis.grid(True, color=COLOR_GRID, alpha=0.45, linewidth=0.55, zorder=1)
    ax.set_axisbelow(True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)
    return out_path


def _load_shine_scores(run_dir: Path) -> dict[tuple[str, str, str], int]:
    """Map ``(repo_id, document_id, qa_id) -> judge score`` for SHINE rows."""
    return _load_condition_scores(run_dir, condition="shine")


def _load_condition_scores(
    run_dir: Path, *, condition: str
) -> dict[tuple[str, str, str], int]:
    """Map ``(repo_id, document_id, qa_id) -> judge score`` for one condition.

    The composite key matters: ``qa_id`` alone is *not* unique across the
    five-repo eval — synthetic ids like ``overview_purpose_..._0001`` are
    reused per repo, so collapsing on ``qa_id`` silently merges different
    questions and shrinks the paired set.
    """
    src = Path(run_dir) / "judgments.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")
    out: dict[tuple[str, str, str], int] = {}
    for lineno, raw in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {src}:{lineno}: {exc}") from exc
        if row.get("condition") != condition:
            continue
        repo_id = row.get("repo_id") or ""
        document_id = row.get("document_id") or ""
        qa_id = row.get("qa_id") or ""
        score = (row.get("judge") or {}).get("score")
        if not (repo_id and document_id and qa_id) or not isinstance(score, int):
            continue
        if 1 <= score <= 5:
            out.setdefault((repo_id, document_id, qa_id), score)
    if not out:
        raise ValueError(f"no {condition!r} rows with valid scores in {src}")
    return out


def _proportions(scores: Iterable[int]) -> dict[int, float]:
    counter = Counter(scores)
    n = sum(counter.values())
    if n == 0:
        return {s: 0.0 for s in SCORE_LEVELS}
    return {s: counter.get(s, 0) / n for s in SCORE_LEVELS}


def render_per_repo_means(
    *,
    oracle_run_dir: Path,
    embedding_run_dir: Path,
    out_path: Path,
    paired: bool = True,
    oracle_label: str = "oracle",
    embedding_label: str = "embedding (top-1)",
    naive_label: str = "naive",
    show_naive: bool = False,
    title: str | None = None,
) -> Path:
    """Per-repo grouped bars: mean SHINE score for oracle vs embedding.

    Computes the mean judge score on the ``shine`` condition for each
    repository in both runs and plots them side-by-side. When
    ``show_naive=True`` a third bar per repo is added for the ``naive``
    condition (sourced from ``embedding_run_dir`` since naive answers
    are routing-independent).

    When ``paired=True`` (default) the comparison is restricted to
    ``(repo_id, document_id, qa_id)`` triples present in *both* runs, so
    each repo's bars are computed over the same question set.
    """
    oracle_by_key = _load_shine_scores(oracle_run_dir)
    embed_by_key = _load_shine_scores(embedding_run_dir)
    naive_by_key = (
        _load_condition_scores(embedding_run_dir, condition="naive")
        if show_naive else {}
    )

    if paired:
        common = oracle_by_key.keys() & embed_by_key.keys()
        if not common:
            raise ValueError(
                "no overlapping (repo, document, qa) keys between the two runs"
            )
        oracle_by_key = {k: oracle_by_key[k] for k in common}
        embed_by_key = {k: embed_by_key[k] for k in common}
        if show_naive:
            naive_by_key = {k: naive_by_key[k] for k in common if k in naive_by_key}

    oracle_per_repo: dict[str, list[int]] = defaultdict(list)
    embed_per_repo: dict[str, list[int]] = defaultdict(list)
    naive_per_repo: dict[str, list[int]] = defaultdict(list)
    for (repo_id, _, _), score in oracle_by_key.items():
        oracle_per_repo[repo_id].append(score)
    for (repo_id, _, _), score in embed_by_key.items():
        embed_per_repo[repo_id].append(score)
    for (repo_id, _, _), score in naive_by_key.items():
        naive_per_repo[repo_id].append(score)

    repos = sorted(set(oracle_per_repo) | set(embed_per_repo))
    if not repos:
        raise ValueError("no repos with SHINE scores in either run")

    o_means, _, o_ns = zip(
        *(_mean_ci_n(oracle_per_repo.get(r, [])) for r in repos)
    )
    e_means, _, e_ns = zip(
        *(_mean_ci_n(embed_per_repo.get(r, [])) for r in repos)
    )
    if show_naive:
        n_means, _, n_ns = zip(
            *(_mean_ci_n(naive_per_repo.get(r, [])) for r in repos)
        )
    else:
        n_means = n_ns = ()

    fig_w = max(5.0, 0.9 + 1.0 * len(repos))
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    x = np.arange(len(repos))

    if show_naive:
        # 3 bars per group: naive | oracle | embedding
        width = 0.27
        offsets = (-width, 0.0, width)
        series = (
            (offsets[0], n_means, COLOR_NAIVE, naive_label),
            (offsets[1], o_means, COLOR_ORACLE, oracle_label),
            (offsets[2], e_means, COLOR_EMBEDDING, embedding_label),
        )
    else:
        width = 0.38
        series = (
            (-width / 2, o_means, COLOR_ORACLE, oracle_label),
            (+width / 2, e_means, COLOR_EMBEDDING, embedding_label),
        )

    bar_groups: list = []
    for offset, means, color, label in series:
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=2,
        )
        bar_groups.append((bars, means))

    ax.set_ylim(0, 5.4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_short_repo_label(r) for r in repos], rotation=15, ha="right"
    )
    ax.set_xlabel("Repository")
    ax.set_ylabel("Mean judge score")

    for bars, means in bar_groups:
        for bar, m in zip(bars, means):
            if m <= 0.0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.10,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0 if show_naive else 8.5,
                color="#1a1a1a",
            )

    if title is None:
        title = "Mean SHINE score per repository: oracle vs embedding routing"
    ax.set_title(title, fontsize=11, pad=8)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 0.93),
        frameon=False,
        fontsize=9,
        handlelength=1.3,
        handleheight=0.9,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(axis="both", which="major", labelsize=9, length=3, color="#444")
    ax.yaxis.grid(True, color=COLOR_GRID, alpha=0.45, linewidth=0.55, zorder=1)
    ax.set_axisbelow(True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("wrote %s", out_path)
    return out_path


def _mean_ci_n(scores: list[int]) -> tuple[float, float, int]:
    n = len(scores)
    if n == 0:
        return 0.0, 0.0, 0
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    sd = math.sqrt(var)
    ci = 1.96 * sd / math.sqrt(n)
    return mean, ci, n


def _short_repo_label(repo_id: str) -> str:
    """``antirez__kilo`` → ``kilo`` for tighter axis labels."""
    return repo_id.split("__", 1)[-1] if "__" in repo_id else repo_id


__all__ = ["render_routing_score_delta", "render_per_repo_means"]
