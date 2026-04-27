#!/usr/bin/env python3
"""Judge and report on an existing results JSONL file.

The eval harness judge scores ``predictions.jsonl`` inside a run directory.
This wrapper adapts any compatible ``*.jsonl`` results file by staging it as
``predictions.jsonl``, running the existing judge handler, then generating the
standard report and plots.

Usage:
    python scripts/score_results_jsonl.py artifacts/.../lora_composition_results.jsonl \
        --config config/eval/kilo_easy_v0.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("memcoder.score_results_jsonl")

GENERATED_FILENAMES = (
    "predictions.jsonl",
    "judgments.jsonl",
    "report.md",
    "run_config.yaml",
    "scores_per_condition.png",
    "mean_score_per_condition.png",
    "failure_modes_per_condition.png",
    "per_document_heatmap.png",
    "combined_qa_mean_judge_score.png",
    "combined_qa_mean_token_f1.png",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results_jsonl",
        type=Path,
        help=(
            "Existing JSONL answer records to score. Rows must include at least "
            "question, expected_answer, and answer fields."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/eval/kilo_easy_v0.yaml"),
        help=(
            "Eval config providing judge settings (default: "
            "config/eval/kilo_easy_v0.yaml)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for predictions.jsonl, judgments.jsonl, report.md, and plots. "
            "Defaults to <results_jsonl stem>_scored next to the input file."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previously generated scoring files in --output-dir.",
    )
    parser.add_argument(
        "--show-empty",
        action="store_true",
        help="Render plots for empty built-in conditions as well.",
    )
    return parser.parse_args()


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _default_output_dir(results_jsonl: Path) -> Path:
    return results_jsonl.with_name(f"{results_jsonl.stem}_scored")


def _prepare_output_dir(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in GENERATED_FILENAMES if (output_dir / name).exists()]
    if existing and not force:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"output directory already contains generated scoring files: {paths}; "
            "pass --force to replace them"
        )
    for path in existing:
        path.unlink()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        rows.append(record)
    return rows


def _required_document_ids(record: dict[str, Any]) -> list[str]:
    value = record.get("required_document_ids")
    if value is None and isinstance(record.get("qa_metadata"), dict):
        value = record["qa_metadata"].get("required_document_ids")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _is_lora_condition(condition: str) -> bool:
    return condition == "composition" or condition == "individual" or condition.startswith("individual:")


def _condition_label(record: dict[str, Any]) -> str:
    condition = str(record.get("condition") or "")
    if condition == "individual" and record.get("adapter_document_id"):
        return f"individual:{record['adapter_document_id']}"
    return condition


def _condition_sort_key(condition: str) -> tuple[int, str]:
    if condition == "individual" or condition.startswith("individual:"):
        return (0, condition)
    if condition == "composition":
        return (1, condition)
    return (2, condition)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _plot_combined_metric(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    filename: str,
    metric_name: str,
    value_for_row: Any,
    ylabel: str,
    ylim: tuple[float, float],
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values_by_condition: dict[str, list[float]] = {}
    for record in rows:
        label = _condition_label(record)
        value = value_for_row(record)
        if not label or value is None:
            continue
        values_by_condition.setdefault(label, []).append(float(value))

    if not values_by_condition:
        return None

    conditions = sorted(values_by_condition, key=_condition_sort_key)
    means = [_mean(values_by_condition[condition]) for condition in conditions]
    counts = [len(values_by_condition[condition]) for condition in conditions]

    fig, ax = plt.subplots(figsize=(1.8 + 1.7 * len(conditions), 5.0))
    bars = ax.bar(
        range(len(conditions)),
        means,
        color=["#6f9fd8" if condition.startswith("individual") else "#5cb85c" for condition in conditions],
        edgecolor="white",
    )
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Combined QA {metric_name} by LoRA condition")
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, count in zip(bars, means, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ylim[1] - ylim[0]) * 0.015,
            f"{mean:.2f}\n(n={count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_combined_qa_plots(output_dir: Path) -> list[Path]:
    judgments_path = output_dir / "judgments.jsonl"
    rows = [
        record
        for record in _load_jsonl(judgments_path)
        if len(_required_document_ids(record)) > 1 and _is_lora_condition(str(record.get("condition") or ""))
    ]
    if not rows:
        LOGGER.info("No combined-QA LoRA rows found; skipping combined-QA plots")
        return []

    plot_paths = []
    judge_score = _plot_combined_metric(
        rows,
        output_dir=output_dir,
        filename="combined_qa_mean_judge_score.png",
        metric_name="mean judge score",
        value_for_row=lambda record: (record.get("judge") or {}).get("score"),
        ylabel="Mean judge score",
        ylim=(0.0, 5.5),
    )
    if judge_score is not None:
        plot_paths.append(judge_score)

    token_f1 = _plot_combined_metric(
        rows,
        output_dir=output_dir,
        filename="combined_qa_mean_token_f1.png",
        metric_name="mean token F1",
        value_for_row=lambda record: (record.get("scores") or {}).get("token_f1"),
        ylabel="Mean token F1",
        ylim=(0.0, 1.1),
    )
    if token_f1 is not None:
        plot_paths.append(token_f1)

    return plot_paths


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    results_jsonl = _resolve_repo_path(args.results_jsonl)
    config_path = _resolve_repo_path(args.config)
    output_dir = _resolve_repo_path(args.output_dir) if args.output_dir else _default_output_dir(results_jsonl)

    if not results_jsonl.exists():
        print(f"error: results JSONL not found: {results_jsonl}", file=sys.stderr)
        return 1

    try:
        from eval.config import load_run_config
        from eval.judge import run_judging
        from eval.plots import render_naive_filtered_plots, render_run_plots
        from eval.report import write_report

        _prepare_output_dir(output_dir, force=args.force)
        cfg = load_run_config(config_path)
        staged_predictions = output_dir / "predictions.jsonl"
        shutil.copyfile(results_jsonl, staged_predictions)
        cfg.snapshot(output_dir)

        judgments = run_judging(cfg, output_dir)
        report = write_report(output_dir)
        plots = render_run_plots(output_dir, show_empty=args.show_empty)
        plots.extend(render_naive_filtered_plots(output_dir))
        plots.extend(render_combined_qa_plots(output_dir))
    except (FileExistsError, FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(judgments)
    print(report)
    for path in plots:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
