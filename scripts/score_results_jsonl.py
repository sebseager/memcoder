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
import logging
import shutil
import sys
from pathlib import Path

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
        from eval.plots import render_run_plots
        from eval.report import write_report

        _prepare_output_dir(output_dir, force=args.force)
        cfg = load_run_config(config_path)
        staged_predictions = output_dir / "predictions.jsonl"
        shutil.copyfile(results_jsonl, staged_predictions)
        cfg.snapshot(output_dir)

        judgments = run_judging(cfg, output_dir)
        report = write_report(output_dir)
        plots = render_run_plots(output_dir, show_empty=args.show_empty)
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
