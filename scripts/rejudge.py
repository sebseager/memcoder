#!/usr/bin/env python3
"""Re-judge existing predictions against a different rubric prompt.

Writes a sidecar judgments file (default ``judgments_v1.jsonl``) into each
specified run directory, leaving the original ``judgments.jsonl`` untouched.
This is the safe way to evaluate a rubric change without losing the v0
grades.

Usage:

    python scripts/rejudge.py \\
        --prompt prompts/llm_judge_grading_v1.md \\
        --rubric-version v1 \\
        --taxonomy-version v1 \\
        --run-dir results/marimo_easy_v0_20260427T1526 \\
        --run-dir results/marimo_easy_v0_detail_20260427T1720 \\
        --run-dir results/marimo_easy_v0_adapted_20260427T1728 \\
        --run-dir results/kilo_easy_v0_20260427T0138 \\
        --run-dir results/kilo_easy_v0_detail_20260427T1731 \\
        --run-dir results/kilo_easy_v0_adapted_20260427T1747

Each run dir must contain a ``predictions.jsonl`` and a ``run_config.yaml``
(used to source the OpenAI model + dotenv path + concurrency etc.). Outputs
land at ``<run-dir>/<output-name>``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.config import load_snapshot  # noqa: E402
from eval.judge import judge_predictions_to  # noqa: E402

LOGGER = logging.getLogger("memcoder.rejudge")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="Existing run directory containing predictions.jsonl and "
             "run_config.yaml. Repeat for multiple dirs.",
    )
    p.add_argument(
        "--prompt",
        type=Path,
        default=REPO_ROOT / "prompts" / "llm_judge_grading_v1.md",
        help="Judge prompt to use (default: prompts/llm_judge_grading_v1.md).",
    )
    p.add_argument(
        "--rubric-version",
        type=str,
        default="v1",
        help="rubric_version to record in the judgments output (default: v1).",
    )
    p.add_argument(
        "--taxonomy-version",
        type=str,
        default="v1",
        help="taxonomy_version to record in the judgments output (default: v1).",
    )
    p.add_argument(
        "--output-name",
        type=str,
        default="judgments_v1.jsonl",
        help="Filename for the sidecar judgments under each run dir "
             "(default: judgments_v1.jsonl). The script refuses to clobber "
             "existing files at this path.",
    )
    return p.parse_args()


def main() -> int:
    _setup_logging()
    args = parse_args()

    if not args.prompt.exists():
        raise FileNotFoundError(f"prompt not found: {args.prompt}")

    failures: list[tuple[Path, str]] = []
    successes: list[Path] = []

    for run_dir in args.run_dir:
        if not run_dir.exists():
            LOGGER.error("run dir not found: %s", run_dir)
            failures.append((run_dir, "not found"))
            continue

        try:
            cfg = load_snapshot(run_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("failed to load snapshot for %s: %s", run_dir, exc)
            failures.append((run_dir, f"snapshot load failed: {exc}"))
            continue

        predictions_path = run_dir / "predictions.jsonl"
        output_path = run_dir / args.output_name

        try:
            judge_predictions_to(
                predictions_path=predictions_path,
                output_path=output_path,
                judge_cfg=cfg.judge,
                prompt_path=args.prompt,
                rubric_version=args.rubric_version,
                taxonomy_version=args.taxonomy_version,
            )
            successes.append(output_path)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("rejudge failed for %s: %s", run_dir, exc)
            failures.append((run_dir, str(exc)))

    print()
    print("=" * 72)
    if successes:
        print(f"Rejudged {len(successes)} run dir(s):")
        for p in successes:
            print(f"  {p}")
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for run_dir, reason in failures:
            print(f"  {run_dir}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
