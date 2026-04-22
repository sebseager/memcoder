#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from config import (
    DATASET_NAME,
    DEFAULT_DOCKER_NAMESPACE,
    DEFAULT_TARGET_FINAL_INSTANCES,
    DEFAULT_VERIFY_MAX_WORKERS,
    DEFAULT_VERIFY_TIMEOUT_SECONDS,
    OUTPUTS_DIR,
    ensure_stage0_dirs,
)
from patch_utils import combine_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wipe verification via SWE-rebench harness."
    )
    parser.add_argument(
        "--attempts-jsonl", default=str(OUTPUTS_DIR / "03_candidate_attempts.jsonl")
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--namespace", default=DEFAULT_DOCKER_NAMESPACE)
    parser.add_argument("--cache-level", default="instance")
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_VERIFY_TIMEOUT_SECONDS
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_VERIFY_MAX_WORKERS)
    parser.add_argument(
        "--target-final-instances", type=int, default=DEFAULT_TARGET_FINAL_INSTANCES
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Instances per harness invocation at each rank (0 means all remaining at that rank).",
    )
    parser.add_argument("--run-id-prefix", default="stage0-verify")
    return parser.parse_args()


def read_report(run_id: str, model_name: str, instance_id: str) -> dict | None:
    report_path = (
        Path("logs")
        / "run_evaluation"
        / run_id
        / model_name
        / instance_id
        / "report.json"
    )
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload.get(instance_id)


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    attempts = read_jsonl(Path(args.attempts_jsonl))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        grouped[row["instance_id"]].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda x: x["rank"])

    remaining = set(grouped.keys())
    selected: dict[str, dict] = {}
    verify_rows: list[dict] = []

    max_rank = max((row["rank"] for row in attempts), default=0)

    for rank in range(1, max_rank + 1):
        if not remaining:
            break
        if len(selected) >= args.target_final_instances:
            break

        ranked_candidates: list[dict] = []
        for iid in sorted(remaining):
            candidate = next(
                (row for row in grouped[iid] if int(row["rank"]) == rank), None
            )
            if candidate is not None:
                ranked_candidates.append(candidate)

        if not ranked_candidates:
            continue

        chunk_size = args.batch_size if args.batch_size > 0 else len(ranked_candidates)

        for offset in range(0, len(ranked_candidates), chunk_size):
            if len(selected) >= args.target_final_instances:
                break

            batch_attempts = ranked_candidates[offset : offset + chunk_size]
            if not batch_attempts:
                continue

            model_name = f"stage0-wipe-r{rank}"
            run_id = f"{args.run_id_prefix}-r{rank}-{int(time.time())}"

            predictions = []
            instance_ids = []
            for candidate in batch_attempts:
                iid = candidate["instance_id"]
                patch = combine_patches(
                    candidate.get("gold_patch", ""), candidate.get("mask_patch", "")
                )
                predictions.append(
                    {
                        "instance_id": iid,
                        "model_name_or_path": model_name,
                        "model_patch": patch,
                    }
                )
                instance_ids.append(iid)

            pred_path = OUTPUTS_DIR / f"04_predictions_rank_{rank}_{offset}.jsonl"
            write_jsonl(pred_path, predictions)

            cmd = [
                "python",
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                args.dataset_name,
                "--predictions_path",
                str(pred_path),
                "--instance_ids",
                *instance_ids,
                "--cache_level",
                args.cache_level,
                "--run_id",
                run_id,
                "--namespace",
                args.namespace,
                "--timeout",
                str(args.timeout_seconds),
                "--max_workers",
                str(args.max_workers),
            ]

            proc = subprocess.run(cmd, check=False)
            run_exit = proc.returncode

            for attempt in batch_attempts:
                iid = attempt["instance_id"]
                report = read_report(
                    run_id=run_id,
                    model_name=model_name,
                    instance_id=iid,
                )
                report_exists = report is not None
                patch_applied = (
                    bool(report.get("patch_successfully_applied")) if report else False
                )
                resolved = bool(report.get("resolved")) if report else False

                # Keep only attempts where gold+wipe was applied and no longer resolves.
                accepted = patch_applied and (not resolved)
                if accepted and iid not in selected:
                    selected[iid] = {
                        **attempt,
                        "verify_rank": rank,
                        "verify_run_id": run_id,
                        "verify_model_name": model_name,
                        "verify_patch_applied": patch_applied,
                        "verify_resolved": resolved,
                    }

                verify_rows.append(
                    {
                        "instance_id": iid,
                        "rank": rank,
                        "run_id": run_id,
                        "model_name": model_name,
                        "run_exit_code": run_exit,
                        "report_exists": report_exists,
                        "patch_successfully_applied": patch_applied,
                        "resolved": resolved,
                        "accepted": accepted,
                    }
                )

        remaining = {iid for iid in remaining if iid not in selected}

    selected_rows = sorted(
        selected.values(), key=lambda row: (row["verify_rank"], row["instance_id"])
    )

    output_runs = OUTPUTS_DIR / "04_verify_runs.jsonl"
    output_verified = OUTPUTS_DIR / "04_verified_instances.jsonl"
    output_summary = OUTPUTS_DIR / "04_verify_summary.json"

    write_jsonl(output_runs, verify_rows)
    write_jsonl(output_verified, selected_rows)
    write_json(
        output_summary,
        {
            "attempt_instances": len(grouped),
            "selected_instances": len(selected_rows),
            "target_final_instances": args.target_final_instances,
            "max_rank_considered": max_rank,
            "outputs": {
                "verify_runs_jsonl": str(output_runs),
                "verified_instances_jsonl": str(output_verified),
            },
        },
    )

    print(f"Verified instances selected: {len(selected_rows)}")


if __name__ == "__main__":
    main()
