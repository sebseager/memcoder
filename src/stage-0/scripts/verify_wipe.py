#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
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


def verify_reports_dir() -> Path:
    out_dir = OUTPUTS_DIR / "verify_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def move_report_to_outputs(model_name: str, run_id: str) -> str:
    src_name = f"{model_name.replace('/', '__')}.{run_id}.json"
    src_path = Path(src_name)
    if not src_path.exists():
        return ""

    dst_path = verify_reports_dir() / src_path.name
    if dst_path.exists():
        dst_path.unlink()
    src_path.replace(dst_path)
    return str(dst_path)


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse existing outputs/04_verify_runs.jsonl and "
            "outputs/04_verified_instances.jsonl to avoid rerunning completed attempts."
        ),
    )
    parser.add_argument(
        "--retry-missing-only",
        action="store_true",
        help=(
            "With --resume, retry attempts that previously had no per-instance report "
            "(typically infrastructure failures)."
        ),
    )
    parser.add_argument(
        "--docker-config-mode",
        choices=["safe", "inherit"],
        default="safe",
        help=(
            "Docker auth config mode for harness subprocesses. 'safe' writes a minimal "
            "DOCKER_CONFIG under outputs to avoid broken credential helper binaries."
        ),
    )
    return parser.parse_args()


def normalize_rank(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def build_harness_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    if mode != "safe":
        return env

    safe_dir = OUTPUTS_DIR / ".docker-safe-config"
    safe_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = safe_dir / "config.json"

    # Intentionally omit credsStore/credHelpers so docker SDK never calls host helpers.
    payload: dict[str, dict] = {"auths": {}}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        auths = existing.get("auths")
        if isinstance(auths, dict):
            payload["auths"] = auths

    cfg_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    env["DOCKER_CONFIG"] = str(safe_dir)
    return env


def load_resume_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def completed_pairs_from_runs(
    verify_rows: list[dict], *, retry_missing_only: bool
) -> set[tuple[str, int]]:
    pair_has_report: dict[tuple[str, int], bool] = {}
    for row in verify_rows:
        iid = row.get("instance_id")
        rank = normalize_rank(row.get("rank"))
        if not iid or rank < 1:
            continue
        key = (iid, rank)
        report_exists = bool(row.get("report_exists"))
        pair_has_report[key] = pair_has_report.get(key, False) or report_exists

    if retry_missing_only:
        return {key for key, has_report in pair_has_report.items() if has_report}
    return set(pair_has_report.keys())


def selected_from_verified_rows(rows: list[dict]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for row in rows:
        iid = row.get("instance_id")
        if iid:
            selected[iid] = row
    return selected


def selected_from_verify_rows(
    verify_rows: list[dict], attempts_by_pair: dict[tuple[str, int], dict]
) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    accepted_rows = [row for row in verify_rows if row.get("accepted")]
    accepted_rows.sort(
        key=lambda row: (normalize_rank(row.get("rank")), row.get("instance_id", ""))
    )

    for row in accepted_rows:
        iid = row.get("instance_id")
        rank = normalize_rank(row.get("rank"))
        if not iid or rank < 1 or iid in selected:
            continue
        attempt = attempts_by_pair.get((iid, rank))
        if attempt is None:
            continue
        selected[iid] = {
            **attempt,
            "verify_rank": rank,
            "verify_run_id": row.get("run_id"),
            "verify_model_name": row.get("model_name"),
            "verify_patch_applied": bool(row.get("patch_successfully_applied")),
            "verify_resolved": bool(row.get("resolved")),
        }

    return selected


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

    output_runs = OUTPUTS_DIR / "04_verify_runs.jsonl"
    output_verified = OUTPUTS_DIR / "04_verified_instances.jsonl"
    output_summary = OUTPUTS_DIR / "04_verify_summary.json"

    attempts = read_jsonl(Path(args.attempts_jsonl))
    grouped: dict[str, list[dict]] = defaultdict(list)
    attempts_by_pair: dict[tuple[str, int], dict] = {}
    for row in attempts:
        iid = row["instance_id"]
        rank = normalize_rank(row.get("rank"))
        grouped[iid].append(row)
        if rank > 0:
            attempts_by_pair[(iid, rank)] = row

    for rows in grouped.values():
        rows.sort(key=lambda x: x["rank"])

    verify_rows: list[dict] = []
    completed_pairs: set[tuple[str, int]] = set()
    selected: dict[str, dict] = {}

    if args.resume:
        prior_verify_rows = load_resume_rows(output_runs)
        verify_rows.extend(prior_verify_rows)
        completed_pairs = completed_pairs_from_runs(
            prior_verify_rows,
            retry_missing_only=args.retry_missing_only,
        )

        prior_selected_rows = load_resume_rows(output_verified)
        selected = selected_from_verified_rows(prior_selected_rows)
        if not selected and prior_verify_rows:
            selected = selected_from_verify_rows(prior_verify_rows, attempts_by_pair)

    remaining = {iid for iid in grouped.keys() if iid not in selected}

    max_rank = max((row["rank"] for row in attempts), default=0)
    harness_env = build_harness_env(args.docker_config_mode)

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
            if candidate is None:
                continue
            if args.resume and (iid, rank) in completed_pairs:
                continue
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

            proc = subprocess.run(cmd, check=False, env=harness_env)
            run_exit = proc.returncode
            run_report_json = move_report_to_outputs(
                model_name=model_name, run_id=run_id
            )

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
                        "run_report_json": run_report_json,
                        "report_exists": report_exists,
                        "patch_successfully_applied": patch_applied,
                        "resolved": resolved,
                        "accepted": accepted,
                    }
                )

                if report_exists:
                    completed_pairs.add((iid, rank))

        remaining = {iid for iid in remaining if iid not in selected}

    selected_rows = sorted(
        selected.values(), key=lambda row: (row["verify_rank"], row["instance_id"])
    )

    write_jsonl(output_runs, verify_rows)
    write_jsonl(output_verified, selected_rows)
    write_json(
        output_summary,
        {
            "attempt_instances": len(grouped),
            "selected_instances": len(selected_rows),
            "target_final_instances": args.target_final_instances,
            "max_rank_considered": max_rank,
            "resume": {
                "enabled": bool(args.resume),
                "retry_missing_only": bool(args.retry_missing_only),
                "docker_config_mode": args.docker_config_mode,
            },
            "outputs": {
                "verify_runs_jsonl": str(output_runs),
                "verified_instances_jsonl": str(output_verified),
                "verify_reports_dir": str(verify_reports_dir()),
            },
        },
    )

    print(f"Verified instances selected: {len(selected_rows)}")


if __name__ == "__main__":
    main()
