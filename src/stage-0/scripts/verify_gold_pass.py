#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from config import (
    DATASET_NAME,
    DEFAULT_DOCKER_NAMESPACE,
    DEFAULT_VERIFY_MAX_WORKERS,
    DEFAULT_VERIFY_TIMEOUT_SECONDS,
    OUTPUTS_DIR,
    ensure_stage0_dirs,
)


def gold_reports_dir() -> Path:
    out_dir = OUTPUTS_DIR / "gold_verify_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def move_report_to_outputs(model_name: str, run_id: str) -> str:
    src_name = f"{model_name.replace('/', '__')}.{run_id}.json"
    src_path = Path(src_name)
    if not src_path.exists():
        return ""

    dst_path = gold_reports_dir() / src_path.name
    if dst_path.exists():
        dst_path.unlink()
    src_path.replace(dst_path)
    return str(dst_path)


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


def build_harness_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    if mode != "safe":
        return env

    safe_dir = OUTPUTS_DIR / ".docker-safe-config"
    safe_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = safe_dir / "config.json"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Stage 0 gold-only sanity check. Verifies every selected "
            "instance resolves in SWE-rebench harness when applying gold patch only."
        )
    )
    parser.add_argument(
        "--verified-jsonl", default=str(OUTPUTS_DIR / "04_verified_instances.jsonl")
    )
    parser.add_argument(
        "--instances-jsonl", default=str(OUTPUTS_DIR / "instances.jsonl")
    )
    parser.add_argument(
        "--final-summary-json", default=str(OUTPUTS_DIR / "05_final_summary.json")
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--namespace", default=DEFAULT_DOCKER_NAMESPACE)
    parser.add_argument("--cache-level", default="instance")
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_VERIFY_TIMEOUT_SECONDS
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_VERIFY_MAX_WORKERS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Instances per harness invocation (0 means all selected instances at once).",
    )
    parser.add_argument("--run-id-prefix", default="stage0-goldcheck")
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


def load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def validate_prereqs(
    *,
    verified_path: Path,
    instances_path: Path,
    final_summary_path: Path,
) -> tuple[list[dict], list[dict], dict]:
    missing = [
        str(path)
        for path in (verified_path, instances_path, final_summary_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Stage 0 prerequisites missing. Run stage-0 pipeline first. Missing: "
            + ", ".join(missing)
        )

    verified_rows = read_jsonl(verified_path)
    instance_rows = read_jsonl(instances_path)
    summary = load_json_file(final_summary_path)

    if not isinstance(summary, dict):
        raise ValueError(f"Malformed summary JSON: {final_summary_path}")

    expected_final = int(summary.get("final_instance_count", 0))
    if expected_final <= 0:
        raise ValueError(
            "final_instance_count <= 0 in 05_final_summary.json. "
            "Run finalize_instances.py first."
        )

    if len(instance_rows) != expected_final:
        raise ValueError(
            "instances.jsonl count does not match 05_final_summary.json: "
            f"{len(instance_rows)} != {expected_final}"
        )

    if len(verified_rows) < expected_final:
        raise ValueError(
            "04_verified_instances.jsonl has fewer rows than finalized instances: "
            f"{len(verified_rows)} < {expected_final}"
        )

    instance_ids = {row.get("instance_id") for row in instance_rows}
    verified_ids = {row.get("instance_id") for row in verified_rows}
    missing_in_verified = sorted(
        [iid for iid in instance_ids if iid not in verified_ids]
    )
    if missing_in_verified:
        raise ValueError(
            "Some finalized instances are missing in 04_verified_instances.jsonl: "
            + ", ".join(missing_in_verified[:10])
            + (" ..." if len(missing_in_verified) > 10 else "")
        )

    return verified_rows, instance_rows, summary


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    verified_path = Path(args.verified_jsonl)
    instances_path = Path(args.instances_jsonl)
    final_summary_path = Path(args.final_summary_json)

    verified_rows, instance_rows, _ = validate_prereqs(
        verified_path=verified_path,
        instances_path=instances_path,
        final_summary_path=final_summary_path,
    )

    selected_ids = [row["instance_id"] for row in instance_rows]
    selected_id_set = set(selected_ids)

    by_id = {row.get("instance_id"): row for row in verified_rows}
    selected_rows: list[dict] = []
    for iid in selected_ids:
        row = by_id.get(iid)
        if row is None:
            raise ValueError(f"Missing selected instance in verified rows: {iid}")
        patch = (row.get("gold_patch") or "").strip()
        if not patch:
            raise ValueError(f"Missing gold_patch for selected instance: {iid}")
        selected_rows.append(row)

    batch_size = args.batch_size if args.batch_size > 0 else len(selected_rows)
    env = build_harness_env(args.docker_config_mode)

    verify_rows: list[dict] = []

    for offset in range(0, len(selected_rows), batch_size):
        batch_rows = selected_rows[offset : offset + batch_size]
        if not batch_rows:
            continue

        run_id = f"{args.run_id_prefix}-{int(time.time())}-{offset}"
        model_name = "stage0-gold-pass"

        predictions = []
        instance_ids = []
        for row in batch_rows:
            iid = row["instance_id"]
            predictions.append(
                {
                    "instance_id": iid,
                    "model_name_or_path": model_name,
                    "model_patch": row["gold_patch"],
                }
            )
            instance_ids.append(iid)

        pred_path = OUTPUTS_DIR / f"07_gold_predictions_{offset}.jsonl"
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

        proc = subprocess.run(cmd, check=False, env=env)
        run_exit = proc.returncode
        run_report_json = move_report_to_outputs(model_name=model_name, run_id=run_id)

        for row in batch_rows:
            iid = row["instance_id"]
            report = read_report(run_id=run_id, model_name=model_name, instance_id=iid)
            report_exists = report is not None
            patch_applied = (
                bool(report.get("patch_successfully_applied")) if report else False
            )
            resolved = bool(report.get("resolved")) if report else False
            passed = patch_applied and resolved

            verify_rows.append(
                {
                    "instance_id": iid,
                    "run_id": run_id,
                    "model_name": model_name,
                    "run_exit_code": run_exit,
                    "run_report_json": run_report_json,
                    "report_exists": report_exists,
                    "patch_successfully_applied": patch_applied,
                    "resolved": resolved,
                    "passed": passed,
                }
            )

    output_runs = OUTPUTS_DIR / "07_gold_verify_runs.jsonl"
    output_summary = OUTPUTS_DIR / "07_gold_verify_summary.json"

    write_jsonl(output_runs, verify_rows)

    by_iid: dict[str, dict] = {}
    for row in verify_rows:
        iid = row["instance_id"]
        prev = by_iid.get(iid)
        if prev is None:
            by_iid[iid] = row
            continue
        if not prev.get("report_exists") and row.get("report_exists"):
            by_iid[iid] = row
            continue
        if not prev.get("passed") and row.get("passed"):
            by_iid[iid] = row

    missing_rows = [iid for iid in selected_id_set if iid not in by_iid]
    failed = []
    missing_report = []
    not_resolved = []
    patch_not_applied = []

    for iid in selected_ids:
        row = by_iid.get(iid)
        if row is None:
            missing_report.append(iid)
            continue
        if not row.get("report_exists"):
            missing_report.append(iid)
            continue
        if not row.get("patch_successfully_applied"):
            patch_not_applied.append(iid)
            failed.append(iid)
            continue
        if not row.get("resolved"):
            not_resolved.append(iid)
            failed.append(iid)

    missing_report.extend(missing_rows)
    unique_failed = sorted(set(failed + missing_report))
    passed_count = len(selected_id_set) - len(unique_failed)
    all_passed = len(unique_failed) == 0

    summary = {
        "selected_instances": len(selected_ids),
        "passed_instances": passed_count,
        "failed_instances": len(unique_failed),
        "all_passed": all_passed,
        "failures": {
            "missing_report": sorted(set(missing_report)),
            "patch_not_applied": sorted(set(patch_not_applied)),
            "not_resolved": sorted(set(not_resolved)),
        },
        "run_config": {
            "dataset_name": args.dataset_name,
            "namespace": args.namespace,
            "cache_level": args.cache_level,
            "timeout_seconds": args.timeout_seconds,
            "max_workers": args.max_workers,
            "batch_size": batch_size,
            "docker_config_mode": args.docker_config_mode,
        },
        "inputs": {
            "verified_jsonl": str(verified_path),
            "instances_jsonl": str(instances_path),
            "final_summary_json": str(final_summary_path),
        },
        "outputs": {
            "gold_verify_runs_jsonl": str(output_runs),
            "gold_verify_summary_json": str(output_summary),
            "gold_verify_reports_dir": str(gold_reports_dir()),
        },
    }
    write_json(output_summary, summary)

    print(
        "Gold pass verification: "
        f"{passed_count}/{len(selected_ids)} passed "
        f"(failed={len(unique_failed)})."
    )

    if not all_passed:
        print(
            "ERROR: Not all selected instances passed gold-only verification. "
            f"See {output_summary}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
