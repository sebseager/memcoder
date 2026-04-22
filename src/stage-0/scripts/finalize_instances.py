#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import read_jsonl, write_csv, write_json, write_jsonl
from config import OUTPUTS_DIR, REPOS_CACHE_DIR, ensure_stage0_dirs
from function_masking import mask_function_by_position
from patch_utils import write_text
from repo_utils import ensure_repo_checkout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize Stage 0 retained instances and Stage-1 input artifacts."
    )
    parser.add_argument(
        "--verified-jsonl",
        default=str(OUTPUTS_DIR / "05_gold_passed_verified_instances.jsonl"),
        help="Gold-pass filtered verified instances.",
    )
    parser.add_argument(
        "--wipe-verified-jsonl",
        default=str(OUTPUTS_DIR / "04_verified_instances.jsonl"),
        help="verify_wipe accepted instances (for provenance fields).",
    )
    parser.add_argument(
        "--gold-runs-jsonl",
        default=str(OUTPUTS_DIR / "05_gold_verify_runs.jsonl"),
        help="Gold verification run rows (for per-instance pass metadata).",
    )
    parser.add_argument(
        "--gold-summary-json",
        default=str(OUTPUTS_DIR / "05_gold_verify_summary.json"),
        help="Gold verification summary JSON.",
    )
    return parser.parse_args()


def load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def select_best_gold_rows(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        iid = row.get("instance_id")
        if not iid:
            continue

        prev = best.get(iid)
        if prev is None:
            best[iid] = row
            continue

        prev_report = bool(prev.get("report_exists"))
        row_report = bool(row.get("report_exists"))
        prev_passed = bool(prev.get("passed"))
        row_passed = bool(row.get("passed"))

        if (not prev_report and row_report) or (not prev_passed and row_passed):
            best[iid] = row

    return best


def get_field(primary: dict, secondary: dict, key: str):
    value = primary.get(key)
    if value is None:
        return secondary.get(key)
    return value


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    verified_path = Path(args.verified_jsonl)
    wipe_verified_path = Path(args.wipe_verified_jsonl)
    gold_runs_path = Path(args.gold_runs_jsonl)
    gold_summary_path = Path(args.gold_summary_json)

    rows = read_jsonl(verified_path)
    if not rows:
        raise ValueError(
            f"No rows found in finalized input: {verified_path}. "
            "Run verify_wipe + verify_gold_pass first."
        )

    wipe_rows = read_jsonl(wipe_verified_path)
    wipe_by_id = {
        row.get("instance_id"): row for row in wipe_rows if row.get("instance_id")
    }

    gold_runs = read_jsonl(gold_runs_path)
    gold_by_id = select_best_gold_rows(gold_runs)
    gold_summary = load_json_file(gold_summary_path)

    artifacts_root = OUTPUTS_DIR / "instances"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    # Replace artifacts in place on reruns.
    for existing in artifacts_root.iterdir():
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()

    final_rows: list[dict] = []
    csv_rows: list[dict] = []

    for row in rows:
        instance_id = row["instance_id"]
        wipe_row = wipe_by_id.get(instance_id, {})
        gold_row = gold_by_id.get(instance_id, {})

        repo = get_field(row, wipe_row, "repo")
        base_commit = get_field(row, wipe_row, "base_commit")
        file_path = get_field(row, wipe_row, "file_path")
        start_line = get_field(row, wipe_row, "start_line")
        col_offset = get_field(row, wipe_row, "col_offset")

        if not repo or not base_commit or not file_path:
            raise ValueError(
                "Missing required repo/base_commit/file_path for instance: "
                f"{instance_id}"
            )
        if start_line is None or col_offset is None:
            raise ValueError(
                f"Missing required start_line/col_offset for instance: {instance_id}"
            )

        repo_dir = ensure_repo_checkout(str(repo), str(base_commit), REPOS_CACHE_DIR)
        abs_path = repo_dir / str(file_path)
        source_text = abs_path.read_text(encoding="utf-8")

        masked = mask_function_by_position(
            source_text,
            lineno=int(start_line),
            col_offset=int(col_offset),
        )

        inst_dir = artifacts_root / instance_id
        inst_dir.mkdir(parents=True, exist_ok=True)

        full_file_path = inst_dir / "full_file.py"
        masked_file_path = inst_dir / "masked_file.py"
        ground_truth_path = inst_dir / "ground_truth_body.py"
        function_source_path = inst_dir / "function_source.py"
        mask_patch_path = inst_dir / "mask_patch.diff"
        metadata_path = inst_dir / "metadata.json"

        write_text(full_file_path, source_text)
        write_text(masked_file_path, masked.masked_file_text)
        write_text(ground_truth_path, masked.ground_truth_body)
        write_text(function_source_path, masked.function_source)
        write_text(mask_patch_path, str(get_field(row, wipe_row, "mask_patch") or ""))

        gold_patch_applied = bool(gold_row.get("patch_successfully_applied", True))
        gold_resolved = bool(gold_row.get("resolved", True))
        gold_passed = bool(gold_row.get("passed", True))

        metadata = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "environment_setup_commit": get_field(
                row, wipe_row, "environment_setup_commit"
            ),
            "created_at": get_field(row, wipe_row, "created_at"),
            "docker_image": get_field(row, wipe_row, "docker_image"),
            "image_name": get_field(row, wipe_row, "image_name"),
            "test_cmd": get_field(row, wipe_row, "test_cmd") or "",
            "file_path": file_path,
            "function_name": get_field(row, wipe_row, "function_name"),
            "start_line": start_line,
            "end_line": get_field(row, wipe_row, "end_line"),
            "body_line_count": get_field(row, wipe_row, "body_line_count"),
            "file_token_count": get_field(row, wipe_row, "file_token_count"),
            "external_reference_count": get_field(
                row, wipe_row, "external_reference_count"
            ),
            "external_references": get_field(row, wipe_row, "external_references")
            or [],
            "verify_rank": get_field(row, wipe_row, "verify_rank"),
            "verify_run_id": get_field(row, wipe_row, "verify_run_id"),
            "verify_resolved": get_field(row, wipe_row, "verify_resolved"),
            "verify_patch_applied": get_field(row, wipe_row, "verify_patch_applied"),
            "gold_verify_run_id": gold_row.get("run_id"),
            "gold_verify_report_exists": bool(gold_row.get("report_exists", True)),
            "gold_verify_patch_applied": gold_patch_applied,
            "gold_verify_resolved": gold_resolved,
            "gold_verify_passed": gold_passed,
            "verification": {
                "wipe": {
                    "rank": get_field(row, wipe_row, "verify_rank"),
                    "run_id": get_field(row, wipe_row, "verify_run_id"),
                    "patch_applied": get_field(row, wipe_row, "verify_patch_applied"),
                    "resolved": get_field(row, wipe_row, "verify_resolved"),
                },
                "gold": {
                    "run_id": gold_row.get("run_id"),
                    "report_exists": bool(gold_row.get("report_exists", True)),
                    "patch_applied": gold_patch_applied,
                    "resolved": gold_resolved,
                    "passed": gold_passed,
                },
            },
            # Keep the upstream benchmark fix patch inline in stage1_instances so
            # Stage 1 can evaluate from a single input file.
            "gold_patch": str(get_field(row, wipe_row, "gold_patch") or ""),
            "evaluation_contract": {
                "metric": "pass@1",
                "definition": (
                    "Complete masked function body, apply as patch to base_commit, "
                    "execute FAIL_TO_PASS tests in SWE-rebench harness, record pass/fail."
                ),
                "test_command": get_field(row, wipe_row, "test_cmd") or "",
                "bleu_fallback": False,
            },
        }
        write_json(metadata_path, metadata)

        final_row = {
            **metadata,
            "artifact_dir": str(inst_dir),
            "full_file_artifact": str(full_file_path),
            "masked_file_artifact": str(masked_file_path),
            "ground_truth_artifact": str(ground_truth_path),
            "function_source_artifact": str(function_source_path),
            "mask_patch_artifact": str(mask_patch_path),
        }
        final_rows.append(final_row)
        csv_rows.append(
            {
                "instance_id": instance_id,
                "repo": str(repo),
                "created_at": get_field(row, wipe_row, "created_at") or "",
                "file_path": str(file_path),
                "function_name": get_field(row, wipe_row, "function_name") or "",
                "body_line_count": get_field(row, wipe_row, "body_line_count") or 0,
                "file_token_count": get_field(row, wipe_row, "file_token_count") or 0,
                "external_reference_count": get_field(
                    row, wipe_row, "external_reference_count"
                )
                or 0,
                "verify_rank": get_field(row, wipe_row, "verify_rank") or "",
                "gold_verify_passed": gold_passed,
                "test_cmd": get_field(row, wipe_row, "test_cmd") or "",
            }
        )

    output_stage1_jsonl = OUTPUTS_DIR / "stage1_instances.jsonl"
    output_stage1_csv = OUTPUTS_DIR / "stage1_instances.csv"
    output_summary = OUTPUTS_DIR / "06_final_summary.json"

    write_jsonl(output_stage1_jsonl, final_rows)

    fieldnames = [
        "instance_id",
        "repo",
        "created_at",
        "file_path",
        "function_name",
        "body_line_count",
        "file_token_count",
        "external_reference_count",
        "verify_rank",
        "gold_verify_passed",
        "test_cmd",
    ]
    write_csv(output_stage1_csv, csv_rows, fieldnames=fieldnames)

    wipe_count = len(wipe_rows)
    gold_passed_count = len(rows)
    removed_by_gold = max(0, wipe_count - gold_passed_count)

    write_json(
        output_summary,
        {
            "final_instance_count": len(final_rows),
            "source_counts": {
                "wipe_verified_instances": wipe_count,
                "gold_passed_verified_instances": gold_passed_count,
                "removed_by_gold_verify": removed_by_gold,
            },
            "inputs": {
                "verified_jsonl": str(verified_path),
                "wipe_verified_jsonl": str(wipe_verified_path),
                "gold_runs_jsonl": str(gold_runs_path),
                "gold_summary_json": str(gold_summary_path),
            },
            "gold_verify_summary": gold_summary,
            "outputs": {
                "stage1_instances_jsonl": str(output_stage1_jsonl),
                "stage1_instances_csv": str(output_stage1_csv),
                "artifacts_root": str(artifacts_root),
            },
        },
    )

    print(f"Final Stage 0 instances (Stage-1 input): {len(final_rows)}")


if __name__ == "__main__":
    main()
