#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import read_jsonl, write_csv, write_json, write_jsonl
from config import OUTPUTS_DIR, REPOS_CACHE_DIR, ensure_stage0_dirs
from function_masking import mask_function_by_position
from patch_utils import write_text
from repo_utils import ensure_repo_checkout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize Stage 0 retained instances and artifacts."
    )
    parser.add_argument(
        "--verified-jsonl", default=str(OUTPUTS_DIR / "04_verified_instances.jsonl")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    rows = read_jsonl(Path(args.verified_jsonl))
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
        repo = row["repo"]
        base_commit = row["base_commit"]
        file_path = row["file_path"]

        repo_dir = ensure_repo_checkout(repo, base_commit, REPOS_CACHE_DIR)
        abs_path = repo_dir / file_path
        source_text = abs_path.read_text(encoding="utf-8")

        masked = mask_function_by_position(
            source_text,
            lineno=int(row["start_line"]),
            col_offset=int(row["col_offset"]),
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
        write_text(mask_patch_path, row.get("mask_patch", ""))

        metadata = {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "environment_setup_commit": row.get("environment_setup_commit"),
            "created_at": row.get("created_at"),
            "docker_image": row.get("docker_image"),
            "image_name": row.get("image_name"),
            "test_cmd": row.get("test_cmd", ""),
            "file_path": file_path,
            "function_name": row.get("function_name"),
            "start_line": row.get("start_line"),
            "end_line": row.get("end_line"),
            "body_line_count": row.get("body_line_count"),
            "file_token_count": row.get("file_token_count"),
            "external_reference_count": row.get("external_reference_count"),
            "external_references": row.get("external_references") or [],
            "verify_rank": row.get("verify_rank"),
            "verify_run_id": row.get("verify_run_id"),
            "verify_resolved": row.get("verify_resolved"),
            "verify_patch_applied": row.get("verify_patch_applied"),
            "evaluation_contract": {
                "metric": "pass@1",
                "definition": (
                    "Complete masked function body, apply as patch to base_commit, "
                    "execute FAIL_TO_PASS tests in SWE-rebench harness, record pass/fail."
                ),
                "test_command": row.get("test_cmd", ""),
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
                "repo": repo,
                "created_at": row.get("created_at", ""),
                "file_path": file_path,
                "function_name": row.get("function_name", ""),
                "body_line_count": row.get("body_line_count", 0),
                "file_token_count": row.get("file_token_count", 0),
                "external_reference_count": row.get("external_reference_count", 0),
                "verify_rank": row.get("verify_rank", ""),
                "test_cmd": row.get("test_cmd", ""),
            }
        )

    output_jsonl = OUTPUTS_DIR / "instances.jsonl"
    output_csv = OUTPUTS_DIR / "instances.csv"
    output_summary = OUTPUTS_DIR / "05_final_summary.json"

    write_jsonl(output_jsonl, final_rows)
    write_csv(
        output_csv,
        csv_rows,
        fieldnames=[
            "instance_id",
            "repo",
            "created_at",
            "file_path",
            "function_name",
            "body_line_count",
            "file_token_count",
            "external_reference_count",
            "verify_rank",
            "test_cmd",
        ],
    )
    write_json(
        output_summary,
        {
            "final_instance_count": len(final_rows),
            "outputs": {
                "instances_jsonl": str(output_jsonl),
                "instances_csv": str(output_csv),
                "artifacts_root": str(artifacts_root),
            },
        },
    )

    print(f"Final Stage 0 instances: {len(final_rows)}")


if __name__ == "__main__":
    main()
