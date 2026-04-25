#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from common import read_jsonl, write_json, write_jsonl
from config import OUTPUTS_DIR, REPOS_CACHE_DIR, ensure_stage0_dirs
from function_masking import mask_function_by_position
from patch_utils import build_unified_patch
from repo_utils import ensure_repo_checkout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare masked-function wipe attempts for Stage 0."
    )
    parser.add_argument(
        "--ranked-instances-jsonl",
        default=str(OUTPUTS_DIR / "02_ranked_instance_candidates.jsonl"),
    )
    parser.add_argument("--top-k-per-instance", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    ranked_rows = read_jsonl(Path(args.ranked_instances_jsonl))
    attempts: list[dict] = []
    failed_instances = 0

    for row in ranked_rows:
        top_candidates = row.get("top_candidates") or []
        if not top_candidates:
            failed_instances += 1
            continue

        instance_id = row["instance_id"]
        repo = row["repo"]
        base_commit = row["base_commit"]

        try:
            repo_dir = ensure_repo_checkout(repo, base_commit, REPOS_CACHE_DIR)
        except Exception:  # noqa: BLE001
            failed_instances += 1
            continue

        for rank, candidate in enumerate(
            top_candidates[: args.top_k_per_instance], start=1
        ):
            file_path = candidate["file_path"]
            abs_path = repo_dir / file_path
            if not abs_path.exists():
                continue

            try:
                source_text = abs_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            try:
                masked = mask_function_by_position(
                    source_text,
                    lineno=int(candidate["start_line"]),
                    col_offset=int(candidate["col_offset"]),
                )
            except Exception:
                continue

            mask_patch = build_unified_patch(
                file_path, source_text, masked.masked_file_text
            )
            if not mask_patch.strip():
                continue

            attempts.append(
                {
                    "instance_id": instance_id,
                    "repo": repo,
                    "created_at": row.get("created_at"),
                    "base_commit": base_commit,
                    "environment_setup_commit": row.get("environment_setup_commit"),
                    "docker_image": row.get("docker_image"),
                    "image_name": row.get("image_name"),
                    "test_cmd": row.get("test_cmd", ""),
                    "install_config": row.get("install_config") or {},
                    "FAIL_TO_PASS": row.get("FAIL_TO_PASS") or [],
                    "PASS_TO_PASS": row.get("PASS_TO_PASS") or [],
                    "gold_patch": row.get("gold_patch") or "",
                    "rank": rank,
                    "file_path": file_path,
                    "function_name": candidate["function_name"],
                    "start_line": int(candidate["start_line"]),
                    "end_line": int(candidate["end_line"]),
                    "col_offset": int(candidate["col_offset"]),
                    "body_start_line": int(candidate["body_start_line"]),
                    "body_end_line": int(candidate["body_end_line"]),
                    "body_line_count": int(candidate["body_line_count"]),
                    "file_token_count": int(candidate["file_token_count"]),
                    "external_reference_count": int(
                        candidate["external_reference_count"]
                    ),
                    "external_references": candidate.get("external_references") or [],
                    "mask_patch": mask_patch,
                }
            )

    output_attempts = OUTPUTS_DIR / "03_candidate_attempts.jsonl"
    output_summary = OUTPUTS_DIR / "03_attempts_summary.json"

    attempts.sort(key=lambda row: (row["instance_id"], row["rank"]))
    write_jsonl(output_attempts, attempts)
    write_json(
        output_summary,
        {
            "instances_input": len(ranked_rows),
            "instances_without_attempts": failed_instances,
            "attempt_rows": len(attempts),
            "outputs": {
                "attempts_jsonl": str(output_attempts),
            },
        },
    )

    print(f"Attempt rows: {len(attempts)}")


if __name__ == "__main__":
    main()
