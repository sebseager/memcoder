#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import UTC, datetime

import matplotlib.pyplot as plt
from common import write_csv, write_json, write_jsonl
from config import (
    CUTOFF_DATE,
    DATASET_NAME,
    DEFAULT_MIN_TARGET_CANDIDATES,
    DEFAULT_TARGET_CANDIDATES,
    OUTPUTS_DIR,
    ensure_stage0_dirs,
)
from datasets import get_dataset_split_names, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter SWE-rebench leaderboard instances for Stage 0."
    )
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--min-created-at", default=CUTOFF_DATE)
    parser.add_argument(
        "--target-candidates", type=int, default=DEFAULT_TARGET_CANDIDATES
    )
    parser.add_argument(
        "--min-target-candidates", type=int, default=DEFAULT_MIN_TARGET_CANDIDATES
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional hard cap after filtering (0 means no cap).",
    )
    return parser.parse_args()


def is_month_split(split_name: str) -> bool:
    return bool(re.fullmatch(r"\d{4}_\d{2}", split_name))


def normalize_row(row: dict, source_split: str) -> dict:
    meta = row.get("meta") or {}
    install_config = row.get("install_config") or {}
    return {
        "repo": row.get("repo"),
        "instance_id": row.get("instance_id"),
        "base_commit": row.get("base_commit"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "created_at": row.get("created_at"),
        "patch": row.get("patch"),
        "test_patch": row.get("test_patch"),
        "meta": meta,
        "install_config": install_config,
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS"),
        "PASS_TO_PASS": row.get("PASS_TO_PASS"),
        "docker_image": row.get("docker_image"),
        "image_name": row.get("image_name"),
        "interface": row.get("interface"),
        "source_split": source_split,
        "test_cmd": install_config.get("test_cmd", ""),
    }


def passes_stage0_filters(row: dict, min_created_at: str) -> bool:
    created_at = row.get("created_at") or ""
    meta = row.get("meta") or {}
    num_modified_files = int(meta.get("num_modified_files") or -1)
    has_test_patch = bool(meta.get("has_test_patch"))
    return created_at >= min_created_at and num_modified_files == 1 and has_test_patch


def plot_month_counts(csv_rows: list[dict], output_png: str) -> None:
    months = [row["month"] for row in csv_rows]
    kept = [int(row["kept_instances"]) for row in csv_rows]
    total = [int(row["total_instances"]) for row in csv_rows]

    plt.figure(figsize=(11, 4))
    plt.plot(months, total, marker="o", label="Total in split")
    plt.plot(months, kept, marker="o", label="Kept after filters")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Instance count")
    plt.title("Stage 0 monthly filter counts")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_png, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    split_names = get_dataset_split_names(args.dataset_name)
    month_splits = sorted([s for s in split_names if is_month_split(s)])
    if not month_splits:
        raise RuntimeError("No YYYY_MM month splits found in dataset.")

    dedup: dict[str, dict] = {}
    monthly_rows: list[dict] = []
    prefilter_counter = Counter()
    postfilter_counter = Counter()

    for split in month_splits:
        ds = load_dataset(args.dataset_name, split=split)
        total_count = len(ds)
        kept_count = 0

        for raw in ds:
            normalized = normalize_row(raw, source_split=split)
            created_month = (normalized.get("created_at") or "")[:7]
            if created_month:
                prefilter_counter[created_month] += 1

            if not passes_stage0_filters(
                normalized, min_created_at=args.min_created_at
            ):
                continue

            if created_month:
                postfilter_counter[created_month] += 1

            iid = normalized["instance_id"]
            if iid not in dedup:
                dedup[iid] = normalized
                kept_count += 1

        monthly_rows.append(
            {
                "month": split.replace("_", "-"),
                "total_instances": total_count,
                "kept_instances": kept_count,
            }
        )

    filtered = sorted(dedup.values(), key=lambda row: row.get("created_at") or "")

    if args.max_candidates > 0 and len(filtered) > args.max_candidates:
        filtered = filtered[: args.max_candidates]

    target_status = (
        "met" if len(filtered) >= args.min_target_candidates else "below_minimum"
    )

    output_jsonl = OUTPUTS_DIR / "01_filtered_instances.jsonl"
    output_summary = OUTPUTS_DIR / "01_filter_summary.json"
    output_month_csv = OUTPUTS_DIR / "01_filter_counts_by_month.csv"
    output_month_png = OUTPUTS_DIR / "01_filter_counts_by_month.png"

    write_jsonl(output_jsonl, filtered)
    write_csv(
        output_month_csv,
        monthly_rows,
        fieldnames=["month", "total_instances", "kept_instances"],
    )
    plot_month_counts(monthly_rows, str(output_month_png))

    summary = {
        "dataset_name": args.dataset_name,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "month_splits": month_splits,
        "filters": {
            "created_at_gte": args.min_created_at,
            "meta.num_modified_files_eq": 1,
            "meta.has_test_patch_eq": True,
        },
        "target_candidates": args.target_candidates,
        "min_target_candidates": args.min_target_candidates,
        "target_status": target_status,
        "filtered_count": len(filtered),
        "prefilter_month_counts": dict(sorted(prefilter_counter.items())),
        "postfilter_month_counts": dict(sorted(postfilter_counter.items())),
        "outputs": {
            "filtered_instances_jsonl": str(output_jsonl),
            "month_counts_csv": str(output_month_csv),
            "month_counts_plot": str(output_month_png),
        },
    }
    write_json(output_summary, summary)

    print(f"Filtered instances: {len(filtered)}")
    print(f"Summary: {output_summary}")


if __name__ == "__main__":
    main()
