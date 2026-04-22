#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from common import read_jsonl, write_csv, write_json
from config import CUTOFF_DATE, OUTPUTS_DIR, ensure_stage0_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot contamination cutoff figure for Stage 0."
    )
    parser.add_argument(
        "--instances-jsonl", default=str(OUTPUTS_DIR / "instances.jsonl")
    )
    parser.add_argument("--cutoff-date", default=CUTOFF_DATE)
    return parser.parse_args()


def parse_created_at(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    rows = read_jsonl(Path(args.instances_jsonl))
    if not rows:
        output_summary = OUTPUTS_DIR / "06_contamination_summary.json"
        write_json(
            output_summary,
            {
                "instance_count": 0,
                "note": "No instances found; contamination figure was not generated.",
            },
        )
        print("No instances available for contamination plot.")
        return

    cutoff_dt = datetime.strptime(args.cutoff_date, "%Y-%m-%d")

    csv_rows = []
    y_dates = []
    x_idx = []
    for i, row in enumerate(rows, start=1):
        created_str = row.get("created_at", "")
        created_dt = parse_created_at(created_str)
        delta_days = (created_dt - cutoff_dt).days
        csv_rows.append(
            {
                "instance_id": row.get("instance_id", ""),
                "created_at": created_str,
                "days_after_cutoff": delta_days,
            }
        )
        x_idx.append(i)
        y_dates.append(created_dt)

    output_csv = OUTPUTS_DIR / "contamination.csv"
    output_png = OUTPUTS_DIR / "contamination.png"
    output_summary = OUTPUTS_DIR / "06_contamination_summary.json"

    write_csv(
        output_csv,
        csv_rows,
        fieldnames=["instance_id", "created_at", "days_after_cutoff"],
    )

    plt.figure(figsize=(10, 4.5))
    plt.scatter(x_idx, y_dates, s=16, alpha=0.85, label="Retained instances")
    plt.axhline(
        cutoff_dt,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Qwen cutoff (2024-10)",
    )
    plt.gca().yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xlabel("Retained instance index")
    plt.ylabel("Instance created_at")
    plt.title("Contamination safety: retained instances vs model cutoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()

    min_days = min(row["days_after_cutoff"] for row in csv_rows)
    write_json(
        output_summary,
        {
            "instance_count": len(rows),
            "cutoff_date": args.cutoff_date,
            "minimum_days_after_cutoff": min_days,
            "all_instances_after_cutoff": min_days >= 0,
            "outputs": {
                "contamination_csv": str(output_csv),
                "contamination_plot": str(output_png),
            },
        },
    )

    print(f"Contamination figure written: {output_png}")


if __name__ == "__main__":
    main()
