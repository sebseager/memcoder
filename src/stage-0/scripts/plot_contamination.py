from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from config import DEFAULT_CONFIG, OUTPUTS_DIR, ensure_stage_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot contamination safety figure")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=OUTPUTS_DIR / "repo_candidates.json",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=OUTPUTS_DIR / "contamination_first_commit.png",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=OUTPUTS_DIR / "contamination_first_commit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_stage_dirs()

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    selected = payload.get("selected_repos", [])
    if not selected:
        raise RuntimeError("No selected repos found; run discover_repos.py first")

    rows = []
    for repo in selected:
        first_commit = repo.get("first_commit_date")
        if not first_commit:
            continue
        rows.append(
            {
                "repo": repo["full_name"],
                "stars": repo["stars"],
                "first_commit_date": pd.to_datetime(first_commit),
                "passes_cutoff": bool(repo.get("passes_first_commit_filter", False)),
            }
        )

    df = pd.DataFrame(rows).sort_values("first_commit_date").reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)

    cutoff = pd.to_datetime(DEFAULT_CONFIG.first_commit_cutoff.isoformat())

    plt.figure(figsize=(12, 6))
    colors = df["passes_cutoff"].map({True: "#1b9e77", False: "#d95f02"})
    plt.scatter(df["repo"], df["first_commit_date"], c=colors, s=45)
    plt.axhline(
        cutoff,
        color="#7570b3",
        linestyle="--",
        linewidth=2,
        label=f"Qwen3-8B estimated cutoff ({cutoff.date()})",
    )

    plt.title("Contamination Check: Repo First Commit Dates")
    plt.xlabel("Repository")
    plt.ylabel("First commit date")
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(args.output_figure, dpi=180)

    pass_count = int(df["passes_cutoff"].sum())
    print(f"Saved figure to {args.output_figure}")
    print(f"Saved table to {args.output_csv}")
    print(f"Repos passing cutoff: {pass_count}/{len(df)}")


if __name__ == "__main__":
    main()
