"""
Exp 1 — Oracle Ceiling: Select and run a pilot subset.

Selects ~12 instances balanced across the main repos in the constrained subset,
then orchestrates: oracle LoRA training → patch generation (all conditions) →
SWE-bench evaluation → analysis.

Usage:
    python run_pilot.py                  # Select pilot + run everything
    python run_pilot.py --select-only    # Just select and print pilot IDs
    python run_pilot.py --skip-train     # Skip LoRA training (already done)
    python run_pilot.py --skip-eval      # Skip SWE-bench eval (just analyze)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    CONDITIONS,
    RESULTS_DIR,
)
from helpers import load_subsets, load_token_counts

PILOT_IDS_FILE = Path(__file__).resolve().parent / "pilot_ids.txt"


def select_pilot(n_target: int = 12, seed: int = 42) -> list[str]:
    """Select a pilot subset balanced across repos.

    Allocates slots proportional to each repo's share of the constrained set,
    with at least 1 instance per repo (for repos with constrained instances).
    """
    subsets = load_subsets()
    token_counts = load_token_counts()

    constrained_ids = set(subsets["constrained_instance_ids"])
    tc_map = {r["instance_id"]: r for r in token_counts}

    # Group by repo
    repo_instances = {}
    for iid in constrained_ids:
        rec = tc_map[iid]
        repo = rec["repo"]
        repo_instances.setdefault(repo, []).append(iid)

    # Sort repos by size (descending)
    repos = sorted(repo_instances, key=lambda r: len(repo_instances[r]), reverse=True)

    # Proportional allocation
    total = len(constrained_ids)
    allocation = {}
    remaining = n_target
    for repo in repos:
        share = len(repo_instances[repo]) / total
        n = max(1, round(share * n_target))
        allocation[repo] = min(n, len(repo_instances[repo]), remaining)
        remaining -= allocation[repo]
        if remaining <= 0:
            break

    # Distribute any leftover to largest repos
    for repo in repos:
        if remaining <= 0:
            break
        extra = min(remaining, len(repo_instances[repo]) - allocation.get(repo, 0))
        if extra > 0:
            allocation[repo] = allocation.get(repo, 0) + extra
            remaining -= extra

    # Sample from each repo
    rng = np.random.RandomState(seed)
    pilot_ids = []
    for repo in repos:
        n = allocation.get(repo, 0)
        if n == 0:
            continue
        candidates = sorted(repo_instances[repo])
        chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        pilot_ids.extend(chosen.tolist())

    print(f"Pilot selection ({len(pilot_ids)} instances):")
    for repo in repos:
        n = allocation.get(repo, 0)
        if n > 0:
            ids = [iid for iid in pilot_ids if tc_map[iid]["repo"] == repo]
            print(f"  {repo}: {len(ids)} instances")
            for iid in ids:
                print(f"    - {iid} ({tc_map[iid]['total_tokens']} tokens)")

    return pilot_ids


def run_step(cmd: list[str], description: str):
    """Run a subprocess and check for errors."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"{'=' * 60}")
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent))
    if result.returncode != 0:
        print(f"  ERROR: step failed with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run Exp 1 pilot")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--n-pilot", type=int, default=12)
    args = parser.parse_args()

    # Step 0: Select pilot instances
    if PILOT_IDS_FILE.exists():
        pilot_ids = PILOT_IDS_FILE.read_text().strip().split("\n")
        print(
            f"Using existing pilot IDs from {PILOT_IDS_FILE} ({len(pilot_ids)} instances)"
        )
    else:
        pilot_ids = select_pilot(n_target=args.n_pilot)
        PILOT_IDS_FILE.write_text("\n".join(pilot_ids))
        print(f"Saved pilot IDs to {PILOT_IDS_FILE}")

    if args.select_only:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    # Step 1: Train oracle LoRAs (only for condition D)
    if not args.skip_train:
        run_step(
            [python, "train_oracle.py", "--ids-file", str(PILOT_IDS_FILE)],
            "Train oracle LoRAs for pilot instances",
        )

    # Step 2: Generate patches for all conditions
    for cond in CONDITIONS:
        run_step(
            [
                python,
                "generate_patches.py",
                "--condition",
                cond,
                "--ids-file",
                str(PILOT_IDS_FILE),
            ],
            f"Generate patches for condition {cond}",
        )

    # Step 3: Run SWE-bench evaluation
    if not args.skip_eval:
        for cond in CONDITIONS:
            run_step(
                [python, "evaluate.py", "--condition", cond],
                f"Evaluate condition {cond}",
            )

    # Step 4: Analyze results
    run_step(
        [python, "analyze.py"],
        "Analyze results across all conditions",
    )


if __name__ == "__main__":
    main()
