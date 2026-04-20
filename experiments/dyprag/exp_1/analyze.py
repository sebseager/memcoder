"""
Exp 1 — Oracle Ceiling: Analyze results across all conditions.

Computes:
  1. Resolve rate per condition
  2. Patch edit distance (secondary metric)
  3. Gate criterion: (D - B) / (C - B) > 0.5
  4. Bootstrap 95% CIs on the gate ratio
  5. Per-repo breakdown

Usage:
    python analyze.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ANALYSIS_DIR,
    CONDITION_LABELS,
    CONDITIONS,
    EVAL_DIR,
    EXP1_DIR,
    PATCHES_DIR,
)
from helpers import load_subsets, load_swebench_dataset, load_token_counts


def load_eval_results(condition: str) -> dict | None:
    """Load the evaluation summary for a condition."""
    # swebench writes report to CWD as dyprag_exp1_condition_X.exp1_condition_X.json
    summary_path = (
        EXP1_DIR / f"dyprag_exp1_condition_{condition}.exp1_condition_{condition}.json"
    )
    if not summary_path.exists():
        # Fallback to old location
        summary_path = EVAL_DIR / f"condition_{condition}" / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        raw = json.load(f)
    # Normalise keys
    total = raw.get("total", raw.get("total_instances", 0))
    resolved_count = raw.get("resolved_count", raw.get("resolved_instances", 0))
    resolved = raw.get("resolved", raw.get("resolved_ids", []))
    return {
        "total": total,
        "resolved_count": resolved_count,
        "resolve_rate": resolved_count / total if total else 0.0,
        "resolved": resolved,
    }


def load_patches(condition: str) -> dict:
    """Load generated patches as {instance_id: patch}."""
    path = PATCHES_DIR / f"condition_{condition}.jsonl"
    if not path.exists():
        return {}
    patches = {}
    with open(path) as f:
        for line in f:
            pred = json.loads(line)
            patches[pred["instance_id"]] = pred["model_patch"]
    return patches


def edit_distance(a: str, b: str) -> int:
    """Line-level edit distance between two patches."""
    a_lines = a.strip().split("\n") if a.strip() else []
    b_lines = b.strip().split("\n") if b.strip() else []
    m, n = len(a_lines), len(b_lines)

    # Use DP for line-level Levenshtein
    if m == 0:
        return n
    if n == 0:
        return m

    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a_lines[i - 1] == b_lines[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def bootstrap_ci(
    data: np.ndarray, stat_fn, n_boot: int = 10000, ci: float = 0.95, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (point_estimate, ci_low, ci_high)."""
    rng = np.random.RandomState(seed)
    point = stat_fn(data)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_fn(sample))
    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    return (
        float(point),
        float(np.percentile(boot_stats, alpha * 100)),
        float(np.percentile(boot_stats, (1 - alpha) * 100)),
    )


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval results for each condition
    results = {}
    for cond in CONDITIONS:
        r = load_eval_results(cond)
        if r is None:
            print(f"WARNING: No eval results for condition {cond}")
        results[cond] = r

    # --- Resolve rates ---
    print("\n" + "=" * 60)
    print("RESOLVE RATES")
    print("=" * 60)
    for cond in CONDITIONS:
        r = results.get(cond)
        if r:
            rate = r["resolve_rate"]
            n = r["resolved_count"]
            total = r["total"]
            print(f"  {cond} ({CONDITION_LABELS[cond]}): {n}/{total} = {rate:.1%}")
        else:
            print(f"  {cond}: NO DATA")

    # --- Gate criterion ---
    if results.get("B") and results.get("C") and results.get("D"):
        rB = results["B"]["resolve_rate"]
        rC = results["C"]["resolve_rate"]
        rD = results["D"]["resolve_rate"]

        print(f"\n{'=' * 60}")
        print("GATE CRITERION: (D - B) / (C - B) > 0.5")
        print(f"{'=' * 60}")

        if rC - rB > 0:
            gate_ratio = (rD - rB) / (rC - rB)
            print(f"  B→C gap: {rC - rB:.1%}")
            print(f"  B→D gap: {rD - rB:.1%}")
            print(f"  Gate ratio: {gate_ratio:.3f}")
            print(f"  Gate {'PASSES' if gate_ratio > 0.5 else 'FAILS'}")

            # Bootstrap CI on gate ratio
            # Need per-instance resolved indicators
            resolved_B = set(results["B"].get("resolved", []))
            resolved_C = set(results["C"].get("resolved", []))
            resolved_D = set(results["D"].get("resolved", []))

            subsets = load_subsets()
            instance_ids = subsets["constrained_instance_ids"]

            # Per-instance binary outcomes
            b_arr = np.array([1 if iid in resolved_B else 0 for iid in instance_ids])
            c_arr = np.array([1 if iid in resolved_C else 0 for iid in instance_ids])
            d_arr = np.array([1 if iid in resolved_D else 0 for iid in instance_ids])

            def gate_stat(indices):
                b = b_arr[indices].mean()
                c = c_arr[indices].mean()
                d = d_arr[indices].mean()
                if c - b <= 0:
                    return 0.0
                return (d - b) / (c - b)

            rng = np.random.RandomState(42)
            boot_ratios = []
            n = len(instance_ids)
            for _ in range(10000):
                idx = rng.choice(n, size=n, replace=True)
                boot_ratios.append(gate_stat(idx))
            boot_ratios = np.array(boot_ratios)

            ci_low = float(np.percentile(boot_ratios, 2.5))
            ci_high = float(np.percentile(boot_ratios, 97.5))
            print(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        else:
            print(f"  WARNING: C does not exceed B (gap = {rC - rB:.1%})")
            print(
                "  The budget cutoff may be too generous. See README logical gap check."
            )

    # --- Logical gap check: C must exceed B by ≥5pp ---
    if results.get("B") and results.get("C"):
        gap = results["C"]["resolve_rate"] - results["B"]["resolve_rate"]
        print(f"\n{'=' * 60}")
        print("LOGICAL GAP CHECK: C - B ≥ 5pp")
        print(f"{'=' * 60}")
        print(f"  C - B = {gap:.1%}")
        if gap < 0.05:
            print("  FAIL: Gap is < 5pp. Budget cutoff may be too high.")
        else:
            print("  PASS")

    # --- Edit distance analysis ---
    print(f"\n{'=' * 60}")
    print("EDIT DISTANCE TO GROUND TRUTH")
    print(f"{'=' * 60}")

    swebench_data = load_swebench_dataset()
    subsets = load_subsets()
    instance_ids = subsets["constrained_instance_ids"]

    edit_dist_results = {}
    for cond in CONDITIONS:
        patches = load_patches(cond)
        if not patches:
            continue
        distances = []
        for iid in instance_ids:
            if iid not in patches or iid not in swebench_data:
                continue
            gold_patch = swebench_data[iid]["patch"]
            pred_patch = patches[iid]
            dist = edit_distance(pred_patch, gold_patch)
            distances.append(dist)
        if distances:
            mean_d = np.mean(distances)
            median_d = np.median(distances)
            print(
                f"  {cond}: mean={mean_d:.1f}, median={median_d:.1f} (n={len(distances)})"
            )
            edit_dist_results[cond] = {
                "mean": float(mean_d),
                "median": float(median_d),
                "n": len(distances),
            }

    # --- Per-repo breakdown ---
    if any(results.get(c) for c in CONDITIONS):
        print(f"\n{'=' * 60}")
        print("PER-REPO BREAKDOWN (resolve rate)")
        print(f"{'=' * 60}")

        token_counts = load_token_counts()
        repo_map = {}
        for rec in token_counts:
            if rec["instance_id"] in instance_ids:
                repo = rec["repo"].split("/")[-1]
                repo_map.setdefault(repo, []).append(rec["instance_id"])

        for repo in sorted(repo_map):
            iids = repo_map[repo]
            print(f"\n  {repo} ({len(iids)} instances):")
            for cond in CONDITIONS:
                r = results.get(cond)
                if not r:
                    continue
                resolved = set(r.get("resolved", []))
                n_resolved = sum(1 for iid in iids if iid in resolved)
                rate = n_resolved / len(iids) if iids else 0
                print(f"    {cond}: {n_resolved}/{len(iids)} = {rate:.0%}")

    # --- Save full analysis ---
    analysis = {
        "resolve_rates": {
            c: results[c]["resolve_rate"] for c in CONDITIONS if results.get(c)
        },
        "edit_distances": edit_dist_results,
        "gate_criterion": {},
    }

    if results.get("B") and results.get("C") and results.get("D"):
        rB = results["B"]["resolve_rate"]
        rC = results["C"]["resolve_rate"]
        rD = results["D"]["resolve_rate"]
        if rC - rB > 0:
            analysis["gate_criterion"] = {
                "ratio": (rD - rB) / (rC - rB),
                "passes": (rD - rB) / (rC - rB) > 0.5,
                "bc_gap": rC - rB,
                "bd_gap": rD - rB,
            }

    analysis_path = ANALYSIS_DIR / "exp1_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
