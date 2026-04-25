"""
Exp 1 — Oracle Ceiling: Analyze results across all conditions.

Computes:
  1. Resolve rate per condition
  2. Patch edit distance (secondary metric)
  3. Gate criterion: (D - B) / (C - B) > 0.5
  4. Bootstrap 95% CIs on gate ratio
  5. Per-repo breakdown

Usage:
    python analyze.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ANALYSIS_DIR, CONDITION_LABELS, CONDITIONS, EVAL_DIR, PATCHES_DIR
from helpers import load_subsets, load_swebench_dataset, load_token_counts


def load_eval_results(condition: str) -> dict | None:
    """Load normalized evaluation summary for one condition."""
    summary_path = EVAL_DIR / f"condition_{condition}.summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        raw = json.load(f)

    resolved = raw.get("resolved", [])
    total = raw.get("total_target_instances", raw.get("total", 0))
    resolved_count = raw.get("resolved_count", len(resolved))

    return {
        "total": total,
        "resolved_count": resolved_count,
        "resolve_rate": resolved_count / total if total else 0.0,
        "resolved": resolved,
        "target_ids": raw.get("target_instance_ids", []),
        "completed_ids": raw.get("completed_ids", []),
        "run_id": raw.get("run_id"),
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


def get_gate_instance_universe(results: dict) -> list[str]:
    """Aligned per-instance universe for B/C/D gate analysis."""
    needed = ["B", "C", "D"]
    if not all(results.get(c) for c in needed):
        return []

    id_sets = []
    for cond in needed:
        ids = results[cond].get("target_ids", [])
        if not ids:
            return []
        id_sets.append(set(ids))

    return sorted(set.intersection(*id_sets))


def bootstrap_gate_ci(
    b_arr: np.ndarray,
    c_arr: np.ndarray,
    d_arr: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for gate ratio."""

    def gate_stat(indices):
        b = b_arr[indices].mean()
        c = c_arr[indices].mean()
        d = d_arr[indices].mean()
        if c - b <= 0:
            return 0.0
        return (d - b) / (c - b)

    rng = np.random.RandomState(seed)
    n = len(b_arr)
    stats = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        stats.append(gate_stat(idx))

    stats = np.array(stats)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for cond in CONDITIONS:
        r = load_eval_results(cond)
        if r is None:
            print(f"WARNING: No eval results for condition {cond}")
        results[cond] = r

    print("\n" + "=" * 60)
    print("RESOLVE RATES")
    print("=" * 60)
    for cond in CONDITIONS:
        r = results.get(cond)
        if r:
            print(
                f"  {cond} ({CONDITION_LABELS[cond]}): "
                f"{r['resolved_count']}/{r['total']} = {r['resolve_rate']:.1%}"
            )
        else:
            print(f"  {cond}: NO DATA")

    gate_analysis = {}
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

            gate_ids = get_gate_instance_universe(results)
            if gate_ids:
                resolved_B = set(results["B"]["resolved"])
                resolved_C = set(results["C"]["resolved"])
                resolved_D = set(results["D"]["resolved"])

                b_arr = np.array([1 if iid in resolved_B else 0 for iid in gate_ids])
                c_arr = np.array([1 if iid in resolved_C else 0 for iid in gate_ids])
                d_arr = np.array([1 if iid in resolved_D else 0 for iid in gate_ids])
                ci_low, ci_high = bootstrap_gate_ci(b_arr, c_arr, d_arr)
                print(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
            else:
                ci_low, ci_high = float("nan"), float("nan")
                print("  WARNING: no aligned instance universe for gate CI")

            gate_analysis = {
                "ratio": gate_ratio,
                "passes": gate_ratio > 0.5,
                "bc_gap": rC - rB,
                "bd_gap": rD - rB,
                "bootstrap_ci_95": [ci_low, ci_high],
                "n_instances": len(gate_ids),
            }
        else:
            print(f"  WARNING: C does not exceed B (gap = {rC - rB:.1%})")
            print(
                "  The budget cutoff may be too generous. See README logical gap check."
            )

    if results.get("B") and results.get("C"):
        gap = results["C"]["resolve_rate"] - results["B"]["resolve_rate"]
        print(f"\n{'=' * 60}")
        print("LOGICAL GAP CHECK: C - B ≥ 5pp")
        print(f"{'=' * 60}")
        print(f"  C - B = {gap:.1%}")
        print(
            "  PASS"
            if gap >= 0.05
            else "  FAIL: Gap is < 5pp. Budget cutoff may be too high."
        )

    print(f"\n{'=' * 60}")
    print("EDIT DISTANCE TO GROUND TRUTH")
    print(f"{'=' * 60}")

    swebench_data = load_swebench_dataset()
    gate_ids = get_gate_instance_universe(results)
    if not gate_ids:
        subsets = load_subsets()
        gate_ids = subsets["constrained_instance_ids"]

    edit_dist_results = {}
    for cond in CONDITIONS:
        patches = load_patches(cond)
        if not patches:
            continue
        r = results.get(cond)
        target_ids = set(r.get("target_ids", [])) if r else set()

        distances = []
        for iid in gate_ids:
            if target_ids and iid not in target_ids:
                continue
            if iid not in patches or iid not in swebench_data:
                continue
            gold_patch = swebench_data[iid]["patch"]
            pred_patch = patches[iid]
            distances.append(edit_distance(pred_patch, gold_patch))

        if distances:
            mean_d = float(np.mean(distances))
            median_d = float(np.median(distances))
            print(
                f"  {cond}: mean={mean_d:.1f}, median={median_d:.1f} (n={len(distances)})"
            )
            edit_dist_results[cond] = {
                "mean": mean_d,
                "median": median_d,
                "n": len(distances),
            }

    if any(results.get(c) for c in CONDITIONS):
        print(f"\n{'=' * 60}")
        print("PER-REPO BREAKDOWN (resolve rate)")
        print(f"{'=' * 60}")

        token_counts = load_token_counts()
        repo_map = {}
        for rec in token_counts:
            if rec["instance_id"] in gate_ids:
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

    analysis = {
        "resolve_rates": {
            c: results[c]["resolve_rate"] for c in CONDITIONS if results.get(c)
        },
        "edit_distances": edit_dist_results,
        "gate_criterion": gate_analysis,
    }

    analysis_path = ANALYSIS_DIR / "exp1_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
