"""
Exp 1 — Oracle Ceiling: Run SWE-bench evaluation on generated patches.

Uses the swebench harness to run test suites in Docker containers.

Usage:
    python evaluate.py --condition B
    python evaluate.py --condition B --instance-ids-file pilot_ids.txt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import EVAL_DIR, PATCHES_DIR, SWEBENCH_DATASET, SWEBENCH_SPLIT


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Parallel Docker containers"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="Per-instance timeout in seconds"
    )
    args = parser.parse_args()

    predictions_path = PATCHES_DIR / f"condition_{args.condition}.eval.jsonl"
    if not predictions_path.exists():
        print(f"ERROR: {predictions_path} not found. Run generate_patches.py first.")
        sys.exit(1)

    # Read instance IDs from predictions
    instance_ids = []
    with open(predictions_path) as f:
        for line in f:
            pred = json.loads(line)
            instance_ids.append(pred["instance_id"])

    print(f"Evaluating condition {args.condition}: {len(instance_ids)} instances")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report_dir = EVAL_DIR / f"condition_{args.condition}"
    report_dir.mkdir(parents=True, exist_ok=True)

    from swebench.harness.run_evaluation import main as run_eval

    run_eval(
        dataset_name=SWEBENCH_DATASET,
        split=SWEBENCH_SPLIT,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=args.max_workers,
        force_rebuild=False,
        cache_level="env",
        clean=False,
        open_file_limit=4096,
        run_id=f"exp1_condition_{args.condition}",
        timeout=args.timeout,
        namespace=None,
        rewrite_reports=False,
        modal=False,
        report_dir=str(report_dir),
    )

    print(f"\nEvaluation reports saved to {report_dir}")

    # Parse and summarize results
    summarize_eval(report_dir, args.condition, instance_ids)


def summarize_eval(report_dir: Path, condition: str, instance_ids: list[str]):
    """Parse swebench report JSON and print a summary."""
    # swebench writes reports as JSON files
    report_files = list(report_dir.rglob("*.json"))
    if not report_files:
        print("  No report files found yet.")
        return

    resolved = []
    failed = []
    errored = []

    for rf in report_files:
        with open(rf) as f:
            report = json.load(f)

        # swebench report format varies; handle both old and new
        if isinstance(report, dict):
            for iid, result in report.items():
                if isinstance(result, dict):
                    if result.get("resolved", False):
                        resolved.append(iid)
                    else:
                        failed.append(iid)
                elif isinstance(result, str):
                    if result == "RESOLVED":
                        resolved.append(iid)
                    else:
                        failed.append(iid)

    total = len(instance_ids)
    n_resolved = len(set(resolved))
    print(f"\n=== Condition {condition} Results ===")
    print(f"  Total instances: {total}")
    print(f"  Resolved: {n_resolved} ({n_resolved / total * 100:.1f}%)")
    print(f"  Failed: {len(set(failed))}")

    # Save summary
    summary = {
        "condition": condition,
        "total": total,
        "resolved": sorted(set(resolved)),
        "resolved_count": n_resolved,
        "resolve_rate": n_resolved / total if total > 0 else 0,
    }
    summary_path = report_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
