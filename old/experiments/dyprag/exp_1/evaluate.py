"""
Exp 1 — Oracle Ceiling: Run SWE-bench evaluation on generated patches.

Uses the swebench harness to run test suites in Docker containers.

Usage:
    python evaluate.py --condition B
    python evaluate.py --condition B --instance-ids-file pilot_ids.txt
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memcoder.old.experiments.dyprag.exp_1.config import EVAL_DIR, PATCHES_DIR, SWEBENCH_DATASET, SWEBENCH_SPLIT


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument(
        "--max-workers", type=int, default=1, help="Parallel Docker containers"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="Per-instance timeout in seconds"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional explicit run id (default: exp1_condition_<COND>)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force image rebuild in SWE-bench harness",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean harness artifacts aggressively after run",
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
    run_id = args.run_id or f"exp1_condition_{args.condition}"

    from swebench.harness.run_evaluation import main as run_eval

    report_path = run_eval(
        dataset_name=SWEBENCH_DATASET,
        split=SWEBENCH_SPLIT,
        instance_ids=instance_ids,
        predictions_path=str(predictions_path),
        max_workers=args.max_workers,
        force_rebuild=args.force_rebuild,
        cache_level="env",
        clean=args.clean,
        open_file_limit=4096,
        run_id=run_id,
        timeout=args.timeout,
        namespace=None,
        rewrite_reports=False,
        modal=False,
        report_dir=".",
    )

    if report_path is None:
        print("\nWARNING: SWE-bench returned no report path.")
        return

    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    if not report_path.exists():
        print(f"\nERROR: expected report file not found: {report_path}")
        sys.exit(1)

    report_copy_run = EVAL_DIR / f"condition_{args.condition}.{run_id}.report.json"
    shutil.copy2(report_path, report_copy_run)

    # Also update a stable "latest" location for this condition.
    report_copy_latest = EVAL_DIR / f"condition_{args.condition}.report.json"
    shutil.copy2(report_path, report_copy_latest)
    print(f"\nRun report saved to {report_copy_run}")

    summarize_eval(report_copy_run, args.condition, instance_ids, run_id)


def summarize_eval(
    report_path: Path,
    condition: str,
    instance_ids: list[str],
    run_id: str,
):
    """Normalize SWE-bench report and persist a stable summary."""
    with open(report_path) as f:
        report = json.load(f)

    total_target = len(instance_ids)
    resolved_ids = sorted(set(report.get("resolved_ids", [])))
    unresolved_ids = sorted(set(report.get("unresolved_ids", [])))
    error_ids = sorted(set(report.get("error_ids", [])))
    empty_patch_ids = sorted(set(report.get("empty_patch_ids", [])))
    completed_ids = sorted(set(report.get("completed_ids", [])))

    n_resolved = len(resolved_ids)
    resolved_rate = n_resolved / total_target if total_target else 0.0

    print(f"\n=== Condition {condition} Results ===")
    print(f"  Target instances: {total_target}")
    print(f"  Completed instances: {len(completed_ids)}")
    print(f"  Resolved: {n_resolved} ({resolved_rate * 100:.1f}%)")
    print(f"  Unresolved: {len(unresolved_ids)}")
    print(f"  Empty patch: {len(empty_patch_ids)}")
    print(f"  Errors: {len(error_ids)}")

    # Save summary
    summary = {
        "condition": condition,
        "run_id": run_id,
        "report_path": str(report_path),
        "total_target_instances": total_target,
        "target_instance_ids": instance_ids,
        "completed_instances": len(completed_ids),
        "completed_ids": completed_ids,
        "resolved": resolved_ids,
        "unresolved": unresolved_ids,
        "errors": error_ids,
        "empty_patch": empty_patch_ids,
        "resolved_count": n_resolved,
        "resolve_rate": resolved_rate,
    }
    summary_path_run = EVAL_DIR / f"condition_{condition}.{run_id}.summary.json"
    with open(summary_path_run, "w") as f:
        json.dump(summary, f, indent=2)

    summary_path_latest = EVAL_DIR / f"condition_{condition}.summary.json"
    with open(summary_path_latest, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path_run}")


if __name__ == "__main__":
    main()
