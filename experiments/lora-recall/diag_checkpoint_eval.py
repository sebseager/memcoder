"""Diagnostic 1: Validate the checkpoint using doc-to-lora's own eval.

Runs the vendor's run_eval.py against the same checkpoint we use in our
experiment, on the SQuAD validation split.  If F1 is poor here, the
checkpoint itself is the problem.

Usage:
  cd experiments/lora-recall
  uv run python diag_checkpoint_eval.py
"""

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_D2L_ROOT = PROJECT_ROOT / "vendor" / "doc-to-lora"
CHECKPOINT_PATH = (
    Path(__file__).parents[1]
    / "doc-to-lora"
    / "trained_d2l"
    / "gemma_demo"
    / "checkpoint-80000"
    / "pytorch_model.bin"
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print("DIAGNOSTIC 1: Checkpoint validation via vendor eval")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    # Run vendor eval on SQuAD validation split
    cmd = [
        sys.executable,
        str(VENDOR_D2L_ROOT / "run_eval.py"),
        "--checkpoint_path",
        str(CHECKPOINT_PATH),
        "--datasets",
        "squad",
        "--split",
        "validation",
        "--max_val_samples_per_ds",
        "200",
        "--max_new_tokens",
        "256",
        "--eval_batch_size",
        "8",
    ]

    import os

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    # Ensure the vendor src is importable by the subprocess
    vendor_src = str(VENDOR_D2L_ROOT / "src")
    env["PYTHONPATH"] = vendor_src + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\nRunning: {' '.join(cmd)}")
    print(f"CWD: {VENDOR_D2L_ROOT}\n")

    result = subprocess.run(
        cmd,
        cwd=str(VENDOR_D2L_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    print("--- STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- STDERR (last 3000 chars) ---")
        print(result.stderr[-3000:])

    # Save output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"diag1_checkpoint_eval_{ts}.txt"
    with open(out_path, "w") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Return code: {result.returncode}\n\n")
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    print(f"\nOutput saved to {out_path}")

    if result.returncode != 0:
        print(f"\nWARNING: Eval exited with code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
