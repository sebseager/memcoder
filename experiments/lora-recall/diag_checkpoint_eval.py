"""Diagnostic 1: Validate the checkpoint using doc-to-lora's own eval.

Runs the vendor's run_eval.py against the same checkpoint we use in our
experiment, on the SQuAD validation split.  If F1 is poor here, the
checkpoint itself is the problem.

Usage:
  cd experiments/lora-recall
  uv run python diag_checkpoint_eval.py
"""

import os
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

DEFAULT_BATCH_SIZES = (8, 4, 2, 1)
MAX_VAL_SAMPLES_PER_DS = 200
MAX_NEW_TOKENS = 256


def _is_cuda_oom(stderr: str) -> bool:
    text = stderr.lower()
    if "cuda" in text and "out of memory" in text:
        return True
    return (
        "cuda out of memory" in text
        or "torch.outofmemoryerror" in text
        or "cublas_status_alloc_failed" in text
    )


def _parse_batch_sizes() -> list[int]:
    raw = os.environ.get("D2L_DIAG1_BATCH_SIZES", "")
    if not raw:
        return list(DEFAULT_BATCH_SIZES)

    batch_sizes = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        try:
            parsed = int(value)
        except ValueError as exc:
            raise SystemExit(
                f"Invalid D2L_DIAG1_BATCH_SIZES entry: '{value}' (must be integers)"
            ) from exc
        if parsed <= 0:
            raise SystemExit(
                f"Invalid D2L_DIAG1_BATCH_SIZES entry: '{value}' (must be > 0)"
            )
        batch_sizes.append(parsed)

    if not batch_sizes:
        raise SystemExit(
            "D2L_DIAG1_BATCH_SIZES is set but no valid batch sizes were parsed"
        )
    return batch_sizes


def _build_eval_cmd(eval_batch_size: int) -> list[str]:
    return [
        sys.executable,
        str(VENDOR_D2L_ROOT / "run_eval.py"),
        "--checkpoint_path",
        str(CHECKPOINT_PATH),
        "--datasets",
        "squad",
        "--split",
        "validation",
        "--max_val_samples_per_ds",
        str(MAX_VAL_SAMPLES_PER_DS),
        "--max_new_tokens",
        str(MAX_NEW_TOKENS),
        "--eval_batch_size",
        str(eval_batch_size),
        "--eval_batch_size_gen",
        str(eval_batch_size),
    ]


def main():
    print("=" * 70)
    print("DIAGNOSTIC 1: Checkpoint validation via vendor eval")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Ensure the vendor src is importable by the subprocess
    vendor_src = str(VENDOR_D2L_ROOT / "src")
    env["PYTHONPATH"] = vendor_src + os.pathsep + env.get("PYTHONPATH", "")

    batch_sizes = _parse_batch_sizes()
    print(f"Batch-size retry plan: {batch_sizes}")

    attempts: list[tuple[list[str], subprocess.CompletedProcess[str]]] = []
    final_cmd: list[str] | None = None
    final_result: subprocess.CompletedProcess[str] | None = None

    for idx, batch_size in enumerate(batch_sizes, start=1):
        cmd = _build_eval_cmd(batch_size)
        print(
            f"\n--- Attempt {idx}/{len(batch_sizes)} (eval_batch_size={batch_size}) ---"
        )
        print(f"Running: {' '.join(cmd)}")
        print(f"CWD: {VENDOR_D2L_ROOT}\n")

        result = subprocess.run(
            cmd,
            cwd=str(VENDOR_D2L_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        attempts.append((cmd, result))

        print("--- STDOUT ---")
        print(result.stdout)
        if result.stderr:
            print("--- STDERR (last 3000 chars) ---")
            print(result.stderr[-3000:])

        if result.returncode == 0:
            final_cmd = cmd
            final_result = result
            if idx > 1:
                print(f"\nRecovered after reducing eval_batch_size to {batch_size}.")
            break

        if _is_cuda_oom(result.stderr) and idx < len(batch_sizes):
            print(
                f"\nCUDA OOM at eval_batch_size={batch_size}; "
                "retrying with a smaller batch size..."
            )
            continue

        final_cmd = cmd
        final_result = result
        break

    if final_cmd is None or final_result is None:
        print("ERROR: No eval attempts were executed")
        sys.exit(1)

    # Save output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"diag1_checkpoint_eval_{ts}.txt"
    with open(out_path, "w") as f:
        for idx, (attempt_cmd, attempt_result) in enumerate(attempts, start=1):
            f.write(f"=== Attempt {idx} ===\n")
            f.write(f"Command: {' '.join(attempt_cmd)}\n")
            f.write(f"Return code: {attempt_result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(attempt_result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(attempt_result.stderr)
            f.write("\n\n")

        f.write("=== Final result ===\n")
        f.write(f"Command: {' '.join(final_cmd)}\n")
        f.write(f"Return code: {final_result.returncode}\n")

    print(f"\nOutput saved to {out_path}")

    if final_result.returncode != 0:
        print(f"\nWARNING: Eval exited with code {final_result.returncode}")
        sys.exit(final_result.returncode)


if __name__ == "__main__":
    main()
