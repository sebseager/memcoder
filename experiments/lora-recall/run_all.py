#!/usr/bin/env python3
"""run_all.py — Run all LoRA recall diagnostics and the main recall suite.

Diagnostics (run in order of likelihood):
  1. Checkpoint validation: run doc-to-lora's own eval on the checkpoint
  2. Context window check:  token-length analysis + tiny-doc probe
  3. Synthetic canary test: fictitious doc with zero pretraining signal
  4. Reset verification:    confirm model.reset() fully clears LoRA
  5. Recall suite:          full probe suite (diag5_recall_suite.py)

Usage:
  cd experiments/lora-recall
  uv run python run_all.py       # run everything
  uv run python run_all.py 2 3   # run only diagnostics 2 and 3
  uv run python run_all.py 5     # run only the recall suite
  uv run python run_all.py 6 7   # run compositional + routing experiments
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
NC = "\033[0m"


@dataclass(frozen=True)
class Step:
    number: int
    name: str
    script: str


ALL_STEPS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
STEP_DEFS: dict[int, Step] = {
    1: Step(1, "Checkpoint validation (vendor eval)", "diag_checkpoint_eval.py"),
    2: Step(2, "Context window & token length check", "diag_context_window.py"),
    3: Step(3, "Synthetic canary test", "diag_canary.py"),
    4: Step(4, "Reset verification", "diag_reset_verify.py"),
    5: Step(5, "Recall suite", "diag5_recall_suite.py"),
    6: Step(6, "Compositional stability (multi-LoRA merge)", "diag6_compositional.py"),
    7: Step(7, "Routing signal quality", "diag7_routing.py"),
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_hms() -> str:
    return utc_now().strftime("%H:%M:%S")


def make_logger(log_file: Path):
    def log(message: str = "") -> None:
        line = f"[{utc_hms()}] {message}"
        print(line)
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    return log


def run_step(step: Step, script_dir: Path, log_file: Path, log) -> int:
    log("")
    log(f"{CYAN}{'=' * 66}{NC}")
    log(f"{CYAN}  Step {step.number}: {step.name}{NC}")
    log(f"{CYAN}{'=' * 66}{NC}")

    script_path = script_dir / step.script
    if not script_path.exists():
        log(f"{RED}  ERROR: Script not found: {step.script}{NC}")
        return 1

    cmd = ["uv", "run", "python", step.script]
    step_start = time.time()

    process = subprocess.Popen(
        cmd,
        cwd=str(script_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    with log_file.open("a", encoding="utf-8") as fh:
        for line in process.stdout:
            print(line, end="")
            fh.write(line)

    rc = process.wait()
    elapsed = int(time.time() - step_start)

    if rc == 0:
        log(f"{GREEN}  [OK] Step {step.number} completed ({elapsed}s){NC}")
    else:
        log(
            f"{RED}  [FAIL] Step {step.number} FAILED with exit code {rc} ({elapsed}s){NC}"
        )
    return rc


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "steps",
        nargs="*",
        help="Optional step numbers to run (1-7). Default: run all.",
    )
    return parser.parse_args(argv)


def list_result_files(results_dir: Path) -> list[Path]:
    patterns = ("*.json", "*.csv", "*.txt")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(results_dir.glob(pattern))
    return sorted(set(files))


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    log_file = results_dir / f"run_all_{timestamp}.log"
    log = make_logger(log_file)

    steps_to_run = args.steps if args.steps else [str(s) for s in ALL_STEPS]
    valid_range = f"1-{max(ALL_STEPS)}"

    log(f"Starting run_all.py at {utc_now().strftime('%a %b %d %H:%M:%S UTC %Y')}")
    log(f"Steps to run: {' '.join(steps_to_run)}")
    log(f"Results directory: {results_dir}")
    log(f"Log file: {log_file}")

    overall_start = time.time()
    passed = 0
    failed = 0
    skipped = 0

    for raw_step in steps_to_run:
        try:
            step_num = int(raw_step)
        except ValueError:
            log(f"{YELLOW}  Unknown step: {raw_step} (valid: {valid_range}){NC}")
            skipped += 1
            continue

        step = STEP_DEFS.get(step_num)
        if step is None:
            log(f"{YELLOW}  Unknown step: {step_num} (valid: {valid_range}){NC}")
            skipped += 1
            continue

        rc = run_step(step, script_dir, log_file, log)
        if rc == 0:
            passed += 1
        else:
            failed += 1
            if step_num == 1:
                log(
                    f"{YELLOW}  [WARN] Checkpoint eval failed; remaining diagnostics may still be useful.{NC}"
                )

    overall_elapsed = int(time.time() - overall_start)

    log("")
    log(f"{CYAN}{'=' * 66}{NC}")
    log(f"{CYAN}  SUMMARY{NC}")
    log(f"{CYAN}{'=' * 66}{NC}")
    log(f"  Passed:  {passed}")
    log(f"  Failed:  {failed}")
    log(f"  Skipped: {skipped}")
    log(f"  Total time: {overall_elapsed}s")
    log("")
    log(f"  Results in: {results_dir}/")
    log(f"  Log file:   {log_file}")
    log("")

    log("  Result files:")
    for f in list_result_files(results_dir):
        log(f"    {f.name}")

    if failed > 0:
        log(f"{RED}  Some steps failed — review the log above.{NC}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
