# Exp 1 — Oracle Ceiling: Notes

**Date:** 2026-04-20
**Status:** In progress (pipeline audited and hardened; pilot runs executed)

## What Was Audited and Fixed

All scripts in `exp_1/` were audited against the Exp 1 protocol in `experiments/dyprag/README.md`.

### 1) Generation and reproducibility fixes

- Added explicit per-attempt decoding controls in `generate_patches.py`.
- Kept `MAX_NEW_TOKENS=8192` as the upper cap but added adaptive schedules:
  - `GENERATION_TOKEN_SCHEDULE_BD = [1024, 2048, 4096]`
  - `GENERATION_TOKEN_SCHEDULE_C = [1024, 2048, 8192]`
- Fixed sampling behavior so attempts do not silently fall back to greedy:
  - temperature floor per attempt is `0.05`, so all attempts run with `do_sample=True`.
- Added per-attempt diagnostics in prediction records:
  - `attempt_trace[]` with `raw_output_len`, `n_blocks`, `exact_search_hits`, `accepted`, `reason`.

### 2) SEARCH/REPLACE parser hardening

- Found and fixed a parser flaw where malformed blocks (missing closing `>>>> REPLACE`) could leak delimiters into replacement text and produce invalid diffs.
- Parser now:
  - requires valid block closure for canonical format,
  - rejects malformed blocks,
  - filters blocks containing delimiter markers inside SEARCH/REPLACE text,
  - rejects no-op blocks (`SEARCH == REPLACE`).

### 3) Block grounding and quality gates

- Added exact-hit checks before converting blocks to diffs.
- Added degenerate-output detection (runaway repetitive outputs, too many blocks, repeated noop patterns).
- Added strict file-path acceptance checks for generated diffs.

### 4) Evaluation/reporting hygiene

- `evaluate.py` now stores run-id-specific artifacts:
  - `results/eval_reports/condition_<C>.<run_id>.report.json`
  - `results/eval_reports/condition_<C>.<run_id>.summary.json`
- Still writes condition-level latest summary/report for convenience.

### 5) Added capability interference script

- Added `capability_interference.py` implementing the Exp 1 README interference check.
- Runs 20 fixed instruction-following probes before/after adapter load and reports percent drop.

## Output Cleanup Performed

To avoid confusion from iterative runs, volatile outputs were repeatedly cleaned and rerun.

**Kept (generic/reusable):**
- `results/oracle_loras/`
- `results/loss_curves/`
- `results/capability_checks/`

**Regenerated for current run state:**
- `results/patches/condition_{B,C,D}.jsonl`
- `results/eval_reports/condition_{B,C,D}.*`
- `results/analysis/exp1_analysis.json`

Removed duplicate root report files (`dyprag_exp1_condition_*.json`) after copying into `results/eval_reports/`.

## Direct Check of D 0-char Behavior

Inspected `sympy__sympy-23191` under Condition D after fixes.

From `attempt_trace`:
- Attempt 1: `raw_output_len=3773`, `n_blocks=0`, reason `no_patch_extracted`
- Attempt 2: `raw_output_len=4210`, `n_blocks=0`, reason `no_patch_extracted`
- Attempt 3: `raw_output_len=4210`, `n_blocks=0`, reason `no_patch_extracted`

Interpretation: in this case the model is not producing valid parseable blocks, not merely near-miss verbatim spans.

## Pilot Run Commands (Current Clean Iteration)

```bash
cd experiments/dyprag/exp_1
head -n 4 pilot_ids.txt > results/pilot4_ids.txt
for c in B C D; do uv run generate_patches.py --condition "$c" --ids-file results/pilot4_ids.txt; done
uv run evaluate.py --condition B --run-id exp1_condition_B_iter4 --max-workers 1 --timeout 1200
uv run evaluate.py --condition C --run-id exp1_condition_C_iter4 --max-workers 1 --timeout 1200
uv run evaluate.py --condition D --run-id exp1_condition_D_iter4 --max-workers 1 --timeout 1200
uv run analyze.py
```

## Current Pilot Results (iter4, 4 sympy instances)

From run summaries:

- Condition B: completed 1/4, resolved 0/4, empty patches 3/4
- Condition C: completed 2/4, resolved 0/4, empty patches 2/4
- Condition D: completed 1/4, resolved 0/4, empty patches 3/4

`analyze.py` therefore reports no B/C gap on this specific 4-instance all-sympy slice.

## Django Sanity Check (C-only)

Additional quick sanity run:
- IDs: `django__django-13265`, `django__django-13028`
- Run ID: `exp1_condition_C_django2`

Outcome:
- completed 1/2, resolved 0/2, empty patch 1/2

This indicates the issue is not purely sympy-specific in this small sample.

## Practical Interpretation

- The pipeline is now more reproducible and less brittle than earlier iterations.
- The dominant blocker in this pilot is generation validity/grounding (many empty or rejected patches), not harness execution.
- Oracle LoRA (D) still does not improve over B on this pilot subset.

## Next Recommended Experiment Step

Before scaling to the full constrained set, run a 12-instance mixed-repo pilot (existing `pilot_ids.txt`) with current scripts and inspect:
- patch non-empty rate by condition,
- completed-instance rate by condition,
- whether C materially exceeds B on a broader slice.

If C still does not exceed B, revisit prompt format (possibly condition-specific direct unified-diff generation for C only) and/or increase context reserve tuning for long files.
