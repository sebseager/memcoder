# Stage 1 Lab Notebook

## Entry 2026-04-21 1

- Objective: begin Stage 1 (oracle ceiling) pipeline under src/stage-1.
- Confirmed shared environment exists at src/.venv.
- Stage 1 workspace scaffold created:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
mkdir -p stage-1/scripts stage-1/outputs/{oracle_loras,completions,evaluation,analysis,capability,plots,logs}
```

- Small-scale-first policy locked in before scaling:
  - build resumable scripts,
  - run a pilot subset first,
  - validate assumptions and output contracts,
  - then scale to full Stage 1.

## Entry 2026-04-21 2

- Installed Stage 1 dependencies into src/.venv:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
uv pip install 'torch>=2.7.0' transformers peft datasets accelerate bitsandbytes sacrebleu matplotlib scipy
```

- Observations:
  - ML stack is now available for LoRA training and generation.
  - The workload estimated from Stage 0 outputs is significant:
    - instances: 120
    - unique files (oracle adapters needed): 79
  - This validates the need for resumable scripts and pilot-first execution.

## Entry 2026-04-21 3

- Implemented modular Stage 1 scripts under `stage-1/scripts`:
  - `config.py`
  - `helpers.py`
  - `train_oracle.py`
  - `generate_completions.py`
  - `evaluate_completions.py`
  - `analyze_stage1.py`
  - `capability_interference.py`
  - `run_stage1.sh`

- Added resumable behavior:
  - oracle training skips adapters with existing `training_meta.json` unless `--force`.
  - completion generation supports `--force` overwrite and deterministic subset control.

## Entry 2026-04-21 4

- Ran initial pilot orchestration:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode pilot
```

- Pilot findings:
  - Training and B/C generation completed.
  - Condition D generated no usable rows because trained adapters did not overlap pilot instance files.
  - Analysis crashed on empty Condition D CSV (`pandas.errors.EmptyDataError`).

- Fixes applied after pilot validation:
  - Pilot alignment:
    - `run_stage1.sh` now derives pilot instance IDs and pilot file keys from the same first-N Stage 0 rows.
    - training uses `--file-keys-file` for matching adapters.
    - generation uses `--instance-ids-file` for matching evaluation instances.
  - Empty-data robustness:
    - `evaluate_completions.py` now always writes CSV headers, even when a condition has zero valid rows.
    - `analyze_stage1.py` now handles missing/empty condition CSVs and zero-length bootstrap inputs.
  - Warning cleanup:
    - replaced deprecated `torch_dtype` argument with `dtype` in model loading.
    - enabled non-reentrant gradient checkpointing (`use_reentrant=False`) to remove checkpoint warnings we control.

- Additional note:
  - A bitsandbytes CUDA backend FutureWarning remains from upstream package internals; this is not a project-code warning path.

## Entry 2026-04-21 5

- Re-ran pilot after fixes:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode pilot
```

- Pilot execution status:
  - oracle training: completed for 4/4 pilot file adapters (resumed from checkpoints on rerun).
  - generation: completed for conditions B/C/D on 4/4 pilot instances.
  - evaluation + analysis: completed and plots generated.
  - capability interference: completed for pilot adapters.

- Pilot metrics (proxy):
  - B: pass@1 proxy = 0.000, BLEU-4 = 0.008
  - C: pass@1 proxy = 0.000, BLEU-4 = 0.093
  - D: pass@1 proxy = 0.000, BLEU-4 = 0.008

- Additional bug fix after rerun:
  - condition D generation succeeded after sanitizing dynamic adapter names.
  - capability script needed the same adapter-name sanitization and an indentation fix.

- Warning cleanup status:
  - resolved deprecation warning from `torch_dtype` by switching to `dtype`.
  - resolved gradient-checkpointing `use_reentrant` warning with explicit non-reentrant settings.
  - resolved generation-config argument warning by normalizing model generation defaults in `helpers.py`.
  - remaining warning: bitsandbytes backend FutureWarning from third-party package internals.

## Entry 2026-04-21 6

- Re-ran capability interference after final warning cleanup to restore pilot-consistent `latest` outputs:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
python scripts/capability_interference.py --model-id Qwen/Qwen2.5-Coder-1.5B-Instruct --max-adapters 4
```

- Result:
  - baseline probe score: 90%
  - adapter scores: all 90% (0% drop each)
  - flagged adapters (>15% drop): 0

## Entry 2026-04-21 7

- Objective: audit Stage 1 readiness before scaling from tiny to small pilot.
- Review actions:
  - Inspected Stage 1 scripts and outputs for robustness, reproducibility, and metric validity.
  - Compared B/C/D completion artifacts directly to verify whether condition D behavior was genuinely different from B.
- Key findings from audit:
  - `MAX_NEW_TOKENS=256` was too low for this dataset. Ground-truth body lengths (Qwen2.5 tokenizer, n=120):
    - p50=482, p75=618, p90=729, p95=789, p99=942, max=1738
    - 100/120 instances exceed 256 tokens
  - Truncated context budget was slightly violated due a post-truncation marker append (observed ~2052 tokens under a 2048 budget).
  - Reuse of existing adapters relied only on `training_meta.json` presence and did not validate provenance (model/hparams), risking stale adapter reuse.
  - Capability check could include stale adapters from unrelated prior runs because it scanned adapter directories globally.

## Entry 2026-04-21 8

- Applied hardening patches before small pilot:
  - `scripts/config.py`
    - generation defaults updated for deterministic and less truncation-prone evaluation:
      - `MAX_NEW_TOKENS: 256 -> 1024`
      - `TEMPERATURE: 0.2 -> 0.0`
      - `TOP_P: 0.95 -> 1.0`
  - `scripts/helpers.py`
    - fixed `truncate_to_budget` so returned context (including marker) stays within token budget.
  - `scripts/train_oracle.py`
    - added adapter compatibility checks against current run config (`model_id`, chunk size, batch size, grad accum, lr, min/max epochs).
    - incompatible existing adapters are now retrained (directory reset) instead of silently reused.
    - training metadata now records provenance fields.
  - `scripts/generate_completions.py`
    - validates adapter `base_model_name_or_path` against requested model id before loading.
    - marks incompatible adapters as skipped with explicit reason.
    - added generation diagnostics per row:
      - `generated_token_count`
      - `hit_max_new_tokens`
  - `scripts/evaluate_completions.py`
    - added summary fields:
      - `mean_generated_token_count`
      - `max_new_token_hit_rate`
  - `scripts/capability_interference.py`
    - added `--file-keys-file` to restrict evaluation to current subset.
    - skips adapters whose base model mismatches requested model id.
  - `scripts/run_stage1.sh`
    - added uv bootstrap if `src/.venv` missing.
    - mode support now includes `tiny`, `small`, `full` (`pilot` kept as alias for `tiny`).
    - subset alignment generalized beyond pilot (shared subset files for training/generation/capability).
    - capability check now receives subset file-key filter.

## Entry 2026-04-21 9

- Re-ran tiny pilot with hardened scripts:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode tiny
```

- Tiny results (n=4):
  - B: pass@1 proxy = 0.000, BLEU-4 = 0.036
  - C: pass@1 proxy = 0.000, BLEU-4 = 0.235
  - D: pass@1 proxy = 0.000, BLEU-4 = 0.036
- Interpretation:
  - C >> B signal exists (full context helps), but D still does not improve over B in tiny.

## Entry 2026-04-21 10

- Executed small pilot:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode small
```

- Small execution status:
  - trained adapters: 12/12
  - generation: B/C/D completed on 12/12 instances
  - capability check: completed on 12/12 subset adapters

- Small results (n=12):
  - B: pass@1 proxy = 0.000, BLEU-4 = 0.074, syntax_valid_rate = 0.167
  - C: pass@1 proxy = 0.083, BLEU-4 = 0.308, syntax_valid_rate = 0.250
  - D: pass@1 proxy = 0.000, BLEU-4 = 0.013, syntax_valid_rate = 0.167
  - Recovery summary:
    - pass recovery: 0.000 (CI [0.000, 0.000])
    - BLEU recovery: -0.482 (CI [-5.714, 2.361])
  - Length diagnostics:
    - `max_new_token_hit_rate` = 0.0 for all conditions (no active truncation at 1024 tokens)

- Additional diagnostics:
  - Condition D adapter statuses: 1 `loaded_initial`, 11 `swapped`, 0 skipped.
  - B vs D output comparison (small): 8/12 identical, 4/12 changed.
  - The changed outputs were mostly harmful (shorter or off-target continuations), matching the D BLEU drop.

- Conclusion (Stage 1 gate):
  - Not promising yet for progression to Stage 2.
  - We satisfy the precondition that truncated context hurts (C > B), but oracle adapters fail to recover and currently degrade performance (D <= B).
  - Following the research plan's stop condition, we should pause and diagnose Stage 1 before advancing.
