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
bash scripts/run_stage1.sh --mode tiny
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
bash scripts/run_stage1.sh --mode tiny
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
    - mode support now includes `tiny`, `small`, `full`.
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

## Entry 2026-04-21 11

- Implemented FIX1 core repair in Stage 1 training/inference code:
  - `scripts/train_oracle.py` objective changed from raw file LM chunking to supervised prompt->body completion with prompt-token loss masking.
  - `scripts/helpers.py` extended with AST function extraction plus Stage 0 instance fallback builders for supervised records.
  - `scripts/generate_completions.py` now logs adapter activity per row (`adapter_status`, `active_adapter_before_generate`, first output tokens) for swap/load verification.
  - Added objective/truncation/eval compatibility checks in adapter metadata to prevent stale adapter reuse.

- Gate diagnostics executed:
  - Gate 1: strict baseline-vs-oracle next-token probe on evaluation prompt showed near-identical behavior pre-fix, confirming training distribution mismatch.
  - Gate 2: inspect-only sample verified prompt/target boundaries and loss masking on paper.
  - Gate 3: single-adapter retrain converged and produced B vs D divergence.

## Entry 2026-04-21 12

- During Gate 3 and early tiny reruns, discovered a second major issue:
  - `ORACLE_CHUNK_SIZE=1536` forced truncation of essentially all supervised records (prompt+target lengths were mostly ~2.1k-3.0k tokens).
  - This violated FIX1's train/eval distribution alignment intent.

- Fix applied:
  - `scripts/config.py`: `ORACLE_CHUNK_SIZE` raised to 3072.
  - Eval kept optional/off by default in training to avoid eval-time OOM.

- Gate 4 verification:
  - Checked `loaded_initial` and `swapped` paths on explicit two-instance runs.
  - Active adapter names matched expected file keys at generation time.

## Entry 2026-04-21 13

- Post-fix reruns (no extra intervention beyond supervised objective + 3072 sequence cap):

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode tiny
bash scripts/run_stage1.sh --mode small
```

- Tiny (n=4):
  - B BLEU-4 = 0.036
  - C BLEU-4 = 0.235
  - D BLEU-4 = 0.283
  - Gate 5 minimum (`D > B` aggregate) passed.

- Small (n=12):
  - B BLEU-4 = 0.074
  - C BLEU-4 = 0.308
  - D BLEU-4 = 0.163
  - BLEU recovery ratio mean = 0.675, but CI lower bound remained below 0.
  - Strict Gate 6 CI criterion did not clear.

## Entry 2026-04-21 14

- FIX1 Step 7 interventions run independently:

1) Intervention A (tighten truncation budget 2048 -> 1024)
  - Tiny (n=4): directional success (`D >> B` on BLEU).
  - Small (n=12): unstable/weak aggregate recovery; not selected.

2) Intervention B (behavioral second-pass regularization)
  - Added optional flags to `scripts/train_oracle.py`:
    - `--behavioral-probes`
    - `--behavioral-epochs`
    - `--behavioral-lr-mult`
  - Second pass trains on up to 5 eval-style probe prompts per file using the same masked completion objective.

- Intervention B tiny run (n=4):
  - B BLEU-4 = 0.036
  - C BLEU-4 = 0.235
  - D BLEU-4 = 0.347
  - Gate 7 condition satisfied (`D > B` on tiny).

- Intervention B small run (n=12):
  - B: pass@1 proxy = 0.000, BLEU-4 = 0.074
  - C: pass@1 proxy = 0.083, BLEU-4 = 0.308
  - D: pass@1 proxy = 0.083, BLEU-4 = 0.207
  - Recovery summary:
    - pass recovery mean = 0.824 (CI [0.000, 3.000])
    - BLEU recovery mean = 0.739 (CI [-1.940, 4.507])

- Current status:
  - Mean recovery is now directionally strong and D reaches C on pass@1 proxy at n=12.
  - Strict Gate 6 requirement (BLEU recovery CI lower bound > 0) is still not met with current sample size/noise.

## Entry 2026-04-21 15

- Objective: implement BRIDGE.md pre-flight requirements before Stage 2 transition.

- Code changes implemented:
  - `scripts/config.py`
    - switched LoRA target modules to FFN-only (`up_proj`, `down_proj`) with explicit decision/justification/reference fields.
    - added model-scoped output path helpers (`get_stage1_paths`) and run-config filename constants.
    - added gap-stratification thresholds (`LOW_GAP_BLEU_THRESHOLD=0.05`, `HIGH_GAP_BLEU_THRESHOLD=0.20`).
  - `scripts/helpers.py`
    - added deterministic JSON helpers (`load_json`, `write_json`, `json_sha256`).
    - added `build_run_config(...)` for run-level provenance.
    - made `load_model_and_tokenizer(...)` explicitly seed-aware.
    - extended `FunctionExample` with source line spans.
    - added deterministic behavioral probe manifest builder (`build_behavioral_probe_manifest`) keyed by AST order.
  - `scripts/train_oracle.py`
    - migrated outputs to model-scoped directories.
    - added `--seed` and propagated deterministic seeding through split/training args.
    - behavioral second pass now uses deterministic AST-order probes (`examples[:N]`) instead of instance-ID ordering.
    - writes `adapter_metadata.json` with saved probe manifest for each adapter.
    - enforces probe-manifest mismatch guard: refuses retraining when saved probes conflict with current generation logic.
    - adapter compatibility now checks `seed` and `behavioral_probe_manifest_sha256`.
  - `scripts/generate_completions.py`
    - migrated to model-scoped outputs and adapter directories.
    - added `--seed` for generation determinism.
  - `scripts/evaluate_completions.py`
    - migrated to model-scoped outputs.
    - added `--model-id`.
    - added pre-analysis B->C gap computation and persisted `bleu_gap_bc` + `gap_stratum` (`low|medium|high|unknown`) to per-instance CSVs.
  - `scripts/analyze_stage1.py`
    - migrated to model-scoped outputs.
    - added `--model-id` and seed-locked bootstrap.
    - now reads model-specific `run_config.json` and embeds run config in analysis JSON, per-instance CSV columns, and plot metadata sidecars.
    - added stratified recovery summaries by `gap_stratum` for pass and BLEU.
  - `scripts/capability_interference.py`
    - migrated to model-scoped outputs and adapter paths.
    - added `--seed` and safe handling when no adapter directory exists.
  - `scripts/init_run_config.py` (new)
    - writes model-scoped `run_config.json` at run start.
  - `scripts/identifier_overlap.py` (new)
    - computes `novel_identifier_count`, `novel_identifier_rate`, and novel identifier lists for generated completions.
    - writes JSONL/CSV/summary outputs under model-scoped analysis directory.
  - `scripts/run_stage1.sh`
    - now initializes run config at start.
    - propagates seed and generation/truncation/behavioral parameters to all scripts.
    - stores subset logs under model-scoped `outputs/<model-slug>/logs`.
    - runs `identifier_overlap.py` after completion generation.
    - final outputs now reported under model-scoped path.
  - `README.md`
    - updated script list and output layout to model-scoped format.

- Commands run for validation:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
python scripts/init_run_config.py --model-id Qwen/Qwen3-4B --mode small --seed 42 --behavioral-probes 5
python scripts/train_oracle.py --model-id Qwen/Qwen3-4B --max-files 1 --dry-run --seed 42
python -m py_compile scripts/*.py
```

- Validation observations:
  - model-scoped run config creation succeeded (`outputs/qwen-qwen3-4b/run_config.json`).
  - dry-run adapter indexing with new path plumbing succeeded.
  - Python scripts compile cleanly.
  - expected failure when trying to evaluate a model without model-scoped completion files yet (`Missing completions file`), which is correct given the new output contract.

- Conclusion:
  - BRIDGE Step 1 pre-flight implementation is now in place in Stage 1 code.
  - Next execution step is the planned `small` run on `Qwen/Qwen3-4B` via `run_stage1.sh` to populate model-scoped B/C/D outputs and run the qualitative/quantitative gates.

## Entry 2026-04-21 16

- Objective: change Stage 1 output slug format from `<model-slug>` to `<mode>.<model-slug>`.

- Changes made:
  - `scripts/config.py`
    - added `mode_to_slug(...)` and `build_output_slug(...)`.
    - updated `get_stage1_paths(...)` to resolve output root as `outputs/<mode>.<model-slug>` when mode is provided or exported via `STAGE1_MODE`.
  - `scripts/init_run_config.py`
    - now passes `--mode` into `get_stage1_paths(...)` so run config writes to the same mode-scoped root.
  - `scripts/run_stage1.sh`
    - exports `STAGE1_MODE="$MODE"` for downstream Python scripts.
    - updates shell slug construction to `mode.model_slug`.
  - `README.md`
    - output path docs updated to `<mode>.<model-slug>`.

- Validation command:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
python scripts/init_run_config.py --model-id Qwen/Qwen3-4B --mode small --seed 42
```

- Observation:
  - run config now writes to `outputs/small.qwen-qwen3-4b/run_config.json`, confirming slug format is `mode<dot><model_slug>`.

## Entry 2026-04-21 17

- Objective: remove `pilot` mode alias entirely and keep only `tiny`, `small`, `full` in Stage 1.

- Changes made:
  - `scripts/run_stage1.sh`
    - default mode changed from `pilot` to `tiny`.
    - removed alias branch mapping `pilot -> tiny`.
    - invalid-mode message now lists only `tiny|small|full`.
    - renamed subset helper vars from `PILOT_*` to `SUBSET_*`.
    - subset log prints now use neutral wording (`Subset ...`).
  - `README.md`
    - updated orchestration description to tiny/small/full.
    - updated tiny example command to `--mode tiny`.
    - removed alias statement.
  - `NOTES.md`
    - updated historical command snippets to use `--mode tiny`.
    - updated mode-support note to remove alias language.

- Validation run:

```bash
cd /home/seb/Developer/Classes/continual-learning/src/stage-1
bash -n scripts/run_stage1.sh
python -m py_compile scripts/*.py
```

- Result:
  - syntax checks passed.
  - grep check confirms mode handling now lists only `tiny|small|full`.
