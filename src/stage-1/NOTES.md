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
