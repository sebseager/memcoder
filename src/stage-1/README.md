# Stage 1 - Oracle Ceiling

This stage evaluates whether oracle LoRA adapters recover completion quality lost by truncating context.

## Goals

- Train one oracle LoRA per unique source file from Stage 0.
- Evaluate completion quality under conditions B/C/D:
  - B: Truncated context, no LoRA
  - C: Full context, no LoRA
  - D: Truncated context + oracle LoRA
- Compute recovery ratio: (D - B) / (C - B)
- Run capability interference probes before/after LoRA injection.

## Environment

Use the shared environment in src/.venv.

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
```

## Scripts

- scripts/config.py: stage constants and paths.
- scripts/helpers.py: shared IO/model/prompt utilities.
- scripts/init_run_config.py: writes model-scoped run_config.json at run start.
- scripts/train_oracle.py: trains file-level oracle adapters.
- scripts/generate_completions.py: runs B/C/D generation.
- scripts/identifier_overlap.py: computes novel identifier overlap diagnostics for condition D.
- scripts/evaluate_completions.py: computes pass@1 proxy and BLEU.
- scripts/analyze_stage1.py: recovery-ratio analysis + bootstrap CI + gap-stratified summaries + plots.
- scripts/capability_interference.py: probe-based adapter interference check.
- scripts/run_stage1.sh: pilot/full orchestration.

## Typical Tiny Pilot Run

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode pilot
```

`pilot` is an alias for `tiny`.

## Small Pilot Run

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode small
```

## Outputs

Outputs are now model-scoped to avoid cross-model contamination:

- outputs/<model-slug>/run_config.json
- outputs/<model-slug>/oracle_loras/
- outputs/<model-slug>/completions/
- outputs/<model-slug>/evaluation/
- outputs/<model-slug>/analysis/
- outputs/<model-slug>/capability/
- outputs/<model-slug>/plots/
- outputs/<model-slug>/logs/
