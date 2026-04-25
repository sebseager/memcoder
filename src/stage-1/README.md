# Stage 1 - Oracle Ceiling

This stage evaluates whether oracle LoRA adapters recover completion quality lost by truncating context.

It was mostly a failed experiment, but might be good contextualization for the paper. Superseded by `stage-1x`.

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
- scripts/evaluate_completions.py: computes exact_match, execution-based pass@1, BLEU, and syntax validity.
- scripts/analyze_stage1.py: recovery-ratio analysis + bootstrap CI + gap-stratified summaries + plots.
- scripts/capability_interference.py: probe-based adapter interference check.
- scripts/run_stage1.sh: tiny/small/full orchestration.

## Evaluation Metrics

- exact_match: strict normalized string identity between prediction and ground-truth body.
- pass_at_1: targeted pytest execution after patching the predicted body into the original repo file.
  - Test selection heuristic uses likely-relevant test files (module import hits and function-name mentions).
  - If execution is infeasible (for example no relevant tests discovered or missing repo clone), evaluator falls back to exact_match and labels the fallback in per-instance columns (`pass_at_1_source`, `pass_at_1_status`).
- bleu4: token n-gram overlap.
- syntax_valid: AST parse check of reconstructed function.

By default, `evaluate_completions.py` runs execution-based pass@1 (`--pass-at-1-mode pytest_heuristic`). Use `--pass-at-1-mode exact_only` to disable test execution.

## Tiny Run

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode tiny
```

## Small Run

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-1
bash scripts/run_stage1.sh --mode small
```

## Outputs

Outputs are now mode + model scoped to avoid cross-run contamination:

- outputs/<mode>.<model-slug>/run_config.json
- outputs/<mode>.<model-slug>/oracle_loras/
- outputs/<mode>.<model-slug>/completions/
- outputs/<mode>.<model-slug>/evaluation/
- outputs/<mode>.<model-slug>/analysis/
- outputs/<mode>.<model-slug>/capability/
- outputs/<mode>.<model-slug>/plots/
- outputs/<mode>.<model-slug>/logs/
