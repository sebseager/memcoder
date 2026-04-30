# Scripts

This directory contains command-line entry points for building artifacts,
running evaluations, and analyzing results.

Primary commands:

- `run_eval.py`: main harness with `predict`, `judge`, `report`, and `all`
  subcommands.
- `generate_shine_lora.py`: bake a SHINE-generated LoRA dictionary from one
  design document and update the artifact ledger.
- `embedding_router.py`: produce embedding-based routing JSONL files.
- `validate_artifacts.py`: validate generated artifacts against schemas.
- `rejudge.py`, `score_results_jsonl.py`, `plot_eval.py`,
  `plot_naive_hard.py`, `prompt_ab_test.py`, and `rubric_impact.py`: scoring
  and analysis utilities for judged runs.
- `run_lora_composition_eval.py`, `run_fake_lora_composition_eval.py`,
  `run_two_checkpoint_eval.py`, and related display scripts: composition and
  checkpoint-comparison experiments.
- `run_shine_eval.py` and `run_routed_lora_eval.py`: older/specialized
  evaluation paths retained for reproducibility.

Run scripts with `uv run python scripts/<name>.py ...` from the repository root.
