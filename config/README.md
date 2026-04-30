# Config

This directory stores reusable MemCoder run configs.

- `eval/kilo_easy_v0.yaml`: oracle-routed easy evaluation for
  `artifacts/antirez__kilo`.
- `eval/kilo_easy_v0_embedding.yaml`: embedding-routed variant for kilo.
- `eval/marimo_easy_v0.yaml`: oracle-routed easy evaluation for
  `artifacts/marimo-team__marimo`.
- `eval/marimo_smoke_v2.yaml`: smaller smoke config for marimo/judge plumbing.
- `eval/two_checkpoint_kilo_v0.yaml`: comparison config for multiple SHINE
  checkpoints.

The YAML files are self-contained run definitions for `scripts/run_eval.py`.
Machine-specific paths, especially `model.qwen_base`, should be edited locally
or overridden before running on a new machine.
