# Eval

This package implements the MemCoder evaluation harness used by
`scripts/run_eval.py`.

- `config.py`: loads and snapshots YAML run configs.
- `runner.py`: runs prediction conditions and writes `predictions.jsonl`.
- `model.py`: loads Qwen/SHINE runtime pieces and applies LoRA dictionaries.
- `routing.py`: oracle and embedding-based LoRA selection.
- `composition.py`: rank-concatenation LoRA composition for top-k routing.
- `judge.py`: calls the configured LLM judge and writes `judgments.jsonl`.
- `report.py` and `plots.py`: aggregate judged outputs into reports and plots.

The package is importable directly from a checkout; the CLI prepends the repo
root to `sys.path`, so installation as a package is not required for local runs.
