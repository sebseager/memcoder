# Tests

This directory contains lightweight tests for shared MemCoder code.

Current test:

```bash
uv run python -m tests.test_composition
```

`test_composition.py` checks the LoRA composition routines in
`eval/composition.py` without loading the full Qwen/SHINE runtime.
