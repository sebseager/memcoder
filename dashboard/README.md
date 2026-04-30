# Dashboard

This directory contains the Streamlit dashboard for inspecting MemCoder
artifacts and results.

Entrypoint:

```bash
uv run streamlit run dashboard/app.py
```

Pages under `dashboard/pages/` cover artifact browsing, LoRA answer inspection,
routing exploration, and result analysis. Shared loading/runtime helpers live in
`dashboard/lib/`.

Interactive model calls require the same local model paths and LoRA artifacts
used by `config/eval/*.yaml`. Pages that only read JSON, JSONL, and markdown
artifacts can run without GPU access.
