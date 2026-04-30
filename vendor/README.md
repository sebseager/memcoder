# Vendor

This directory contains external research systems vendored as git submodules.

- `SHINE`: upstream SHINE hypernetwork implementation used by the current
  MemCoder path to generate LoRA adapters from design documents.
- `doc-to-lora`: SakanaAI Doc-to-LoRA/D2L implementation explored in earlier
  experiments and kept for reference.
- `DyPRAG`: retrieval/parametric-memory baseline explored before the current
  SHINE-centered design.

Initialize these dependencies with:

```bash
git submodule update --init --recursive
```

Keep third-party target repositories under `target_repos/`, not here.
