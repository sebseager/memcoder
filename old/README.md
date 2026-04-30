# Old Experiments

This directory archives research paths explored before the current MemCoder
implementation. These files are not part of the supported build/run path, but
they document decisions that shaped the current design.

## Paths Explored

- `experiments/doc-to-lora/`: early SakanaAI Doc-to-LoRA/D2L demos. This path
  proved useful for understanding document-to-adapter mechanics and serving, but
  was set aside because the released hypernetwork checkpoints targeted smaller
  models than the Qwen3-8B backbone used for the main project.
- `experiments/dyprag/`: DyPRAG-style oracle/retrieval experiments over
  SWE-Bench Lite. The work characterized context-constrained instances and
  tried oracle LoRAs, but patch-generation validity and weak B/C/D separation
  made it a poor fit for the project timeline.
- `experiments/serving/`: hot-swappable adapter serving experiments using
  vLLM/FastAPI and Doc-to-LoRA. This informed local deployment ideas but is not
  the current MemCoder runtime.
- `experiments/shine/`: the direct predecessor of current MemCoder. It initially
  tried using SHINE for function-body completion with SWE-rebench/SWE-bench
  Docker pass@1 evaluation. That benchmark was too strict for the available
  zero-shot and small IFT runs, but the SHINE integration, code-slice handling,
  and evaluation lessons evolved into the current design-doc QA formulation.
- `experiments/shine/stage-0/`: contamination-safe SWE-rebench dataset
  construction, function masking, and Docker verification.
- `experiments/shine/stage-1/`: oracle LoRA ceiling experiments for masked
  function completion.
- `experiments/shine/stage-1a/`: zero-shot SHINE inference on code slices with
  SWE-rebench scoring. It produced useful infrastructure but poor completion
  quality.
- `experiments/shine/stage-1b/`: IFT SHINE-on-code workflow for
  `(slice, signature+docstring, body)` triples, including lightweight and Docker
  evaluation. It remained focused on function completion rather than the current
  natural-language QA task.
- `experiments/lora-recall/`: focused recall diagnostics for generated LoRAs,
  including canary docs, reset verification, composition, and routing signal
  checks. These diagnostics influenced the current identifier-recall framing.

For the supported current system, start at the root `README.md`, `docs/`, and
`scripts/run_eval.py`.
