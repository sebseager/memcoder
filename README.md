# MemCoder

MemCoder is a local code-QA research prototype. It turns short, natural-language
design documents about a repository into SHINE-generated LoRA adapters, retrieves
the relevant adapter at question time, and asks a local Qwen3-8B model to answer
without sending repository contents to a hosted inference API.

The current project claim is narrower than general code understanding: the pilot
evidence shows SHINE is most useful for identifier-grounded recall from concise
design docs. The detailed pilot findings live in
[`docs/observations.md`](docs/observations.md), and the original project plan is
preserved in [`docs/initial_plan.md`](docs/initial_plan.md).

## Repository Layout

- [`artifacts/`](artifacts/README.md): generated design docs, QA sets, ledgers,
  LoRA files, routing outputs, and per-repo artifact metadata.
- [`config/`](config/README.md): reusable run configs, especially
  `config/eval/*.yaml` for the evaluation harness.
- [`dashboard/`](dashboard/README.md): Streamlit UI for browsing artifacts,
  routing, LoRA answers, and logged results.
- [`docs/`](docs/README.md): project writeups, evaluation replication notes,
  moved planning docs, AI-use disclosure, and pilot observations.
- [`eval/`](eval/README.md): Python package behind prediction, judging, routing,
  LoRA composition, reporting, and plots.
- [`old/`](old/README.md): archived research paths explored before the current
  MemCoder design, including Doc-to-LoRA, DyPRAG, SHINE function-completion, and
  serving experiments.
- [`prompts/`](prompts/README.md): versioned prompts for topic discovery, design
  doc generation, QA generation, routing examples, and LLM-as-judge grading.
- [`schemas/`](schemas/README.md): JSON Schemas for generated artifacts and judge
  outputs.
- [`scripts/`](scripts/README.md): command-line entry points for LoRA baking,
  evaluation, routing, scoring, plotting, and analysis.
- [`target_repos/`](target_repos/README.md): third-party repositories evaluated
  by MemCoder, checked out as git submodules.
- [`tests/`](tests/README.md): lightweight unit tests for shared evaluation code.
- [`vendor/`](vendor/README.md): external research systems vendored as
  submodules and used for comparison or implementation.
- [`wheels/`](wheels/README.md): optional local wheel cache for platform-specific
  Python packages.

## Build

MemCoder is built as a Python 3.12 project managed by `uv`.

```bash
git submodule update --init --recursive
uv sync
```

The full inference path also needs local model/checkpoint assets:

- Qwen3-8B base weights, referenced by `model.qwen_base` in
  `config/eval/*.yaml`.
- A SHINE checkout at `vendor/SHINE` and compatible SHINE hypernetwork
  checkpoint paths.
- CUDA-capable PyTorch for practical SHINE/Qwen inference. The lockfile points
  `torch` at the CUDA 12.4 PyTorch index.
- `OPENAI_API_KEY` in the environment or `.env` when running the LLM judge.

Run the lightweight test suite after installation:

```bash
uv run python -m tests.test_composition
```

## Run And Deploy

This project is deployed locally rather than as a hosted service. A normal
deployment is a Linux workstation or cluster node with the model weights,
SHINE checkpoints, target repo submodules, and generated artifact directories
available on disk.

Bake SHINE LoRA files for design docs when needed:

```bash
uv run python scripts/generate_shine_lora.py \
  --config config/eval/kilo_easy_v0.yaml \
  --design-doc artifacts/antirez__kilo/easy/docs/overview_purpose_1.json
```

Run an end-to-end evaluation config:

```bash
uv run python scripts/run_eval.py all --config config/eval/kilo_easy_v0.yaml
```

The command writes a fresh `results/<run_name>_<timestamp>/` directory with
`predictions.jsonl`, `judgments.jsonl`, `report.md`, plots, a run config
snapshot, and a manifest. To inspect artifacts and results interactively:

```bash
uv run streamlit run dashboard/app.py
```

## Evaluation Replication

Detailed replication instructions are in
[`docs/evaluation.md`](docs/evaluation.md). In short, the included evaluation
uses the generated docs, QA pairs, ledgers, and LoRAs under `artifacts/` for
`antirez__kilo` and `marimo-team__marimo`, with run configs in `config/eval/`.

The primary compared systems are:

- `naive`: Qwen3-8B with no document context and no LoRA.
- `in_context`: Qwen3-8B with the ground-truth design document in context.
- `shine`: Qwen3-8B with a SHINE-generated LoRA and no source document in the
  prompt.
- `shine` with embedding routing for configs such as
  `config/eval/kilo_easy_v0_embedding.yaml`.

The target datasets and benchmarks needed for replication are included or linked
in the repo: target repositories are submodules under `target_repos/`; generated
design docs, QA test cases, and ledgers are under `artifacts/`; judge rubrics are
under `prompts/`; artifact schemas are under `schemas/`.

## External Artifacts

MemCoder extends or builds on these non-standard external artifacts:

- [`vendor/SHINE`](vendor/SHINE/README.md): upstream SHINE hypernetwork code for
  generating LoRA adapters from context.
- [`vendor/doc-to-lora`](vendor/doc-to-lora/README.md): SakanaAI Doc-to-LoRA /
  D2L code used in earlier experiments and diagnostics.
- [`vendor/DyPRAG`](vendor/DyPRAG/README.md): DyPRAG code used in an archived
  baseline exploration.
- Qwen3-8B and `Qwen/Qwen3-Embedding-0.6B` model families, configured through
  `config/eval/*.yaml` and `scripts/embedding_router.py`.
- OpenAI `gpt-5.1` as the default LLM-as-judge model in the current eval configs.
- Third-party target repositories listed in `.gitmodules`.

## AI Tool Use

AI assistance was used during project development for planning, code edits,
debugging, prompt drafting, and documentation. The disclosure and configuration
notes are in [`docs/ai_usage.md`](docs/ai_usage.md).
