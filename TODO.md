# TODO

## Initial `antirez/kilo` Test

Scope: one repository, one topic, one design document, one SHINE LoRA, and all
QA pairs that can be generated from that single document.

Source repository:

```text
repo_id: antirez__kilo
path: target_repos/antirez__kilo
commit: 323d93b29bd89a2cb446de90c4ed4fea1764176e
```

Initial topic:

```text
Overview / purpose
```

## Setup

- [x] Add `antirez/kilo` as a target repository submodule.
- [x] Document the `target_repos/{owner}__{repo}` convention.
- [x] Create a `prompts/` directory for standardized generator prompts.
- [x] Create a reusable config directory with a starter SHINE eval config.
- [x] Create the initial artifact directory layout:

```text
artifacts/
  antirez__kilo/
    easy/
      docs/
      qas/
      loras/
      ledger.md
      lora_store.json
      eval_results.jsonl
```

- [x] Record the source repo identity in the generated artifacts.

## Schemas

- [x] Define the design document schema.
- [x] Define the QA pair schema.
- [x] Define the ledger entry schema.
- [x] Define the LoRA store entry schema.
- [x] Define the evaluation result log schema.
- [x] Decide whether schemas live as JSON Schema files, documented examples, or both.

## Prompts

- [x] Add a topic discovery prompt template.
- [x] Add a design document generation prompt template.
- [x] Add a doc-derived QA generation prompt template.
- [x] Add a ledger example question generation prompt template.
- [x] Add prompt version identifiers to all generated artifacts.
- [x] Run one prompt pass manually on `antirez__kilo` and inspect quality before automation.

## Document And QA Generation

- [x] Generate the first easy design document for `Overview / purpose`.
- [x] Keep the design document under the SHINE context limit.
- [ ] Generate doc-derived QA pairs from only that design document.
- [ ] Generate ledger example questions from the same document, disjoint from eval QAs.
- [ ] Write `ledger.md` for the single LoRA.
- [ ] Write `lora_store.json` for the single LoRA.

## SHINE And Model Integration

- [x] Add an initial script adapted from `vendor/SHINE/inference.ipynb` for
  running naive, in-context, and SHINE evaluation from artifact paths.
- [ ] Confirm SHINE can generate a LoRA from the design document.
- [ ] Store the LoRA under `artifacts/antirez__kilo/easy/loras/`.
- [ ] Confirm Qwen3-8B runs without LoRA.
- [ ] Confirm Qwen3-8B runs with the generated LoRA loaded.
- [ ] Confirm Qwen3-8B runs with the design document in context.

## Initial Evaluation

- [x] Implement or script the three evaluation conditions:
  - Naive Qwen3-8B with no context and no LoRA.
  - In-context Qwen3-8B with the design document in the prompt.
  - SHINE Qwen3-8B with the generated LoRA and no document in the prompt.
- [ ] Run all generated QA pairs against all three conditions.
- [ ] Log each result to `eval_results.jsonl`.
- [ ] Manually grade each answer as correct, partial, or incorrect.
- [ ] Compare SHINE against naive and in-context.
- [ ] Decide whether to continue to multi-topic routing and blind QAs.

## Later Work

- [ ] Add additional `kilo` topics after the single-doc test passes.
- [ ] Add blind QA generation.
- [ ] Add embedding-based retrieval.
- [ ] Add Qwen-as-router.
- [ ] Add RAG-over-design-docs baseline.
- [ ] Add more target repositories.
