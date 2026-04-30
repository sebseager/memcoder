# Artifacts

This directory stores generated MemCoder data products: design documents, QA
pairs, LoRA dictionaries, ledgers, routing outputs, and small scored-result
fixtures.

Current repo-level artifact directories include:

- `antirez__kilo/`: easy-tier artifacts for the kilo editor pilot.
- `marimo-team__marimo/`: easy-tier artifacts for the marimo pilot.
- `fake_lora_composition/`: synthetic fixtures for LoRA-composition tests and
  demos.

Common layout:

```text
artifacts/<repo_id>/
  repo.json
  topics.json
  ledger.json
  easy/
    docs/
    qas/
    qas_v1/
    qa_examples/
    loras/
    ledger.md
    routing_results.*.jsonl
```

`repo.json` records target repository identity. `topics.json` records the audited
topic set before documents are written. `ledger.json` is the canonical mapping
from `document_id` to the generated files and metadata used by the eval harness.

Document, QA, LoRA, and routing files should share the same document ID stem
where applicable, for example:

```text
easy/docs/overview_purpose_1.json
easy/qas/overview_purpose_1.json
easy/loras/overview_purpose_1.pt
```

The harness treats missing LoRA paths as "skip the `shine` row for this document"
while still allowing `naive` and `in_context` rows to run.
