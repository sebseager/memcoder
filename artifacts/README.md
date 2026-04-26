# Artifacts

This directory stores generated MemCoder artifacts.

The initial pilot layout is:

```text
artifacts/
  antirez__kilo/
    repo.json
    ledger.json
    easy/
      docs/
      qas/
      loras/
      ledger.md
      eval_results.jsonl
```

Generated artifacts should record the target repository identity in
`repo.json`:

```text
repo_id: antirez__kilo
commit: 323d93b29bd89a2cb446de90c4ed4fea1764176e
```

Within each repo directory, `ledger.json` is the canonical mapping from
`document_id` to the generated files. The `docs/`, `loras/`, and `qas/`
filenames should use the same document ID stem, for example
`easy/docs/overview_purpose_1.json`, `easy/loras/overview_purpose_1.pt`,
and `easy/qas/overview_purpose_1.json`. `doc_embedding` and `qa_examples`
are included in the file map and remain `null` until those artifacts are
generated.
