# Artifacts

This directory stores generated MemCoder artifacts.

The initial pilot layout is:

```text
artifacts/
  antirez__kilo/
    repo.json
    topics.json
    ledger.json
    easy/
      docs/
      qas/
      qa_examples/
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

Within each repo directory, `topics.json` records the discovered topic set before
documents are written. `ledger.json` is the canonical mapping from `document_id`
to the generated files. The `docs/`, `loras/`, `qas/`, and `qa_examples/`
filenames should use the same document ID stem, for example
`easy/docs/overview_purpose_1.json`, `easy/loras/overview_purpose_1.pt`,
`easy/qas/overview_purpose_1.json`, and
`easy/qa_examples/overview_purpose_1.json`. `doc_embedding` remains `null` until
that artifact is generated. `qa_examples` is `null` until routing examples are
generated, then points to the separate example-question artifact.
