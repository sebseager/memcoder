# Artifact Schemas

Schemas live as JSON Schema files in this directory. The JSON fields are the
canonical contract; examples in prompts should match these schemas.

Important eval-script contracts:

- Design document JSON must contain `document`; the full JSON object is passed
  through as `doc_metadata` in the eval sidecar.
- QA JSON must contain top-level `qa_pairs`; each QA record must contain
  `question` and `answer`. The full QA record is passed through as
  `qa_metadata` in each eval result line.
- Repo-level `ledger.json` maps each `document_id` to canonical difficulty,
  topic metadata, and `docs/`, `loras/`, `qas/`, `doc_embedding`, and
  `qa_examples` paths. `doc_embedding` and `qa_examples` are `null` until
  those artifacts are generated.
- Repo-level `repo.json` contains only the artifact repository ID and source
  commit.
- Evaluation results are emitted by `scripts/run_shine_eval.py` as JSONL
  records joined to `<output>.meta.json` by `run_id`.

