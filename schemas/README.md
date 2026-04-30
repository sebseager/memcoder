# Schemas

Schemas live as JSON Schema files in this directory. The JSON fields are the
canonical artifact contracts; examples in prompts should match these schemas.

Important contracts:

- Design document JSON must contain `document`; the full JSON object is passed
  through as `doc_metadata` during evaluation.
- QA JSON must contain top-level `qa_pairs`; each QA record must contain
  `question` and `answer`. The full QA record is passed through as
  `qa_metadata`.
- Repo-level `topics.json` records the audited topic set before document
  generation starts.
- Repo-level `ledger.json` maps each `document_id` to difficulty, topic
  metadata, and file paths for docs, LoRAs, QA pairs, routing examples, and
  embeddings.
- `judge_result.schema.json` describes the structured score, reasoning, and
  failure-mode payload returned by the LLM judge.
- `eval_result.schema.json` covers legacy JSONL result records. The newer
  `scripts/run_eval.py` pipeline writes `predictions.jsonl`, `judgments.jsonl`,
  `manifest.json`, and `run_config.yaml` into `results/<run_name>_<timestamp>/`.
