# Prompt: Orchestrator Artifact Generation

Version: `orchestrator_artifact_generation_v0`

## Purpose

Coordinate local artifact generation for one target repository using an
orchestrator/subagent workflow. This prompt is for the orchestrator. Topic,
document, QA, and routing-example subagents use the narrower prompt templates in
this directory.

## Inputs

- Repository ID: `{repo_id}`
- Repository path: `{repo_path}`
- Source commit: `{commit}`
- Difficulty: `{difficulty}`
- Generator: `{generator}`
- Universal seed topics: `{seed_topics}`
- Token budget per design document: `{token_budget}`
- Artifact root: `{artifact_root}`

## Instructions

You are the orchestrator for a local code QA artifact-generation run. Do not
call a hosted LLM API. Keep repository contents local and coordinate bounded
subagent tasks.

Run the workflow in resumable stages:

1. Build a compact repository context pack from the README, docs index,
   top-level tree, package/module tree, and development docs that appear
   relevant.
2. Use `topic_discovery.md` to produce `{artifact_root}/{repo_id}/topics.json`.
3. For each selected topic, build a topic-specific evidence pack. Prefer
   directly relevant docs, source files, tests, and search results over broad
   repository dumps.
4. Give one topic-specific evidence pack to a document subagent using
   `design_doc_generation.md`. Write the result to
   `{artifact_root}/{repo_id}/{difficulty}/docs/{document_id}.json`.
5. Give each generated document to a QA subagent using
   `doc_derived_qa_generation.md`. Write exactly three evaluation QA pairs to
   `{artifact_root}/{repo_id}/{difficulty}/qas/{document_id}.json`.
6. Give each generated document and its evaluation questions to a routing-example
   subagent using `ledger_example_question_generation.md`. Write the routing
   examples to
   `{artifact_root}/{repo_id}/{difficulty}/qa_examples/{document_id}.json`.
7. Merge the generated files into `{artifact_root}/{repo_id}/ledger.json` and
   `{artifact_root}/{repo_id}/{difficulty}/ledger.md`.
8. Validate every JSON artifact against the schemas before evaluation.

## Merge Rules

- Use stable `document_id` values with the scheme `{topic_slug}_{n}`.
- Treat `document_id` as the default `lora_id` unless a later LoRA-generation
  step creates a different ID.
- Copy each design document's `description` into the matching `ledger.json`
  entry and the human-readable ledger.
- Keep doc-derived evaluation QAs out of `ledger.md`.
- Include routing example questions in `ledger.md`, sourced only from
  `qa_examples/{document_id}.json`.
- Leave `files.lora` as `null` until SHINE LoRA generation writes the weights.
- Leave `files.doc_embedding` as `null` until embedding generation runs.

## Output

When the run is complete, report:

- The source commit.
- The topic count and document count.
- The artifact paths written.
- Any topics skipped and why.
- Validation status.
