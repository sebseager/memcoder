# Prompt: Ledger Example Question Generation

Version: `ledger_example_question_generation_v0`

## Purpose

Generate routing example questions for a ledger entry. These questions help a
router select the right LoRA, but they are not evaluation questions.

## Inputs

- Repository ID: `{repo_id}`
- Source commit: `{source_commit}`
- LoRA ID: `{lora_id}`
- Document ID: `{document_id}`
- Topic slug: `{topic_slug}`
- Design document: `{design_document}`
- Evaluation questions to avoid: `{eval_questions}`

## Instructions

Generate 3 to 5 example questions that this LoRA should be able to answer.
The questions must be grounded in the design document and must be different from
the provided evaluation questions.

The goal is routing, not grading. Prefer questions that clearly signal the topic
covered by the LoRA.

## Output Format

Return JSON:

```json
{
  "repo_id": "{repo_id}",
  "source_commit": "{source_commit}",
  "lora_id": "{lora_id}",
  "document_id": "{document_id}",
  "topic_slug": "{topic_slug}",
  "prompt_version": "ledger_example_question_generation_v0",
  "example_questions": [
    "..."
  ]
}
```
