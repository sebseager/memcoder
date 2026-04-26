# Prompt: Design Document Generation

Version: `design_doc_generation_v0`

## Purpose

Generate one natural-language design document for a single repository topic.

## Inputs

- Repository ID: `{repo_id}`
- Repository path: `{repo_path}`
- Source commit: `{source_commit}`
- Topic slug: `{topic_slug}`
- Topic title: `{topic_title}`
- Difficulty: `{difficulty}`
- Token budget: `{token_budget}`
- Files or summaries provided by the caller: `{repo_context}`

## Instructions

Write a prose design document about the requested topic. The document will be
compressed into a SHINE LoRA, so it should be self-contained and factually dense
without becoming a code listing.

Requirements:

- Stay under `{token_budget}` tokens.
- Ground all claims in the provided repository context.
- Explain the purpose of the topic and the important implementation details.
- Mention concrete files, functions, or data structures when they matter.
- Avoid quoting large code blocks.
- Avoid speculation about behavior not shown in the repository.
- Write for a developer who needs to answer questions about this repository
  without rereading the code.

For `easy` difficulty, focus on high-level behavior and the most important facts.

## Output Format

Return JSON:

```json
{
  "repo_id": "{repo_id}",
  "source_commit": "{source_commit}",
  "topic_slug": "{topic_slug}",
  "topic_title": "{topic_title}",
  "difficulty": "{difficulty}",
  "prompt_version": "design_doc_generation_v0",
  "document": "..."
}
```
