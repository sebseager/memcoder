# Prompt: Doc-Derived QA Generation

Version: `doc_derived_qa_generation_v0`

## Purpose

Generate evaluation QA pairs whose answers are explicitly grounded in one design
document.

## Inputs

- Repository ID: `{repo_id}`
- Source commit: `{commit}`
- Document ID: `{document_id}`
- Topic slug: `{topic_slug}`
- Topic title: `{topic}`
- Difficulty: `{difficulty}`
- Generator: `{generator}`
- Design document: `{design_document}`

## Instructions

Generate as many useful QA pairs as the design document can support. Every answer
must be answerable from the design document alone. Do not use outside knowledge
or facts from the repository that are absent from the document.

Questions should be natural developer questions. Prefer questions that test
specific facts, relationships, responsibilities, or constraints described in the
document. Avoid adversarial trick questions.

For each QA pair:

- `qa_id` must follow the scheme `{topic_slug}_{document_id}_{n:04d}` where n is 1-indexed position within this response.
- The question must not quote the answer directly.
- The answer should be concise but complete.
- Include the source document ID.
- Mark the QA set as `doc_derived`.

## Output Format

Return JSON:

```json
{
  "repo_id": "{repo_id}",
  "commit": "{commit}",
  "document_id": "{document_id}",
  "topic": "{topic}",
  "topic_slug": "{topic_slug}",
  "difficulty": "{difficulty}",
  "generator": "{generator}",
  "prompt_version": "doc_derived_qa_generation_v0",
  "qa_pairs": [
    {
      "qa_id": "{topic_slug}_{document_id}_{n:04d}",
      "question": "...",
      "answer": "...",
      "document_id": "{document_id}",
      "qa_set": "doc_derived"
    }
  ]
}
```
