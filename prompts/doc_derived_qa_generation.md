# Prompt: Doc-Derived QA Generation

Version: `doc_derived_qa_generation_v1`

## Purpose

Generate evaluation QA pairs whose answers are explicitly grounded in one
design document. These QAs measure whether a SHINE LoRA built from the
document preserves enough information to answer real questions about the
topic.

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

Generate as many useful QA pairs as the design document can support. Every
answer must be answerable from the design document alone. Do not use
outside knowledge or facts from the repository that are absent from the
document.

Questions should be natural developer questions. Prefer questions that
test specific facts, relationships, responsibilities, or constraints
described in the document. Avoid adversarial trick questions.

The harness scores answers on a 1–5 integer scale via an LLM judge that
sees only the question, the ground-truth answer, and the model's answer.
The judge is told to be lenient on phrasing and strict on specifics.
Write QAs that align with that rubric: **the question and the
ground-truth answer should make it unambiguous what a fully-correct
response looks like, with substance that goes beyond what a generic LLM
already knows.**

### Avoid pretrained-knowledge leakage (binding — this is the v1 change)

The pilot found that ~21% of doc-derived QAs were answerable by base
Qwen3-8B with no document and no LoRA, because the answers happened to
be:

- **Standards the model has memorized.** ANSI / VT100 escape codes,
  POSIX `errno` constants, common HTTP status codes, well-known file
  formats. If the answer is a public-domain code or constant any LLM
  knows, the question is testing pretrained knowledge — not the doc.
- **Facts inferable from a self-documenting function name.** A question
  like "what does `editorInsertNewline` do?" answers itself from the
  name. The model gets the gist right without ever seeing the doc.
- **Generic conventions.** "How does an editor render tabs?" — almost
  any plausible answer scores well because tabs-as-spaces is the
  industry default.

Before keeping a QA pair, run this **self-check**:

> If a strong general-purpose LLM never saw this design document, would
> it answer this question correctly anyway from common knowledge or
> from the question phrasing alone?

If the answer is "yes" or "probably," the QA pair leaks. Either rewrite
it to depend on a non-obvious specific from the doc, or drop it.

### Prefer questions that probe document-specific content

Strong question shapes:

- Why did this codebase make a particular choice? (e.g., "Why does
  kilo handle terminal input itself instead of relying on a UI library?")
- What is the relationship between two named entities?
- What is the specific contract or behavior in *this* implementation
  that a generic implementation might handle differently?
- Under what condition does a function take a non-default path, and
  what does that path do?
- What is the data flow from one component to another?

Weak (leaky) question shapes — avoid:

- "What is the standard ANSI escape sequence for X?"
- "What does the function `editorInsertChar` do?" (name self-documents)
- "What is the value of constant `ENOENT`?" (POSIX standard)
- "What is the typical convention for X in editors?"

### Keep ground-truth answers grounded but precise

- Concise but complete. One or two sentences for most easy questions.
- Quote a specific identifier, function name, or value **only when the
  question genuinely requires it** and the doc itself names it.
- Avoid loose hedging that the judge might accept a wrong answer
  against. "kilo subtracts two from `E.screenrows` to make room for
  status UI" is better than "kilo reserves some rows at the bottom."

### Per-QA fields

- `qa_id` must follow the scheme `{topic_slug}_{document_id}_{n:04d}`
  where n is the 1-indexed position within this response.
- The question must not quote the answer directly.
- The answer should be concise but complete.
- Include the source `document_id`.
- Mark `qa_set` as `doc_derived`.

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
  "prompt_version": "doc_derived_qa_generation_v1",
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
