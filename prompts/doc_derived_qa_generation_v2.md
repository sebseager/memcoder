# Prompt: Doc-Derived QA Generation

Version: `doc_derived_qa_generation_v2`

## Purpose

Generate exactly three evaluation QA pairs whose answers are explicitly grounded
in one design document. These QAs measure whether a SHINE LoRA built from the
document preserves enough information to answer real questions about the topic.

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

You are a QA subagent in a local orchestrator/subagent workflow. Generate exactly
3 useful QA pairs. Every answer must be answerable from the design document
alone. Do not use outside knowledge or facts from the repository that are absent
from the document.

Questions should be natural developer questions. Prefer questions that test
specific facts, relationships, responsibilities, or constraints described in the
document. Avoid adversarial trick questions. If the document supports more than
3 possible questions, choose the 3 highest-signal questions for evaluating
whether the document was preserved.

The harness scores answers on a 1–5 integer scale via an LLM judge that
sees only the question, the ground-truth answer, and the model's answer.
The judge is told to be lenient on phrasing and strict on specifics, with
a specificity floor that caps vapor answers at score 1 and vague-but-
directional answers at score 2. Write QAs that align with that rubric:
**the question and the ground-truth answer should make it unambiguous
what a fully-correct response looks like, with substance that goes
beyond what a generic LLM already knows.**

### Avoid pretrained-knowledge leakage (binding)

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

### Identifier-targeted vs non-identifier-targeted questions (v2 change)

The pilot eval surfaced that SHINE preserves *identifier associations*
much more cleanly than *multi-fact mechanism explanations*. To support
stratified analysis of this finding across repos, every QA must be
tagged on the `identifier_targeted` axis.

**A QA is `identifier_targeted: true`** when the ground-truth answer
hinges on a specific named entity from the document — a function name,
struct, type, file path, configuration key, command, environment
variable, or fixed value/constant — and the user could not score full
marks without naming that entity (or a close paraphrase of it).

Examples of identifier-targeted Qs:
- "Which global structure holds most of kilo's editor state?" → `struct editorConfig`
- "Which CLI command opens a notebook for editing?" → `marimo edit`
- "What helper does marimo expose for adding to existing output rather than replacing it?" → `mo.output.append`

**A QA is `identifier_targeted: false`** when the ground-truth answer
is a conceptual explanation, a sequence of steps, a rationale, or any
multi-fact answer that does not turn on naming one specific entity.
These questions test whether the LoRA preserved a mechanism, not a
name.

Examples of non-identifier Qs:
- "Why is assigning a marimo UI element to a global variable important?"
- "What sequence does the document describe when DirectedGraph registers a new cell?"
- "How does kilo react to terminal resize events?"

#### Soft floor: at least one identifier-targeted question per
#### identifier-rich document

Count distinct identifiers in the source document. If the document
names **two or more** distinct identifiers (function names, struct
names, type names, configuration keys, command names — anything
appearing in backticks or in conventional CamelCase / snake_case),
**at least one of the three QA pairs must be identifier-targeted.**

If the document names fewer than two distinct identifiers (e.g., a
high-level overview document with only prose), this floor does not
apply — generate the most natural three questions, all of which will
likely be `identifier_targeted: false`. Mark this case explicitly in
the optional `qa_set_notes` field if it applies.

For most identifier-rich
documents the natural mix will be 1–2 identifier-targeted out of 3,
which is fine.

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
- What identifier holds / produces / consumes a specific value
  described in the document? (good identifier-targeted shape)

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
- Set `identifier_targeted` to `true` or `false` per the rules above.

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
  "prompt_version": "doc_derived_qa_generation_v2",
  "qa_set_notes": "",
  "qa_pairs": [
    {
      "qa_id": "{topic_slug}_{document_id}_{n:04d}",
      "question": "...",
      "answer": "...",
      "document_id": "{document_id}",
      "qa_set": "doc_derived",
      "identifier_targeted": true
    }
  ]
}
```

The `qa_set_notes` field is optional; populate it (one short sentence)
only when the source document had fewer than two distinct identifiers
and the soft floor for identifier-targeted questions therefore did not
apply. Otherwise leave it empty.
