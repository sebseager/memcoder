# Prompt: Design Document Generation

Version: `design_doc_generation_v1`

## Purpose

Generate one natural-language design document for a single repository topic plus
a short description of what the document explains. The document will be
compressed into a SHINE LoRA and used at inference time to answer developer
questions about the topic without the document being in context.

## Inputs

- Repository ID: `{repo_id}`
- Repository path: `{repo_path}`
- Source commit: `{commit}`
- Generator: `{generator}`
- Topic slug: `{topic_slug}`
- Topic title: `{topic_title}`
- Difficulty: `{difficulty}`
- Token budget: `{token_budget}`
- Topic-specific evidence pack provided by the orchestrator: `{repo_context}`

## Instructions

You are a topic subagent in a local orchestrator/subagent workflow. Write a prose
design document about the requested topic using only the topic-specific evidence
pack provided by the orchestrator. The document is the *only* thing the SHINE
hypernetwork sees about this topic — it must be self-contained and factually
dense, but **must not become a code listing or a catalog of identifiers**. The
pilot established that catalog-style docs produce LoRAs that degrade the model's
output (gibberish loops, invented identifiers). Prose-first docs at the same
conceptual content produce visibly better LoRAs.

Hard requirements:

- Stay strictly under `{token_budget}` tokens.
- Ground every claim in the provided repository context. Do not speculate
  about behavior that is not in the source material.
- Explain the purpose of the topic and the important implementation details.
- Mention concrete files, functions, or data structures when they matter, while
  following the style requirements below.
- Write for a developer who needs to answer questions about this
  repository without rereading the code.
- Include a short `description` field summarizing what questions this document
  is useful for routing or retrieval.

### Style requirements (binding — these are the v1 changes)

- **Prose-first.** Each paragraph should describe a coherent subsystem at
  the level of "what this does and why," not "step 1, step 2, step 3."
  Aim for the style of the kilo `overview_purpose` reference document:
  paragraphs that name the major actors (functions, structs, fields)
  once where they belong and describe their purpose alongside them.
- **Backtick budget: target ≤ 2.5% of word count, hard cap 4%.**
  In a 600-word document that's roughly 15 backticked terms (target) and
  24 (cap). Count every \`identifier\`, \`FLAG_NAME\`, \`CODE_LITERAL\`,
  and \`function()\` that appears between backticks.
- **No enumeration of flags, constants, or struct fields by name.**
  If the topic touches a set of flags, escape codes, or members,
  describe them by *what they do* in one sentence rather than listing
  the symbol names. For example, write "kilo disables echo, canonical
  line buffering, signal generation from control keys, and extended
  input processing" instead of "kilo clears `ECHO`, `ICANON`, `IEXTEN`,
  and `ISIG`." This dramatically improves SHINE compression and does
  not lose meaning at the easy / medium difficulty levels.
- **Name a function or struct once at most when introducing it,** then
  refer back to it by its role. Prefer "the cleanup hook" over
  repeating `editorAtExit` four times.
- **Describe purpose alongside the named entity.** When you do mention
  an identifier, attach a why or what to it in the same sentence.
  "`enableRawMode` captures the current terminal settings and applies
  attributes tuned for byte-by-byte editing" — not just
  "`enableRawMode` calls `tcgetattr` and `tcsetattr`."
- **Avoid quoting code, error messages verbatim, or numeric values when
  prose can convey the same meaning.** "Returns within roughly a tenth
  of a second" is preferred over "`VMIN=0`, `VTIME=1`."

For `easy` difficulty, focus on high-level behavior and the most important
facts. For `medium`, retain the prose style but include the specific facts
that genuinely require an exact identifier (function name, well-known
constant). For `hard`, you may exceed the backtick cap if needed, but
follow the same prose-first structural rules.

### Quick style check before returning

Before returning, verify:

1. Word count is roughly 400–700 (or the difficulty band's expectation).
2. Backtick density is ≤ 4% (count backticked spans / word count).
3. There is no list, bullet, or comma-separated catalog of identifiers
   longer than 3 items.
4. Every paragraph could be read aloud as a paragraph; none reads as
   step-by-step procedure.
5. Every named entity has a "what" or "why" attached in the same sentence.

## Output Format

Return JSON:

```json
{
  "repo_id": "{repo_id}",
  "commit": "{commit}",
  "document_id": "{document_id}",
  "topic": "{topic_title}",
  "topic_slug": "{topic_slug}",
  "difficulty": "{difficulty}",
  "generator": "{generator}",
  "prompt_version": "design_doc_generation_v1",
  "description": "Short summary of what this document explains and what kinds of questions it can answer.",
  "document": "..."
}
```
