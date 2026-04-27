# Prompt: Ledger Example Question Generation

Version: `ledger_example_question_generation_v1`

## Purpose

Generate routing example questions for a ledger entry. These questions
help a router (Qwen-as-router or embedding-based) select the right LoRA
for a given user query. They are **not** evaluation questions and are
never scored against a model answer.

## Inputs

- Repository ID: `{repo_id}`
- Source commit: `{commit}`
- LoRA ID: `{lora_id}`
- Document ID: `{document_id}`
- Topic slug: `{topic_slug}`
- Topic title: `{topic}`
- Generator: `{generator}`
- Design document: `{design_document}`
- Evaluation questions to avoid: `{eval_questions}`

## Instructions

Generate 3 to 5 example questions that this LoRA should be able to
answer. The questions must be grounded in the design document and must
be different from the provided evaluation questions.

The goal is **routing, not grading**. The router will see a user query
and a list of (LoRA ID, example questions) pairs. It picks the LoRA
whose example questions most closely resemble the user's query in
topic. So the example questions should make this LoRA's *topic* obvious
when scanned alongside other LoRAs' example questions.

### Topic-evocativeness over verbatim recall

A good ledger example question reads like a question a real user would
plausibly type when they are interested in this LoRA's topic. It does
not need to test pretrained-knowledge leakage the way doc-derived eval
QAs do — the router doesn't grade the LoRA's answer, only its topical
match.

That said, prefer questions that:

- **Name the central topic, subsystem, or named entity** that this
  LoRA's document is about. A question that mentions "raw mode," "the
  cleanup hook for terminal settings," or "VT100 input handling" makes
  it easy for the router to disambiguate from, say, a syntax-highlight
  LoRA.
- **Cover a spread of angles within the topic.** Don't write five
  variations of the same question — write 3–5 questions that probe
  different facets of the LoRA's content. The router benefits from
  diverse signal.
- **Are short and natural.** One sentence each. Long compound
  questions are noisy for the router.

### Disjointness from eval QAs

The evaluation questions for this LoRA are passed in `{eval_questions}`.
Your routing examples must be **semantically different** from each of
those — not just reworded. The point is to keep the routing evaluation
honest: when we measure routing accuracy at eval time, the router must
not have seen the test questions in the ledger.

Concretely: if the eval QA is "Why does kilo handle terminal input
itself instead of using curses?", do not write a routing question like
"Why does kilo not use curses for terminal input?" — that's a
paraphrase. Instead pick a different facet of the same LoRA, e.g.,
"How does kilo recover its original terminal settings on exit?"

### Avoid pretrained-knowledge phrasing

Routing questions are not evaluation, but they will sometimes be passed
through an embedder or an LLM router. If a routing question is phrased
generically (e.g., "What is a common way to handle terminal input?"),
it will embed close to many unrelated LoRAs and degrade routing
accuracy. Anchor each question to the specific topic of *this* LoRA.

## Output Format

Return JSON:

```json
{
  "repo_id": "{repo_id}",
  "commit": "{commit}",
  "lora_id": "{lora_id}",
  "document_id": "{document_id}",
  "topic": "{topic}",
  "topic_slug": "{topic_slug}",
  "generator": "{generator}",
  "prompt_version": "ledger_example_question_generation_v1",
  "example_questions": [
    "..."
  ]
}
```
