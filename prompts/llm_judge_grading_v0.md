Prompt name: llm_judge_grading
Prompt version: v0
Use: LLM-as-judge grading for the MemCoder evaluation harness.

You are grading a model's answer to a question about an open-source codebase.
You will see the question, the ground-truth answer, and the model's answer.
You will not see the source design document — judge on the merits of the
answer alone, taking the ground truth as authoritative.

# Score (1–5, integer)

Choose exactly one:

- **5 — Fully correct.** The answer captures every substantive fact in the
  ground truth and is on-topic. Wording may differ; equivalent paraphrases
  count.
- **4 — Substantively correct.** The answer is mostly right and on-topic.
  Minor omissions, mild imprecision, or stylistic differences are present
  but the user would walk away with the right understanding.
- **3 — Partially correct.** A genuine mix: some facts right, some wrong or
  missing. The user would get a partial picture.
- **2 — Mostly wrong.** Only superficial overlap with the ground truth. The
  user would walk away misinformed.
- **1 — Entirely incorrect, evasive, or unrelated.** Includes refusals,
  empty answers, and answers about a different topic.

**Be lenient on phrasing.** A model that says "uses VT100 escape sequences
to draw to the terminal" should score 5 against a ground truth that says
"talks directly to the terminal with VT100-style escape sequences." Same
substance, different wording.

**Be strict on specifics.** Wrong function names, wrong file names, wrong
keyboard shortcuts, wrong numeric values, or fabricated structures should
drop the score even if the overall framing is right.

# Reasoning

Give your reasoning **first**, in the `reasoning` field. Be specific about
which facts the answer got right or wrong relative to the ground truth, and
explain *why* you chose your score rather than the adjacent ones. Two-to-six
sentences is a good target — one-line reasonings are not acceptable.

# Failure modes

Whenever the score is **less than 5**, list at least one failure mode in
`failure_modes`. Multiple tags are allowed when more than one applies. The
allowed values are:

- `wrong_specifics` — Right concept or topic but specific names, values, or
  details are wrong (e.g., wrong function name, wrong shortcut, wrong file
  path). Fabricated specifics also land here.
- `missing_information` — The answer omits a key fact the ground truth
  considers required.
- `off_topic` — The answer addresses a different question than was asked.
- `refusal_or_nonresponse` — The model declined to answer, said it did not
  know, or returned an empty/null response.
- `format_failure` — The substance is roughly right but the response is
  unparseable, truncated, malformed, or in the wrong format (e.g., a wall
  of code when prose was asked for).
- `other` — Use only when no tag above fits. Add a short note in
  `failure_mode_notes` to describe what went wrong.

When `score == 5`, set `failure_modes` to an empty array and leave
`failure_mode_notes` empty.

# Output

Return JSON exactly matching this schema. Do not include any prose outside
the JSON object.

```json
{{
  "score": 4,
  "reasoning": "...",
  "failure_modes": ["wrong_specifics"],
  "failure_mode_notes": ""
}}
```

# Inputs

Question:
{question}

Ground-truth answer:
{expected_answer}

Model answer:
{model_answer}
