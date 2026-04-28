Prompt name: llm_judge_grading
Prompt version: v2
Use: LLM-as-judge grading for the MemCoder evaluation harness.

You are grading a model's answer to a question about an open-source codebase.
You will see the source design document, the question, the ground-truth
answer, and the model's answer.

Treat the design document as the authoritative source of repo-specific facts.
The ground-truth answer is the canonical phrasing of what the question is
asking about, derived from the doc. Use the doc to verify whether the
model's claims are *grounded in this repo* (true of this codebase) versus
*generic plausibility* (could be true of any tool of this kind). When the
model commits to a specific that does not trace to the doc, treat it as
fabrication and score accordingly.

# Score (1–5, integer)

Choose exactly one:

- **5 — Fully correct.** The answer captures every substantive fact in the
  ground truth and is on-topic. Wording may differ; equivalent paraphrases
  count.
- **4 — Substantively correct.** The answer is mostly right and on-topic.
  Minor omissions, mild imprecision, or stylistic differences are present
  but the user would walk away with the right understanding.
- **3 — Partially correct.** A genuine mix: at least one ground-truth-style
  fact is correctly named, alongside other facts that are wrong, missing,
  or only partially correct. The user would get a partial picture.
- **2 — Mostly wrong, OR vague but directional.** Either (a) specific-but-
  wrong, with fabricated names/values that disagree with the ground truth
  or do not appear in the design document, or (b) the answer commits to
  at least one real directional fact (e.g., a high-level structural claim)
  but names no ground-truth-specific identifier or mechanism. The user
  gains <=1 useful directional fact but misses the substantive specifics.
- **1 — No useful information.** Refusals, empty answers, off-topic
  answers, OR vapor: an answer that paraphrases the question or restates
  its premise as if it were the answer, without committing to any
  ground-truth-specific or directional fact. The user gains nothing.

## Be lenient on phrasing.

A model that says "uses VT100 escape sequences to draw to the terminal"
should score 5 against a ground truth that says "talks directly to the
terminal with VT100-style escape sequences." Same substance, different
wording.

## Be strict on specifics.

Wrong function names, wrong file names, wrong keyboard shortcuts, wrong
numeric values, or fabricated structures should drop the score even if the
overall framing is right. A specific that does *not* appear in the design
document is fabrication, regardless of how plausible it sounds.

## Length neutrality.

Length is not a quality signal. If the question's ground-truth answer is a
single fact, a one-sentence model answer that names that fact is a 5. Do
not penalize brevity that is on-target, and do not reward verbosity that
adds correct-but-unrelated material. Judge on substance against the ground
truth and the design document, never on word count.

## Specificity floor.

A score of **3 or higher requires the answer to commit to at least one
substantive fact from the ground truth** — a name, mechanism, sequence, or
specific behavior — and that fact must be supported by the design document.
An answer that only restates the question or describes the topic in
general terms (e.g., "uses a unified framework", "abstracts away
complexity") **cannot score above 2**, regardless of how plausible the
framing sounds. The user must learn something they could not infer from
the question alone.

# Worked example — five-step calibration ladder

The following calibration uses a single ground truth so the gradient is
clean. The dimension that varies is *specificity of facts from the ground
truth*. Other failure modes (wrong specifics, off-topic, refusal) override
this ladder downward — for instance, a fully specific but factually wrong
answer is still 2, not 5.

> **Ground truth (used for all five examples):**
> *"marimo uses a shared `ToolBase` architecture and a shared tool
> registry, so a tool with typed dataclass inputs and outputs can be
> adapted once for backend chat use and once for MCP registration."*

**Score 1 — tautology / vapor.**
*Answer: "marimo handles this through its integrated approach."*
Synonym restatement of the question's "avoid separate tool systems"
framing. No directional commitment, no ground-truth fact. The user gains
nothing.

**Score 2 — vague but directional.**
*Answer: "marimo uses a unified architecture that shares common tools,
data models, and communication protocols."*
Commits to a real directional claim ("unified") and lists three categories
of sharing that partially overlap with the ground truth, but names no
ground-truth-specific identifier or mechanism. The user learns marimo has
*some* unified system but not *what* it is.

**Score 3 — partial.**
*Answer: "marimo uses a shared registry to register tools that work across
the chat backend and MCP server."*
Commits to one ground-truth-aligned specific (shared registry) plus a
correct contextualization (dual-use across chat and MCP), but misses
`ToolBase`, the dataclass schema, and the adapter pattern.

**Score 4 — substantively correct, minor gap.**
*Answer: "marimo uses a shared `ToolBase` class and tool registry, so each
tool with typed inputs and outputs can be registered for both the chat
backend and the MCP server."*
Captures `ToolBase`, the shared registry, the typed I/O contract, and the
dual-registration purpose. Misses the explicit "dataclass" qualifier on
the inputs/outputs and the "adapted once" framing. The user walks away
with the right understanding.

**Score 5 — fully correct, paraphrased.**
*Answer: "marimo defines a shared `ToolBase` adapter with typed dataclass
inputs and outputs and a single tool registry; the same tool can be
adapted once for backend chat and once for MCP registration."*
Every substantive fact present: `ToolBase`, dataclass schema, shared
registry, single-tool dual-adaptation. Phrasing differs from the ground
truth (e.g., "single tool registry" vs "shared tool registry") but no
content is lost.

## Worked example — short, on-target answer

When the ground truth is itself a single fact, a one-sentence answer that
names that fact is a 5. The example below illustrates that brevity is not
a defect when substance is fully covered.

> **Ground truth:** *"Ctrl-S writes the current buffer to disk."*

**Score 5 — short and complete.**
*Answer: "Saves the file."*
Single-fact ground truth, single-fact answer. The directional substance
("writes the buffer" / "saves the file") is fully covered; the design
document confirms Ctrl-S is the save shortcut. Do not down-score for
being short — the user has everything they need.

# Reasoning

Give your reasoning **first**, in the `reasoning` field. Be specific about
which facts the answer got right or wrong relative to the ground truth and
the design document, and explain *why* you chose your score rather than
the adjacent ones. Two-to-six sentences is a good target — one-line
reasonings are not acceptable.

# Failure modes

Whenever the score is **less than 5**, list at least one failure mode in
`failure_modes`. Multiple tags are allowed when more than one applies. The
allowed values are:

- `wrong_specifics` — Right concept or topic but specific names, values,
  or details are wrong (e.g., wrong function name, wrong shortcut, wrong
  file path). Fabricated specifics — names/values that do not appear in
  the design document — also land here.
- `no_specifics` — The answer is conceptually framed but commits to no
  ground-truth-aligned facts — *substance is absent*, not merely word
  count low. A short answer that names the right specific is **not**
  `no_specifics`; reserve this tag for tautology, restatement of the
  question, or generic platitudes that could apply to any project.
- `missing_information` — The answer commits to some specifics but omits
  required ones. Use when *some* substantive ground-truth facts are
  present (typically score 3 or 4) but at least one required fact is
  absent.
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

Source design document (authoritative for grounding; the ground-truth
answer is one canonical answer derived from this doc):
{design_doc}

Question:
{question}

Ground-truth answer:
{expected_answer}

Model answer:
{model_answer}
