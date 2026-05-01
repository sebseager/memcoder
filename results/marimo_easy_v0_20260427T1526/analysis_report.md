# Failure-Mode Analysis — `marimo_easy_v0_20260427T1526`

This is a follow-up investigation of the `marimo_easy_v0` run (8 easy docs ×
3 doc-derived QAs × 3 conditions = 72 graded answers). The headline
numbers from `report.md`:

| Condition    | N  | Mean | %5    | 1 | 2  | 3 | 4 | 5  |
|--------------|----|------|-------|---|----|---|---|----|
| `in_context` | 24 | 4.79 | 79.2% | 0 | 0  | 0 | 5 | 19 |
| `naive`      | 24 | 2.25 |  4.2% | 8 | 5  | 9 | 1 | 1  |
| `shine`      | 24 | 2.21 |  0.0% | 6 | 11 | 3 | 4 | 0  |

After filtering to questions where `naive ≤ 2` (i.e., questions that
actually need the doc), `naive` drops to 1.38 and `shine` rises
relatively to 2.00, but in absolute terms both are well below
`in_context` at 4.77. This report investigates *why*.

The investigation covers four questions the team raised:

1. Why is SHINE doing so poorly? What are the failure modes?
2. For what questions does SHINE fail most?
3. Do document characteristics correlate with SHINE performance?
4. Why is naive performance "so high" in the first place?

I'll take (4) first because it reframes (1)–(3).

---

## 1. The "naive baseline is high" finding is largely an artefact of partial-credit drift, not pretrained knowledge

`naive` scored 2.25 mean — but the score distribution shows where that
came from:

- `5`: 1/24 (4%)
- `4`: 1/24 (4%)
- `3`: 9/24 (38%)
- `1–2`: 13/24 (54%)

So only **2/24** answers were graded as substantively correct. The bulk
of the apparent "naive performance" comes from a 9-question middle band
where the judge gave a `3` for partially-correct vague answers. Reading
the judge reasoning for those nine, a clear pattern emerges:

> "*The answer captures the general idea ... but it omits the concrete
> mechanism described in the ground truth.*"
>
> "*Partly correct but incomplete and somewhat vague.*"
>
> "*Captures the high-level concept but omits the concrete mechanisms
> that the question is really about.*"

Three concrete examples (full text in `judgments.jsonl`):

- **`overview_purpose_1_0003`** — "Why is assigning a marimo UI element
  to a global variable important?" Naive: *"...allows it to be accessed
  and manipulated throughout the code, enabling dynamic updates and
  interactions."* This is true of *any* global variable; the doc-specific
  answer ("synchronizes browser interactions back to Python so dependent
  cells rerun") is absent. Judge gave 3.
- **`public_interface_1_0001`** — "How does marimo's root package make a
  broad implementation feel like a compact public API?" Naive: *"...
  provides a minimal, focused API that abstracts away complexity..."*
  No mention of `__init__.py` or re-exports, but the rationalization is
  generic-engineering plausible. Judge gave 3.
- **`sql_data_integrations_1_0001`** — "Why are marimo SQL cells
  described as part of the Python notebook model instead of a separate
  SQL-only layer?" Naive: *"...maintain a unified and consistent
  interface...allowing seamless integration of SQL functionality..."*
  The ground truth requires "syntactic sugar for `mo.sql`". Judge gave 3.

So the high naive baseline is **not** Qwen3-8B knowing marimo internals
from pretraining (only `ui_outputs_1_0001` fits that pattern, where the
question itself contains enough hints — "global variable" + "reactivity"
— that the LLM can reason out the right answer without the doc). It's
the rubric assigning `3` for plausible-sounding, generic-engineering
restatements of the question. Two structural drivers:

1. **Question-text leakage.** Naive answers re-use 63% of the
   non-stopword content from the question, vs. 43% for SHINE. Many
   questions encode their own answer ("Why is X important?" → "X is
   important because [restate X]"). This pads the naive answer with
   on-topic vocabulary that the judge has trouble fully discounting.
2. **Question-type bias in the rubric.**
   - "why" questions: `naive` mean 3.25 — easiest to BS plausibly
   - "how" questions: `naive` mean 2.67
   - "what" questions: `naive` mean 2.10
   - "which" questions (specific entity demanded): `naive` mean 1.00

   The questions that *force* a specific named answer ("Which two
   state-synchronization behaviors...") are exactly the ones where naive
   collapses; the questions that ask for a rationale are the ones where
   the rubric is most lenient.

This is the same risk noted in `memcoder_plan.md` §18 ("QA quality
leaks into pretrained knowledge"), but with a slightly different
mechanism than what the kilo pilot showed: kilo had verbatim
identifier-recall questions answerable from identifier names; marimo
has rationale questions answerable from generic LLM priors plus
question restatement.

**Implication for the decision gate:** the unfiltered `naive ≈ shine`
finding is misleading. The `naive ≤ 2` filter (54% of questions) is the
honest read; on that set, shine 2.00 vs naive 1.38 is consistent with
shine adding signal — though both are still far below `in_context`'s
4.77.

## 2. SHINE failure modes — what shape do the wrong answers take?

The harness-level taxonomy bins SHINE failures as 20× `missing_information`,
16× `wrong_specifics`, 5× `off_topic` (multi-label, n=24). Reading every
SHINE answer side-by-side with the doc and the expected answer, I cluster
them into four narrower failure modes that are more actionable:

| Failure mode | Count | Description |
|---|---:|---|
| **Topical drift** | 6 | Answer uses doc-related vocabulary but on the wrong sub-topic from the doc |
| **Mode-collapse / short** | 7 | Answer ≤ ~80 chars, vague, often half-restating the question |
| **Wrong specifics (invented)** | 6 | Answer contains specifics that contradict or are absent from the doc |
| **Partial correct (gist captured)** | 5 | Answer captures one or two of the multi-fact ground-truth points |

A few representative cases:

**Topical drift** — `ui_outputs_interactivity_1_0002`
- Q: "What two identity mechanisms does a marimo UI element use, and
  what is each for?"
- Expected: "A stable object id ... A separate random token ..."
- SHINE: *"A marimo UI element uses a visual element and a layout
  object. The visual element is the UI object shown above the cell, and
  the layout object is the UI object returned under `marimo.ui` ..."*
- The doc *contains* the right answer (search "stable object id"). The
  LoRA picked up surrounding vocabulary ("visual output", "above the
  cell", "marimo.ui") but lost the specific concept the question asked
  about. Score: 1.

**Mode-collapse / short** — `overview_purpose_1_0001`
- Q: "Which two state-synchronization behaviors does marimo use ...?"
- Expected: "When a cell runs, marimo reruns dependent cells or marks
  them stale ... when a cell is deleted, marimo removes that cell's
  variables from program memory."
- SHINE: *"Reactivity and version control"* (29 chars).
- "Version control" is mentioned in the doc as a *property* of marimo
  (notebooks are "friendly to version control"). The LoRA latched on to
  high-salience nouns and emitted them as a fake list. Score: 1.

**Wrong specifics** — `setup_cli_1_0003`
- Q: "Which export paths run a marimo app before writing output, and
  what restriction does watch mode have?"
- Expected: "HTML exports and output-including Jupyter exports ...
  Watch mode is supported only when an output file is provided."
- SHINE: *"The export paths run a marimo app before writing output, and
  watch mode is restricted to read-only mode."*
- "Read-only mode" is fabricated; the doc does not mention it. The
  question stem was restated and a plausible-sounding constraint was
  appended. Score: 2.

**Partial correct** — `sql_data_integrations_1_0001`
- Q: "Why are marimo SQL cells described as part of the Python notebook
  model ...?"
- SHINE: *"...syntactically and semantically equivalent to Python cells
  with `mo.sql`, allowing users to write SQL directly in the same
  environment as Python code."*
- Captures the `mo.sql` syntactic-sugar point. Misses the "formatted
  Python string letting queries depend on UI controls" point. Score: 4.

### Two cross-cutting empirical observations

- **SHINE answers are dramatically shorter** than naive's. Mean answer
  length: in_context 306 chars, naive 208 chars, **shine 118 chars**.
  21/24 SHINE answers are shorter than the corresponding naive answer.
  The LoRA appears to suppress generation length — the model either
  doesn't know what else to add or has lost the calibration to keep
  going.
- **SHINE almost never lifts phrases verbatim from the source doc.**
  Mean 5-gram overlap between answer and source doc: in_context 8.42,
  shine 0.21, naive 0.00. SHINE has *some* signal (it beats naive's
  zero), and that signal correlates with score (corr=+0.42 between
  shine 5-gram overlap and shine score), but it is mostly working at
  the level of single concepts, not phrases.

Doc-specific named-entity recall confirms the same picture: SHINE
recovers 30% of the named entities (backticked tokens, CamelCase) that
appear in both the doc and the expected answer — about the same as
naive (31%). The LoRA isn't adding entity-level memorization on this
set; it's mostly contributing topical bias.

## 3. SHINE fails most on the doc-specific, multi-fact "what / which"
questions

Per-question pattern (n=24, sorted by descending shine score):

- shine = 4: 4 questions (17%) — all are "why is X structured this
  way?" or "what does X do at a high level?" questions where a single
  conceptual point captures the answer.
- shine ≤ 2: 17 questions (71%). The cluster is dominated by:
  - "**which** ..." questions asking for two or more named items
    (state-synchronization behaviors, identity mechanisms, export
    paths). SHINE gives a different list or invents one. Mean: 1.50.
  - "**what** sequence/work/extra ..." questions asking for an
    enumerated multi-step description. SHINE compresses the sequence
    into one phrase. Mean: 2.00.
  - "**how** does X make Y available ...?" — multi-stage mechanism
    questions; SHINE supplies the high-level intent but skips the
    middle. Mean: 2.17.

By contrast, "**why**" questions tolerate a single-conceptual-point
answer and SHINE scores 2.75 there.

Per-document SHINE means (3 QAs each):

| Doc | naive | in_ctx | **shine** |
|---|---:|---:|---:|
| `sql_data_integrations_1` | 2.33 | 5.00 | **3.33** |
| `public_interface_1` | 3.00 | 4.67 | **2.67** |
| `reactive_runtime_dataflow_1` | 1.67 | 4.67 | **2.67** |
| `ai_mcp_pairing_1` | 1.67 | 4.67 | **2.00** |
| `ui_outputs_interactivity_1` | 3.33 | 4.67 | **2.00** |
| `apps_scripts_wasm_1` | 2.00 | 5.00 | **1.67** |
| `overview_purpose_1` | 2.33 | 5.00 | **1.67** |
| `setup_cli_1` | 1.67 | 4.67 | **1.67** |

The bottom three docs (`apps_scripts_wasm_1`, `overview_purpose_1`,
`setup_cli_1`) all trigger the topical-drift / mode-collapse failures
above. The two top docs (`sql_data_integrations_1`,
`public_interface_1`) are the docs whose easy questions overlap most
heavily with concepts the doc actually emphasizes (the `mo.sql` story,
the `marimo/__init__.py` namespace) — i.e., the questions where one
sticky doc concept is enough to score.

## 4. Document characteristics vs. SHINE LoRA quality

Doc-level features and per-doc condition means (n=8):

| Doc | words | backticks | inline_code | identifiers | naive | shine |
|---|---:|---:|---:|---:|---:|---:|
| `ai_mcp_pairing_1` | 699 | 10 | 5 | 3 | 1.67 | 2.00 |
| `apps_scripts_wasm_1` | 699 | 8 | 4 | 2 | 2.00 | 1.67 |
| `overview_purpose_1` | 507 | 4 | 2 | 0 | 2.33 | 1.67 |
| `public_interface_1` | 584 | 22 | 11 | 9 | 3.00 | 2.67 |
| `reactive_runtime_dataflow_1` | 582 | 2 | 1 | 1 | 1.67 | 2.67 |
| `setup_cli_1` | 586 | 8 | 4 | 2 | 1.67 | 1.67 |
| `sql_data_integrations_1` | 690 | 12 | 6 | 6 | 2.33 | 3.33 |
| `ui_outputs_interactivity_1` | 651 | 14 | 7 | 7 | 3.33 | 2.00 |

Pearson correlations across docs (n=8, treat with caution):

| Feature | vs shine | vs naive | vs in_context |
|---|---:|---:|---:|
| word count | +0.19 | −0.07 | +0.09 |
| backticks | +0.32 | +0.70 | −0.27 |
| inline-code spans | +0.32 | +0.70 | −0.27 |
| identifiers (\`Name\` / CamelCase) | +0.51 | +0.76 | −0.28 |
| sentences | +0.39 | −0.08 | −0.09 |

Two observations:

- **Backtick / identifier density does not predict SHINE quality
  negatively here.** This contradicts the kilo pilot finding that
  code-density degrades SHINE. The reason is that all the marimo easy
  docs are well below the kilo "code-dense" threshold (max 0.6%
  backtick ratio vs kilo's hard cases at >2%). On the marimo set, the
  weak positive correlation is more plausibly explained by *naive
  rising on the same docs* — when a doc emphasizes well-known marimo
  concepts (`mo.sql`, `marimo.ui`), both the questions and the answers
  align with what Qwen3-8B already knows about marimo, lifting *both*
  naive and shine. The "which doc is hardest for SHINE" signal is
  therefore confounded with "which doc is hardest in general" — it's
  not a clean SHINE-quality signal at this scale.
- **Word count ≈ no effect.** All docs are short (507–699 words, well
  inside SHINE's 1150-token budget), so length headroom is not the
  bottleneck.

The doc-level signal is too weak at n=8 to draw conclusions about which
*document-writing style* helps SHINE on marimo. The clearest correlate
remains question-level: how many distinct facts does the answer
require? Multi-fact answers fail; one-concept answers sometimes work.

## 5. Synthesis — what the run actually shows

- **Naive isn't strong; the rubric is permissive.** 38% of naive
  answers got `3` for plausible generic restatements. Filtering to
  `naive ≤ 2` is the right read for assessing SHINE's contribution.
- **SHINE on this set is producing one-concept gists, not multi-fact
  recall.** It captures the *topic* of a doc, so it can answer "why is
  X structured this way?" reasonably (4 wins). It does not capture
  enumerated sequences, named pairs, or multi-step mechanisms (the
  remaining 17 losses).
- **The dominant failure mode is "topical drift + length collapse"**:
  the LoRA shifts the prior toward doc-related vocabulary, but Qwen
  emits a short, vague answer because the LoRA hasn't preserved enough
  to keep going coherently. This shows up empirically as 5-gram
  overlap with the doc near zero and answer length ~half of naive.
- **Doc characteristics don't show a clean SHINE-quality signal at
  n=8.** Backtick density is positively correlated with shine score on
  marimo, but that correlation rides on top of the same correlation
  for *naive*, so it likely reflects question/topic familiarity rather
  than LoRA fidelity.

## 6. Suggested next steps

These follow from the failure-mode analysis above; they aren't claims,
just the actions that look highest-EV given the data:

1. **Re-judge the doc-derived QAs with a stricter rubric for `3`.** The
   current "partly conceptually right" → 3 path is the single biggest
   contributor to the inflated naive baseline. Alternatively, add a
   second judge pass that re-scores anything tagged `missing_information`
   on a 0/1 "does this answer the doc-specific question" axis.
2. **Generate harder QAs that are unanswerable from question
   restatement.** Many marimo easy QAs encode their own answers in the
   question text. Ask the QA generator to produce questions that *omit*
   the conclusion and force the model to supply it. (The plan's §18
   already names this risk; this run reproduces it on a second repo.)
3. **Probe SHINE with a closed-form recall task** before running the
   open-ended QA again — e.g., "complete this sentence from the doc:
   `marimo's interaction rule reruns cells that ___`". This separates
   "LoRA preserved this fact" from "LoRA preserved the topic". Current
   results suggest the LoRA preserves topic, not facts.
4. **Investigate the length-collapse signal directly.** SHINE answers
   are systematically shorter; this could be a generation-config
   artefact (e.g., LoRA pushing logit mass toward EOS) or a genuine
   fidelity issue. Worth checking whether raising `max_new_tokens` /
   nudging the prompt to "answer in 3–4 sentences" changes scores.
5. **Document-style A/B at constant content.** The pilot's A/B re-bake
   on kilo was the right design but isn't reproduced here. With n=8
   marimo docs all at low backtick density, we cannot tell whether
   prose-vs-listy form would change marimo SHINE outcomes.

## Appendix A — Per-question table

All 24 QAs sorted by SHINE score (then naive):

| qa_id | naive | ic | shine | doc |
|---|---:|---:|---:|---|
| ai_mcp_pairing_1_0001 | 1 | 4 | 1 | ai_mcp_pairing_1 |
| overview_purpose_1_0001 | 1 | 5 | 1 | overview_purpose_1 |
| ui_outputs_1_0002 | 1 | 5 | 1 | ui_outputs_interactivity_1 |
| apps_scripts_wasm_1_0003 | 2 | 5 | 1 | apps_scripts_wasm_1 |
| setup_cli_1_0001 | 2 | 5 | 1 | setup_cli_1 |
| public_interface_1_0003 | 3 | 4 | 1 | public_interface_1 |
| apps_scripts_wasm_1_0001 | 1 | 5 | 2 | apps_scripts_wasm_1 |
| setup_cli_1_0003 | 1 | 4 | 2 | setup_cli_1 |
| reactive_runtime_dataflow_1_0001 | 2 | 4 | 2 | reactive_runtime_dataflow_1 |
| reactive_runtime_dataflow_1_0003 | 2 | 5 | 2 | reactive_runtime_dataflow_1 |
| setup_cli_1_0002 | 2 | 5 | 2 | setup_cli_1 |
| ai_mcp_pairing_1_0002 | 3 | 5 | 2 | ai_mcp_pairing_1 |
| apps_scripts_wasm_1_0002 | 3 | 5 | 2 | apps_scripts_wasm_1 |
| overview_purpose_1_0002 | 3 | 5 | 2 | overview_purpose_1 |
| overview_purpose_1_0003 | 3 | 5 | 2 | overview_purpose_1 |
| sql_data_integrations_1_0002 | 3 | 5 | 2 | sql_data_integrations_1 |
| ui_outputs_1_0003 | 4 | 4 | 2 | ui_outputs_interactivity_1 |
| ai_mcp_pairing_1_0003 | 1 | 5 | 3 | ai_mcp_pairing_1 |
| public_interface_1_0002 | 3 | 5 | 3 | public_interface_1 |
| ui_outputs_1_0001 | 5 | 5 | 3 | ui_outputs_interactivity_1 |
| reactive_runtime_dataflow_1_0002 | 1 | 5 | 4 | reactive_runtime_dataflow_1 |
| sql_data_integrations_1_0003 | 1 | 5 | 4 | sql_data_integrations_1 |
| public_interface_1_0001 | 3 | 5 | 4 | public_interface_1 |
| sql_data_integrations_1_0001 | 3 | 5 | 4 | sql_data_integrations_1 |

## Appendix B — Reproducing the numbers

Everything in this report was computed from
`results/marimo_easy_v0_20260427T1526/judgments.jsonl` and
`artifacts/marimo-team__marimo/easy/docs/*.json`. Key derived
quantities:

- Per-condition score histograms / means: from `judgments.jsonl`,
  group by `condition`, take `judge.score`.
- `naive ≤ 2` filter: handled by `scripts/plot_naive_hard.py`
  (existing); see `plots_naive_le2/summary.md`.
- 5-gram doc / answer overlap, named-entity recall, question
  restatement: re-derivable in ~30 lines of Python over
  `judgments.jsonl` + the doc text.
- Pearson correlations (n=8 docs): standard formula, reported in §4.
  All correlations are reported with the "treat with caution" caveat
  for n=8.
