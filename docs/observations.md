# Evaluation Observations


The findings below are from the easy-difficulty pilot evaluation on two
repositories: `marimo-team__marimo` (8 documents, 24 QAs) and
`antirez__kilo` (3 documents, 43 QAs). All numbers are under the v1 judge
rubric (`prompts/llm_judge_grading_v1.md`). Source artifacts are under
`results/`; the v1 rubric, A/B summary, and rubric-impact comparison live
in `results/prompt_ab_v1_*` and `results/rubric_impact_*`.

### 1. SHINE preserves identifier associations, not multi-fact mechanism explanations

This is the cleanest finding in the data and it's repo-agnostic.
Stratifying questions by whether the ground-truth answer names a specific
identifier (a backticked token like `\`struct editorConfig\`` or a CamelCase
identifier):

| Repo | Stratum | n | naive | shine | gap |
|---|---|---:|---:|---:|---:|
| marimo | identifier-targeted | 3 | 2.00 | 2.33 | +0.33 |
| marimo | non-identifier | 21 | 2.05 | 2.00 | −0.05 |
| kilo | identifier-targeted | 10 | 2.00 | **3.00** | **+1.00** |
| kilo | non-identifier | 33 | 2.48 | 2.55 | +0.06 |

The kilo identifier-targeted gap of +1.00 (one full point on a 5-point
scale) is roughly 16× larger than the non-identifier gap (+0.06) on the
same repo. SHINE's value is concentrated on questions of the form "what
is the name / value / location of X?", not "how does Y work
conceptually?". The implication: the project's headline claim should be
about identifier-grounded recall, not general document-content recall.

The marimo identifier-targeted bucket has only n=3, which is too small
to draw repo-level conclusions on its own; the pattern holds in
direction but the QA generation pipeline did not produce enough
identifier-targeted questions for marimo to power a clean test (see
"Implications for QA generation" below).

### 2. SHINE can hit score 5, but only on single-fact ground truths

| Repo | SHINE score≥4 (n) | of which 5s |
|---|---:|---:|
| marimo | 3/24 (12.5%) | 0 |
| kilo | 11/43 (26%) | 4 |

All 4 kilo 5s are on single-fact ground truths:

- "How is kilo invoked?" → `kilo <filename>`
- "Which global structure holds editor state?" → `struct editorConfig`
- "What is kilo and what is its intended purpose?" → prose summary
- "Where are the original terminal settings kept?" → file-scope variable

Marimo's ground truths typically pack 3–4 facts ("rerun dependents +
mark stale + clear deleted globals + on-cell-deletion"). SHINE's recall
is single-fact / single-concept, so it ceilings at 4 against multi-fact
ground truths even when the one fact it produces is correct. This is
not a rubric problem — it's an honest measurement of what the LoRA
preserves.

### 3. SHINE commits to specifics; naive emits vapor

The v1 rubric splits the prior `missing_information` failure-mode tag
into `no_specifics` (vapor — answer commits to no ground-truth-style
fact) and `missing_information` (answer commits to some specifics but
omits required ones). Under v1 baseline-prompt:

| Repo | Cond | `no_specifics` rate | `wrong_specifics` rate |
|---|---|---:|---:|
| marimo | naive | 62% | 46% |
| marimo | shine | 46% | 42% |
| kilo | naive | 47% | 53% |
| kilo | shine | **23%** | 49% |

Kilo SHINE has *half* the `no_specifics` rate of kilo naive (23% vs
47%), and roughly the same `wrong_specifics` rate. The LoRA is
*reliably nudging Qwen to commit to specifics* — when it gets them
right it scores; when it gets them wrong it's no worse than naive.
The user receives committed information rather than confident-sounding
vapor.

### 4. Per-document gaps reveal question-style sensitivity

Marimo per-doc SHINE-vs-naive gaps under v1 (sorted by gap):

| doc | naive | shine | gap |
|---|---:|---:|---:|
| `sql_data_integrations_1` | 2.00 | 3.00 | **+1.00** |
| `reactive_runtime_dataflow_1` | 1.67 | 2.67 | **+1.00** |
| `setup_cli_1` | 1.33 | 2.00 | +0.67 |
| `public_interface_1` | 2.33 | 2.67 | +0.33 |
| `apps_scripts_wasm_1` | 2.00 | 1.67 | −0.33 |
| `ai_mcp_pairing_1` | 1.67 | 1.33 | −0.33 |
| `overview_purpose_1` | 2.00 | 1.33 | −0.67 |
| `ui_outputs_interactivity_1` | **3.33** | **1.67** | **−1.67** |

`ui_outputs_interactivity_1` is the worst-case for SHINE: high
concept-density doc + multi-fact rationale questions ("Why does marimo
require an interacted UI element to be bound to a global variable for
notebook reactivity?"). Naive's plausible paraphrases get partial
credit; SHINE's topical-drift failure mode lands at 1.

This is the same pattern as Observation 1 surfaced at the document
level: the docs SHINE wins on (`sql_data_integrations`,
`reactive_runtime_dataflow`) name concrete identifiers and mechanisms;
the docs SHINE loses on are conceptual / mechanism-explanation docs.

### 5. Repo-level: kilo is dramatically more amenable to SHINE than marimo

| | marimo | kilo |
|---|---:|---:|
| mean words/doc | 625 | 538 |
| mean identifiers/doc | **3.8** | **10.7** |
| % of expected answers naming an identifier | 12.5% | 23% |
| naive baseline (v1) | 2.04 | 2.37 |
| shine baseline (v1) | 2.04 | **2.65** |

Kilo docs are ~3× more identifier-dense and have ~2× the rate of
identifier-targeted questions. The SHINE-vs-naive gap is 0.00 on marimo
but +0.28 on kilo (full set), and +0.30 vs +0.61 on the naive-le2
filtered subset. The kilo profile (small, identifier-rich,
single-purpose code) is the regime where SHINE delivers; the marimo
profile (large, prose-style design docs covering multi-component
architectures) is the regime where SHINE struggles.

### 6. Prompt-engineering interventions did not unlock hidden knowledge

Two alternative system prompts were tested against the baseline (which
asks Qwen to "answer concisely; output only the final answer"):

- `detail` — drops "concisely / final answer", asks the model to commit
  to specifics and list multiple items if asked.
- `adapted` — `detail` with an additional leading sentence: "Your
  weights have been adapted to encode a specific document about a code
  repository. Draw on that adapted knowledge to answer."

The `adapted` framing was inspired by recent work on introspective
awareness of mechanistically-injected concepts (Macar et al.,
*Mechanisms of Introspective Awareness*, arXiv:2603.21396), which
shows models can detect injected steering vectors. We tested whether
the same framing transfers to LoRA-adapted models.

Result: the absolute score went *down* under both prompts (mean answer
score 2.21 → 1.96 detail → 2.04 adapted on marimo SHINE, similar on
kilo). The adapted prompt was consistently slightly better than detail
(+0.08–0.09 across both repos), but neither beat baseline on absolute
mean score. This is consistent with the paper's finding that
introspection is mostly a training-stage property — prompting alone
has limited marginal effect.

However, under the v1 judge rubric, the `detail` and `adapted` prompts
**widen the SHINE-vs-naive gap** even though absolute scores drop:

| Prompt | marimo gap (shine − naive) | kilo gap |
|---|---:|---:|
| baseline | +0.00 | +0.28 |
| detail | +0.17 | +0.60 |
| adapted (vs naive detail) | +0.26 | +0.65 |

So even though absolute scores are lower, the `detail`/`adapted`
prompts produce a *cleaner separation* between SHINE and naive — they
force both models to commit to specifics, which exposes naive's
knowledge gaps faster than SHINE's. For evaluation purposes (where the
gap matters more than the absolute number), `detail` or `adapted` is
arguably the more honest measurement; for end-user use (where
hedge-friendly answers might be preferable), `baseline` is more
forgiving.

## Implications for QA generation

The identifier-targeted vs non-identifier-targeted finding (Observation
1) is the strongest signal in the project, but the marimo
identifier-targeted bucket has only n=3 — too small for clean
inference. The QA generation pipeline currently produces whatever
question shapes feel natural per document, with no constraint on the
identifier-targeted fraction. The natural distribution under-samples
the question type that most clearly demonstrates SHINE's value.

The recommended change to `prompts/doc_derived_qa_generation.md`:

- Tag each generated QA with `identifier_targeted: bool` (true when
  the ground truth names a specific identifier or value from the doc).
- Require at least one identifier-targeted question per doc when the
  doc names ≥2 distinct identifiers (most marimo docs qualify;
  `overview_purpose_1` and `reactive_runtime_dataflow_1` may not).
- Do **not** force a hard 50/50 split — the goal is statistical power
  on the stratified analysis, not a contrived balance that produces
  stilted questions.

Stratification reporting should then become standard in the eval
report: per-cell means broken out by `identifier_targeted` so the
gap-on-identifier-Qs claim can be made repo-by-repo with adequate n.
