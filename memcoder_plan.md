# Project Plan: Local Code QA via SHINE-Generated LoRAs

## 1. Motivation and Goal

We aim to build a harness that allows small, locally-run language models to substantively answer questions about a code repository whose contents the user does not want to share with external AI providers. The system uses the SHINE hypernetwork (built on Qwen3-8B) to compress prosaic design documents about a repository into LoRA adapters. At question time, the system retrieves the most relevant LoRA(s), loads them into Qwen3-8B, and answers using the resulting adapted model.

The contribution we are testing: whether a SHINE-based system can serve as a viable, fully-local alternative to either (a) sending repository content to a hosted model or (b) running RAG locally over raw documents. The privacy framing constrains every design choice — no component may require sending repository content off-device at inference time.

## 2. System Overview

The system has two phases.

**Indexing phase (one-time per repository).** An agent walks the repository and produces a set of natural-language design documents organized by topic. Each document is converted into a LoRA adapter via the SHINE hypernetwork. Each LoRA is registered in two places: a **ledger** (a flat markdown file with topic sections, intended to be read by the local model when routing) and a **LoRA store** (a separate registry holding all system-level metadata: weights file, source document, generation provenance, embeddings, etc.). The ledger is deliberately lean — it contains only what the model needs to pick the right LoRA. Embeddings used for retrieval are computed from a richer string (described in §9) and stored in the LoRA store, not in the ledger.

**Query phase.** A user poses a natural-language question. A router selects the most relevant LoRA(s) from the ledger. The selected LoRA(s) are loaded into Qwen3-8B (averaged if multiple), and the model produces an answer. No code or full document text is sent to the model at inference time — only the question and the loaded LoRA weights.

## 3. Scope

**In scope:**
- Three small repositories from distinct domains for pilot work.
- Approximately ten small-to-medium repositories for the main evaluation.
- Three difficulty levels of design documents (easy/medium/hard) and matched QA difficulty.
- Two QA generation strategies: doc-derived and blind (codebase-derived without seeing docs).
- Three routing strategies: ground-truth oracle, Qwen-as-router, embedding-based retrieval.
- Doc generation by both Claude Code (or Codex) and by Qwen3-8B itself.
- LoRA composition via averaging of top-k retrieved LoRAs.

**Out of scope (explicit non-goals):**
- Modifying or retraining the SHINE hypernetwork itself.
- Cross-repository question answering.
- Code-in-context approaches (we feed only natural-language documents to SHINE).
- Adversarial or intentionally tricky questions.
- Production-quality engineering (this is a research prototype).
- Full RAG-over-raw-code baseline (RAG-over-design-docs is in scope as a baseline; raw-code RAG is a stretch goal only).

## 4. Pilot Phase: Validating the Core Mechanism

Before any pipeline is built, we validate that SHINE preserves enough information from a natural-language document about code to answer questions about the repository.

### 4.1 Pilot setup

- Select three repositories from distinct domains whose READMEs already fit within SHINE's 1,150-token context limit without truncation.
- For each README, generate 15–20 question-answer pairs based on the README content (using Claude Code).
- Run three conditions per question:
  - **Naive:** Qwen3-8B with no context.
  - **In-context:** Qwen3-8B with the README in the prompt.
  - **SHINE:** Qwen3-8B with a LoRA generated from the README, no README in prompt.
- Score answers using the LLM-as-judge approach described in §12 (correctness rubric + failure-mode tag), with token-level F1 logged as a secondary metric. Manual inspection of all pilot answers given the small sample size.

### 4.2 Decision gate

The pilot must show, at minimum:
- SHINE condition substantially outperforms Naive (gap large enough to matter).
- SHINE condition tracks reasonably close to In-context (within a tolerance the team agrees on after seeing initial numbers; suggest noting this before running).

If SHINE tracks close to In-context, proceed to the full pipeline. If SHINE is much closer to Naive than to In-context, pause and diagnose before committing more effort. Possible diagnoses include: SHINE failing to preserve specific factual detail, Qwen3-8B base model failures unrelated to SHINE, or document/QA quality issues.

### 4.3 Pilot status (live)

The eval harness is implemented under `eval/` with the CLI at
`scripts/run_eval.py` (subcommands `predict`, `judge`, `report`, `all`),
plus a dedicated bake script at `scripts/generate_shine_lora.py`. Run
configs live in `config/eval/<run_name>.yaml`; one config per run.

Pilot artifacts now span both kilo and marimo on the easy tier under
oracle routing with conditions `naive | in_context | shine`. The
original kilo pilot run is `results/kilo_easy_v0_20260427T0037/` over
three docs (`overview_purpose_1`, `raw_terminal_input_1`,
`rows_editing_persistence_1`); the marimo pilot is at
`results/marimo_easy_v0_20260427T1929/`. The lean-prose A/B test of the
code-density hypothesis (§18) has completed and lives as paired
`*_detail_*` and `*_adapted_*` run dirs (e.g.,
`results/kilo_easy_v0_detail_20260427T1731/` and
`results/kilo_easy_v0_adapted_20260427T1747/`); see those reports for
the per-doc deltas. An embedding-routing run on kilo also exists for
the §10 routing comparison.

The pilot has surfaced two important findings that feed back into the
plan: doc code-density appears to drive LoRA quality (see §18), and
doc-derived QAs frequently leak into pretrained knowledge — they are
answerable without the document at all, which dilutes the naive/shine
gap (see §18). Both are reflected as risks rather than non-goals; the
project continues.

## 5. Topic Set Construction

Each repository gets its own topic set, built as follows:

**Universal seed (~3 topics).** A small fixed set of topics that apply to essentially any repository. Suggested seed: Setup/Installation, Overview/Purpose, Public API or Interface. The exact seed is finalized during the pilot.

**Structure-derived topics.** The agent inspects the README sections, top-level directory names, and any existing docs/ folder. Each candidate topic must be justified with a specific file or directory reference.

**Discovery-derived topics.** The agent may propose additional topics that are not directly evident from repository structure but are clearly relevant (e.g., "networking" in a repo where networking is woven through many files but not isolated to one directory). Each such topic must be justified with concrete evidence — at minimum, a list of files or code regions that the topic covers.

The topic set for each repo is recorded as a structured artifact before any documents are written, so we can audit topic discovery separately from document quality.

## 6. Document Generation

For each (repository, topic) pair, generate N documents at each difficulty level. Generation is deterministically capped at the 1,150-token SHINE context limit; documents that would exceed the cap are continued into a second (or further) document under the same topic. N is set to 2 to start; the right value will be revisited based on early results.

Difficulty levels:
- **Easy:** High-level information only.
- **Medium:** High-level plus important specifics, while remaining prose-like.
- **Hard:** Comprehensive detail.

We begin by generating only easy documents and adding medium and hard once the easy pipeline is working end-to-end.

Document generation is performed twice per repo:
- Once by Claude Code or Codex (primary).
- Once by Qwen3-8B itself (for the local-only generation comparison).

Both versions are evaluated separately.

## 7. QA Generation

Three sets of questions are generated per repository, with strict separation between them:

**Doc-derived QAs (for evaluation).** For each generated document, produce question-answer pairs whose answers are explicitly grounded in that document. These test the easier setting where the system has documents written specifically to cover the test questions.

**Blind QAs (for evaluation).** Generated by Claude Code (or Codex) given access to the codebase but not to the design documents. These represent realistic developer questions and test whether the topic decomposition + LoRA generation actually covers what someone would want to know. Blind QAs may naturally span multiple topics; this is expected and not adversarial.

**Ledger example questions (for routing, not evaluation).** A small set (3–5 per LoRA) generated from the same document by the same generator as the doc-derived QAs, but with an explicit instruction to produce questions *different from* the doc-derived QA set. These appear in the ledger to help the router match user questions to LoRAs. Keeping them disjoint from the eval QAs prevents the router from getting a free look at the test set.

All three sets are stored with metadata indicating their origin, the topic(s) they relate to (where known), the difficulty level (for doc-derived), and an explicit flag indicating which set they belong to. The ledger example questions are part of the LoRA store and are surfaced into the ledger; the doc-derived and blind QAs are eval-only and never appear in the ledger.

## 8. LoRA Generation, Ledger, and LoRA Store

For each design document, generate a LoRA via the SHINE hypernetwork in a single forward pass. Each LoRA is registered in two artifacts: a ledger (model-facing) and a LoRA store (system-facing).

### 8.1 Ledger (model-facing artifact)

> **Status (v0):** the markdown ledger described below is a Qwen-as-router
> prerequisite and is **not yet implemented**. The current artifact set
> ships only a structured `artifacts/<repo>/ledger.json` (described in
> §8.2 — closer to the LoRA store than to a model-facing ledger). The
> markdown ledger will be generated programmatically from the JSON store
> when Qwen-as-router work begins.

The ledger is a flat markdown file. Topics are second-level headings; each LoRA appears as a bullet under the topic it belongs to. The ledger is read directly by the local model during Qwen-as-router runs and is intended to be human-readable for debugging.

Each bullet contains:
- A human-readable **ID** following the scheme `{topic_slug}_{n}` (e.g., `setup_1`, `setup_2`, `auth_1`). Integers handle the case where multiple LoRAs cover the same topic.
- A short **title**.
- A **description** of what the LoRA covers.
- A small set of **example questions** the LoRA can answer. These are generated separately from the doc-derived eval QAs (same document, same generator, but explicitly different questions) to avoid contaminating routing evaluation. Roughly 3–5 example questions per LoRA.

Example structure:

```markdown
## Setup
- [setup_1] Installing as a Python package — covers pip install, dependency resolution, supported Python versions. Example questions: ...
- [setup_2] Development environment — covers cloning, virtualenv setup, pre-commit hooks. Example questions: ...

## Authentication
- [auth_1] OAuth flow overview — covers token exchange, refresh logic, scopes. Example questions: ...
```

We maintain one artifact ledger per repo. Each document entry records its own difficulty, so the same repo-level ledger can index easy, medium, and hard artifacts without duplicating topic sections across separate files.

The ledger does not contain: source document text, LoRA weights, file paths, generation timestamps, or embeddings. None of these help the model route, and including them would consume context tokens and risk distracting the router.

### 8.2 LoRA Store (system-facing artifact)

The LoRA store is a registry (JSON or SQLite) keyed by LoRA ID. It holds everything the ledger deliberately omits:

- LoRA ID (matches the ledger).
- Path to the LoRA weights file on disk.
- Difficulty level.
- Source document text.
- The doc-derived QA pairs associated with this LoRA (for evaluation, not for the ledger).
- The example questions exposed in the ledger.
- Embedding vector for embedding-based retrieval (or a path to it).
- Generation metadata: timestamp, doc-generator used (CC/Codex/Qwen), prompt version, code version.

The store is the single source of truth for any system component that needs to look up information about a LoRA (e.g., the eval harness loading weights by ID, the embedding router doing similarity search). The ledger is a view onto the store, generated programmatically.

## 9. Embedding Index

For each LoRA in the LoRA store, compute and store an embedding using a local sentence embedder (suggested: bge-small or e5-small — chosen for being small, local, and well-benchmarked on retrieval).

The embedded text (built by `scripts/embedding_router.py::make_lora_routing_text`) is the concatenation of: LoRA ID, title, description, keywords, the **example questions** exposed in the ledger, and the underlying document text itself. Note: the *example questions* (the routing-only set, kept disjoint from the eval QAs per §7) are deliberately used here rather than the doc-derived eval QAs — feeding doc-derived eval QAs into the embedding would let the router peek at the test set. This richer string is used only for retrieval and is held in the LoRA store; the ledger remains lean.

## 10. Routing Strategies

Three routing strategies are evaluated:

**Ground-truth oracle.** For each test question, a human (or an automatic mapping where unambiguous) provides the correct LoRA. This isolates SHINE-quality from routing-quality.

**Qwen-as-router.** Qwen3-8B is given the ledger markdown directly and asked to return the top-k most relevant LoRA IDs (with confidence). The ledger's ID scheme is designed to be Qwen-readable. *Status: not yet implemented; depends on the markdown ledger described in §8.1.*

**Embedding-based retrieval.** The question is embedded; cosine similarity to LoRA embeddings (held in the LoRA store) determines top-k. The ledger is not consulted in this path — IDs are looked up directly in the store. *Status: implemented (`eval/routing.py::EmbeddingRouter`, fed by `scripts/embedding_router.py`).*

For each strategy, k is varied (k = 1, 2, 3, possibly more). When k > 1, the retrieved LoRAs are averaged before loading into Qwen3-8B via rank-concatenation in `eval/composition.py` (rank_average, scale=1.0); k=1 short-circuits to identity for bit-for-bit reproducibility against pre-composition runs.

## 11. Baselines

- **Naive Qwen3-8B:** No retrieval, no context. Establishes a floor.
- **In-context Qwen3-8B:** Full ground-truth document in the prompt. Establishes a per-document ceiling for what a SHINE LoRA could do.
- **RAG-over-design-docs:** Retrieve top-k design documents using the same embedding model, stuff into Qwen3-8B's context. This is the most important non-SHINE baseline because it shares the privacy story. *Status: required, not nice-to-have. Not yet implemented in the eval harness — `VALID_CONDITIONS` currently is `("naive", "in_context", "shine")`; a `rag` condition still needs to be added. The project's headline claim depends on understanding this comparison.*
- **RAG-over-raw-code:** Stretch goal only.

## 12. Evaluation

**Scoring.** Answer correctness is judged by an LLM-as-judge using a stronger model than Qwen3-8B (v0: OpenAI `gpt-5.1`). The judge receives the question, the ground-truth answer, the model's answer, and — under the current default rubric (v2) — the source design document so it can verify whether the model's claims are grounded in this repo or generic plausibility. It emits structured JSON conforming to `schemas/judge_result.schema.json`:
- An integer **score** on a 1–5 scale; current default rubric is `prompts/llm_judge_grading_v2.md` (`rubric_version: v2`). v0 and v1 prompts are retained on disk for backfill / comparison runs and remain fully supported.
- A free-text **reasoning** field, encouraged to elaborate.
- A list of **failure_modes** drawn from the taxonomy below (required when score < 5; empty when score == 5).
- An optional **failure_mode_notes** for `other`.

The judge prompt explicitly instructs leniency on phrasing / equivalent paraphrases and strictness on specifics (wrong identifiers, values, names). v2 adds two further calibrations: an explicit *length-neutrality* clause (a one-sentence answer that names the right specific is a 5) and *doc-grounding* (a specific that does not trace to the design document is treated as fabrication). The harness post-processes: clears tags on score==5; auto-tags `other` if score<5 returned no tags.

LLM-as-judge has known failure modes (favoring verbose answers, leniency on near-misses, run-to-run drift). v0 mitigation:
- Logging the judge's full output for spot-checking (`judgments.jsonl` row carries `judge.reasoning` and `judge.raw_response`).
- Aggregation report (`report.md`) and per-doc heatmap plots for outlier inspection.
- **Not implemented (v0):** hand-graded calibration set, token-level F1 secondary metric. Hand-grading was explicitly deferred — we rely on the judge with leniency baked into the prompt. F1 may be added later if needed for direct comparability to the SHINE paper.

**Per-question logging.** Every evaluation run logs, per question:
- Question, ground-truth answer, model answer.
- Routing method and which LoRA(s) were retrieved (with scores).
- LoRA composition method (single / averaged top-k).
- Judge correctness score, failure-mode tag, and justification.
- F1 and exact-match scores.
- Wall-clock time and any errors.

**Failure mode taxonomy (taxonomy_version `v1`).** Multi-label, applied
when score is in {1, 2, 3, 4}; cleared when score == 5. The v0 taxonomy
shipped 5 tags; v1 added `no_specifics` to distinguish *absence of
substance* (vapor / restated question) from *wrong substance*
(`wrong_specifics`). v2 of the rubric did not change the tag set, only
the wording of the `no_specifics` definition (it is now reserved for
genuine absence of substance, not merely brief answers).

- `wrong_specifics` — Right concept or topic but specific names, values,
  or details are wrong. Subsumes the originally-planned `hallucinated`
  tag — fabricated specifics, including specifics that do not trace to
  the design document under v2, land here.
- `no_specifics` — The answer is conceptually framed but commits to no
  ground-truth-aligned facts; *substance is absent*, not merely word
  count low.
- `missing_information` — Key required facts omitted while at least one
  ground-truth-style fact is present.
- `off_topic` — Answer addresses a different question.
- `refusal_or_nonresponse` — Model declined, said it didn't know, or
  returned empty.
- `format_failure` — Substance roughly there but unparseable, truncated,
  or in the wrong format.
- `other` — Use only when no tag above fits; populate `failure_mode_notes`.

The originally-planned `retrieval miss`, `composition failure`, and
`LoRA preservation failure` tags were dropped from the per-answer
taxonomy: those describe *which condition / which routing produced this
answer*, not *what shape of error this answer has*. They belong in run
metadata (already captured: `condition`, `routing.strategy`,
`routing.selected_lora_ids`) and are derivable post-hoc by joining tags
to those fields. The originally-planned "genuinely unanswerable" tag was
dropped because the eval harness does not surface unanswerable
questions to the judge — the QA generator is responsible for filtering
those out before they reach the eval.

## 13. Experimental Matrix

For the main evaluation, we sweep:
- 5 repositories × 3 difficulty levels × N documents per (repo, topic, difficulty) × 2 doc-generators (CC/Codex vs Qwen) × 2 QA sets (doc-derived vs blind) × 3 routing strategies × multiple k values × baselines.

The repo lineup (under `target_repos/`) is `antirez__kilo`,
`fogleman__Craft`, `marimo-team__marimo`, `psf__requests`, and
`pytest-dev__pytest`, chosen to span a ~150× size range and to give a
Python-only sub-series (requests → pytest → marimo) for a within-language
size scan. *Status:* full doc/QA/LoRA artifacts currently exist for
`antirez__kilo` and `marimo-team__marimo`; the remaining three are
scraped as submodules but have not yet been put through the
doc-generation / QA / LoRA-baking pipeline.

This is a large matrix. We start with easy-difficulty documents only on 2–3 repos to catch bugs and validate the pipeline end-to-end, then expand to medium and hard difficulties and additional repositories. Some cells (e.g., baselines × difficulty levels) collapse where the variable does not apply.

## 14. Engineering Infrastructure

The following must be built early and shared across all workstreams:

- **Evaluation harness.** Implemented as `eval/` (Python package) and
  `scripts/run_eval.py` (CLI). Run config is a self-contained YAML at
  `config/eval/<run_name>.yaml` listing artifacts to evaluate, model
  paths, conditions, routing, and judge configuration. Subcommands:
  `predict` (Qwen + pre-baked LoRA → `predictions.jsonl`),
  `judge` (OpenAI `gpt-5.1` → `judgments.jsonl`),
  `report` (per-condition aggregation → `report.md` + auto-rendered
  plots), `all` (predict → judge → report). Every invocation writes a
  fresh non-resumable `results/<run_name>_<YYYYMMDDTHHMM>/` directory.
- **Ledger and LoRA store schemas.** v0 ships JSON schemas under
  `schemas/`: `ledger.schema.json`, `eval_result.schema.json`,
  `judge_result.schema.json`, `qa_pairs.schema.json`,
  `design_doc.schema.json`. Markdown ledger (model-facing, for
  Qwen-as-router) is deferred until that workstream begins; see §8.1
  status note.
- **Versioning.** Every generated artifact records `generator`,
  `prompt_version`, source `commit`, and `repo_id`. Eval runs snapshot
  their resolved config to `<run-dir>/run_config.yaml` and a
  `manifest.json` carrying timestamp, git SHA, model paths, and the
  per-doc input record.
- **Storage layout.** Actual layout is
  `artifacts/{repo}/{difficulty}/{docs|qas|loras}/<doc_id>.{json|pt}`.
  The originally-suggested `{generator}` and `{topic}` directory levels
  were dropped — `generator` is recorded inside each artifact's JSON,
  and topic is encoded in the `document_id` (`{topic_slug}_{n}`).
- **Reproducibility.** A single judged row is fully reproducible from
  `<run-dir>/predictions.jsonl` (input + raw model output) plus
  `<run-dir>/run_config.yaml` (model paths, seeds, prompt path) and the
  pre-baked LoRA `.pt` referenced in the row's `lora_path`.

## 15. Workstream Decomposition

The work falls into several roughly-independent streams:

**Stream A: Pilot and SHINE integration.**
- Get SHINE running locally.
- Run the pilot experiments on three READMEs.
- Build the basic LoRA-generation and LoRA-loading utilities.
- Produce the pilot report and decision-gate recommendation.

**Stream B: Document and QA generation pipeline.**
- Design and test the agent prompts for topic discovery, document generation, and QA generation.
- Build the pipeline that takes a repository and produces documents, QAs, ledger files, and LoRA store entries.
- Compare CC/Codex doc generation against Qwen-generated docs.

**Stream C: Retrieval, evaluation, and baselines.**
- Build the embedding index and embedding-based router.
- Build the Qwen-as-router prompt and evaluation.
- Build the RAG-over-design-docs baseline.
- Build the evaluation harness and failure-mode logging.

These streams have dependencies (B and C need outputs from A; C needs outputs from B) but each can be substantially developed in parallel once interfaces are agreed.

## 16. Sequencing

1. Set up shared infrastructure (storage layout, ledger schema, evaluation harness skeleton).
2. Run pilot on three small READMEs. Hit the decision gate.
3. If pilot succeeds: lock pipeline decisions (document length policy, topic-discovery prompt, QA generation prompt, embedding model).
4. Source ~10 small-to-medium repositories across distinct domains.
5. Run document and QA generation pipeline (CC/Codex first, then Qwen) on all repos.
6. Generate all LoRAs and embeddings.
7. Run full evaluation matrix, starting with a 2–3 repo sub-matrix to catch bugs.
8. Analyze, write up, present.

## 17. Open Questions to Resolve Early

- ~~What is the correctness rubric the LLM judge should use (binary,
  three-point, finer)?~~ **Resolved:** integer 1–5 (`gpt-5.1`). Current
  defaults are `rubric_version: v2`, `taxonomy_version: v1`; rubric at
  `prompts/llm_judge_grading_v2.md`. v0/v1 rubric files are retained on
  disk for backfill comparisons.
- ~~How do we score answers when the ground-truth answer is a list,
  code reference, or numeric value?~~ **Resolved (operational):** judge
  prompt instructs leniency on phrasing, strictness on specifics. Pilot
  spot-checks indicate this works for prose-style answers and is
  reasonable for short code-token answers. The v1→v2 nuance — judge
  being lenient when a vague-but-plausible answer overlaps a loosely
  worded ground truth — is partially mitigated by v2's doc-grounding
  clause: claims that do not trace to the design document are now
  treated as fabrication.
- What is the tolerance for the pilot decision gate (how close to
  In-context is "close enough")? **Partially answered by pilot:** on
  the unfiltered kilo `easy` set, in-context is 4.93 and shine is 2.40
  (≈ naive 2.52). On the doc-specific subset (questions where naive
  scored ≤ 2), shine 2.19 vs naive 1.65 — shine wins on the questions
  that actually require the document. A formal threshold still needs
  picking before the main eval.
- Final composition of the universal topic seed.
- How do we handle blind QAs that have no clean ground-truth LoRA
  mapping (relevant for evaluating routing accuracy)?
- What is the right value of N (documents per (repo, topic, difficulty)
  tuple)?

## 18. Risks and Mitigations

- **SHINE compression loses critical detail.** Pilot detects this.
  Mitigation: split topics into more, smaller LoRAs.
- **Code-density of docs degrades SHINE compression. (Pilot finding.)**
  Docs that catalog identifiers, flag names, or escape codes produce
  visibly worse SHINE outputs (gibberish loops, invented identifiers)
  than prose-first docs at the same conceptual content, even though all
  three pilot docs fit comfortably under SHINE's 1150-token cap. The
  in-context ceiling stays high for code-dense docs (so the *document*
  contains the info), but the *LoRA* doesn't preserve it. Mitigation:
  write design docs in prose-first style with backtick density at or
  below the `overview_purpose_1` reference (~2.5%); avoid enumeration
  of constants, flags, or struct fields; describe purpose alongside
  the named entity rather than listing entities. The lean-prose A/B
  re-bake of two kilo docs has completed; paired runs live at
  `results/kilo_easy_v0_detail_*/` and `results/kilo_easy_v0_adapted_*/`
  (and the matching marimo pair under `results/marimo_easy_v0_*`).
  Per-doc deltas are in those reports.
- **QA quality leaks into pretrained knowledge. (Pilot finding.)**
  Doc-derived QAs frequently ask questions that Qwen3-8B can answer
  from pretrained knowledge alone — standard ANSI escape codes,
  self-documenting function names like `editorInsertNewline`, generic
  editor conventions. On the kilo pilot, 21% of naive answers scored
  ≥ 4 without the doc. This dilutes the naive vs shine gap and inflates
  the apparent strength of the naive baseline. Mitigation: (a) filter
  the QA set post-hoc to questions naive scored ≤ N
  (`scripts/plot_naive_hard.py` does this for analysis); (b) generate
  QAs that test conceptual understanding which depends on the specific
  document, not on general LLM knowledge — verbatim identifier-recall
  questions are dangerous because their answers are often inferable
  from the identifier name itself.
- **RAG-over-docs beats SHINE.** Possible. Mitigation: identify
  dimensions where SHINE wins (latency at QA time, ability to compose,
  etc.) and highlight those even if F1 is comparable.
- **Qwen-generated docs are much worse than CC-generated docs.** This
  would weaken the privacy story. Mitigation: explicitly evaluate;
  report honestly even if the gap is large.
- **LoRA composition does not work.** Likely partially. Mitigation:
  fall back to top-1 routing as the default, treat composition as a
  research finding rather than a system requirement.
- **Topic discovery is unreliable across repos.** Mitigation:
  structure-derived topics provide a floor; discovered topics are
  additive and can be evaluated separately.
- **Scope creep.** Mitigation: the non-goals section is firm. Anything
  outside it requires explicit team agreement.

## 19. Additional Thoughts (post-v1 ideas)

Ideas that are likely valuable but should not gate the first working version of the system. Worth revisiting once the v1 pipeline is end-to-end functional.

**Tree-structured ledger for scaling.** The v1 ledger is a flat markdown file with topic sections — appropriate for small-to-medium repos with a modest number of LoRAs. As repos grow (or as we move toward larger codebases), a flat ledger becomes a long context window for Qwen-as-router and risks routing degradation. A tree-structured ledger would address this: top-level sections for high-level topic categories (e.g., "high-level info"), with nested sub-ledgers underneath (e.g., "setup" inside "high-level info", with individual LoRAs at the leaves). Routing then becomes hierarchical: the router first picks a category, descends into its sub-ledger, and selects a leaf LoRA. This bounds the number of options the router sees at any one decision point and naturally supports much larger LoRA libraries.

The tree design is genuinely useful and worth implementing, but is additive: the flat ledger is the right starting point because it's simpler to build, simpler to debug, and sufficient for the scale of this project's main evaluation. Add the tree variant once the flat version is solid and we have evidence (from real routing accuracy numbers) that flatness is becoming a bottleneck.
