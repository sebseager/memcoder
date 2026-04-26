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

We maintain one ledger per difficulty level (three ledgers per repo: easy, medium, hard). The router operates against a single ledger at a time; difficulty becomes a configuration of the run rather than something the router has to reason about. The topic sections are identical across ledgers — only the underlying documents (and therefore the LoRA contents) differ in depth.

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

The embedded text is the concatenation of: topic name, title, description, the doc-derived questions this LoRA is intended to answer, and the underlying document text itself. This richer string is used only for retrieval and is held in the LoRA store; the ledger remains lean.

## 10. Routing Strategies

Three routing strategies are evaluated:

**Ground-truth oracle.** For each test question, a human (or an automatic mapping where unambiguous) provides the correct LoRA. This isolates SHINE-quality from routing-quality.

**Qwen-as-router.** Qwen3-8B is given the ledger markdown directly and asked to return the top-k most relevant LoRA IDs (with confidence). The ledger's ID scheme is designed to be Qwen-readable.

**Embedding-based retrieval.** The question is embedded; cosine similarity to LoRA embeddings (held in the LoRA store) determines top-k. The ledger is not consulted in this path — IDs are looked up directly in the store.

For each strategy, k is varied (k = 1, 2, 3, possibly more). When k > 1, the retrieved LoRAs are averaged before loading into Qwen3-8B.

## 11. Baselines

- **Naive Qwen3-8B:** No retrieval, no context. Establishes a floor.
- **In-context Qwen3-8B:** Full ground-truth document in the prompt. Establishes a per-document ceiling for what a SHINE LoRA could do.
- **RAG-over-design-docs:** Retrieve top-k design documents using the same embedding model, stuff into Qwen3-8B's context. This is the most important non-SHINE baseline because it shares the privacy story. *Status: required, not nice-to-have. The project's headline claim depends on understanding this comparison.*
- **RAG-over-raw-code:** Stretch goal only.

## 12. Evaluation

**Scoring.** Answer correctness is judged by an LLM-as-judge using a stronger model than Qwen3-8B (e.g., Claude or a comparable frontier model). The judge receives the question, the ground-truth answer, and the model's answer, and outputs:
- A correctness score (rubric to be defined during the pilot — at minimum, correct / partially correct / incorrect).
- A failure-mode tag drawn from the taxonomy below (when the answer is not fully correct).
- A brief justification.

LLM-as-judge has known failure modes (favoring verbose answers, leniency on near-misses, run-to-run drift). We mitigate by:
- Hand-grading a sample of 30–50 questions to validate judge calibration before trusting it on the full eval.
- Logging the judge's full output for spot-checking.
- Reporting token-level F1 (matching the SHINE paper) as a secondary metric for sanity-checking and direct comparability to published results.

**Per-question logging.** Every evaluation run logs, per question:
- Question, ground-truth answer, model answer.
- Routing method and which LoRA(s) were retrieved (with scores).
- LoRA composition method (single / averaged top-k).
- Judge correctness score, failure-mode tag, and justification.
- F1 and exact-match scores.
- Wall-clock time and any errors.

**Failure mode taxonomy** (built and refined during the pilot, frozen before main eval; emitted by the LLM judge per question):
- Answer hallucinated (no relation to source).
- Right entity, wrong attribute.
- Right topic, wrong specific fact.
- Retrieval miss (correct LoRA not retrieved).
- Composition failure (multiple LoRAs averaged, result degraded).
- LoRA preservation failure (correct LoRA loaded but did not preserve relevant detail).
- Question genuinely unanswerable from any document.
- Other (with free-text note).

## 13. Experimental Matrix

For the main evaluation, we sweep:
- 10 repositories × 3 difficulty levels × N documents per (repo, topic, difficulty) × 2 doc-generators (CC/Codex vs Qwen) × 2 QA sets (doc-derived vs blind) × 3 routing strategies × multiple k values × baselines.

This is a large matrix. We start with easy-difficulty documents only on 2–3 repos to catch bugs and validate the pipeline end-to-end, then expand to medium and hard difficulties and additional repositories. Some cells (e.g., baselines × difficulty levels) collapse where the variable does not apply.

## 14. Engineering Infrastructure

The following must be built early and shared across all workstreams:

- **Evaluation harness.** A single function that takes (question, retrieval-method, model-config, k) and returns a logged result. All experiments call this.
- **Ledger and LoRA store schemas.** The ledger markdown format and the LoRA store schema (JSON or SQLite) are agreed before any artifacts are written.
- **Versioning.** Every generated artifact (document, LoRA, embedding) is timestamped and tied to a manifest recording the prompts and code version used to generate it.
- **Storage layout.** Decide upfront where artifacts live (suggested: `artifacts/{repo}/{generator}/{topic}/{difficulty}/...`) and document it.
- **Reproducibility.** It must be possible to reproduce a single result from logs without re-running the full pipeline.

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

- What is the tolerance for the pilot decision gate (how close to In-context is "close enough")?
- Final composition of the universal topic seed.
- How do we handle blind QAs that have no clean ground-truth LoRA mapping (relevant for evaluating routing accuracy)?
- What is the right value of N (documents per (repo, topic, difficulty) tuple)?
- How do we score answers when the ground-truth answer is a list, code reference, or numeric value (where the LLM judge or F1 may be misleading)?
- What is the correctness rubric the LLM judge should use (binary, three-point, finer)?

## 18. Risks and Mitigations

- **SHINE compression loses critical detail.** Pilot detects this. Mitigation: split topics into more, smaller LoRAs.
- **RAG-over-docs beats SHINE.** Possible. Mitigation: identify dimensions where SHINE wins (latency at QA time, ability to compose, etc.) and highlight those even if F1 is comparable.
- **Qwen-generated docs are much worse than CC-generated docs.** This would weaken the privacy story. Mitigation: explicitly evaluate; report honestly even if the gap is large.
- **LoRA composition does not work.** Likely partially. Mitigation: fall back to top-1 routing as the default, treat composition as a research finding rather than a system requirement.
- **Topic discovery is unreliable across repos.** Mitigation: structure-derived topics provide a floor; discovered topics are additive and can be evaluated separately.
- **Scope creep.** Mitigation: the non-goals section is firm. Anything outside it requires explicit team agreement.

## 19. Additional Thoughts (post-v1 ideas)

Ideas that are likely valuable but should not gate the first working version of the system. Worth revisiting once the v1 pipeline is end-to-end functional.

**Tree-structured ledger for scaling.** The v1 ledger is a flat markdown file with topic sections — appropriate for small-to-medium repos with a modest number of LoRAs. As repos grow (or as we move toward larger codebases), a flat ledger becomes a long context window for Qwen-as-router and risks routing degradation. A tree-structured ledger would address this: top-level sections for high-level topic categories (e.g., "high-level info"), with nested sub-ledgers underneath (e.g., "setup" inside "high-level info", with individual LoRAs at the leaves). Routing then becomes hierarchical: the router first picks a category, descends into its sub-ledger, and selects a leaf LoRA. This bounds the number of options the router sees at any one decision point and naturally supports much larger LoRA libraries.

The tree design is genuinely useful and worth implementing, but is additive: the flat ledger is the right starting point because it's simpler to build, simpler to debug, and sufficient for the scale of this project's main evaluation. Add the tree variant once the flat version is solid and we have evidence (from real routing accuracy numbers) that flatness is becoming a bottleneck.
