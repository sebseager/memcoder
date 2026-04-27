# Prompt Versioning

Prompt versions use the convention `{prompt_name}_v{major}`.

Increment the major version whenever a template change can materially change
generated artifact content or schema shape. Every generated JSON artifact must
include the exact `prompt_version` string from the prompt template, populated by
the generator or manual run.

The current version of each prompt lives in this directory under its canonical
filename; older versions are recovered from git history using the
`prompt_version` string in the artifact as a key.

# Prompts

This directory stores versioned prompt templates used to generate MemCoder
artifacts from target repositories.

Templates:

- `topic_discovery.md` (**v1**): discover and justify repository topics as a
  first-class artifact with evidence-pack hints for subagents.
- `orchestrator_artifact_generation.md` (v0): coordinate the local
  orchestrator/subagent artifact-generation workflow.
- `design_doc_generation.md` (**v1**): generate one prose design document for a
  topic plus a short routing/retrieval description.
- `doc_derived_qa_generation.md` (**v1**): generate exactly three evaluation QA
  pairs from a design document.
- `ledger_example_question_generation.md` (**v1**): generate a separate routing
  example question artifact kept distinct from evaluation QAs.
- `llm_judge_grading_v0.md`: rubric used by the eval harness's LLM judge phase. Version is in the filename per a different convention because the harness loads it by literal path.

Generated artifacts should record the prompt filename and version used to create
them.

## Pilot-driven changes (v0 → v1)

The kilo `easy` pilot (see `memcoder_plan.md` §4.3) surfaced two findings
that prompted the v1 bumps to the doc and QA prompts:

- **Code-density of a doc strongly drives SHINE compression quality.**
  Code-dense docs (high backtick density, enumerated flag/constant/struct
  catalogs) produced LoRAs that visibly degraded the model's output.
  Lean-prose rewrites of the same content gave +0.31 to +0.70 mean-score
  improvements on the affected docs. `design_doc_generation_v1` adds a
  binding style section (prose-first, ≤ 2.5% backtick density target,
  no enumeration of flags/constants/struct fields).
- **Doc-derived QAs leak into pretrained knowledge.** ~21% of pilot QAs
  were answerable by base Qwen3-8B without any document or LoRA, because
  they tested ANSI/POSIX standards, self-documenting function names, or
  generic editor conventions. This dilutes the naive vs SHINE gap.
  `doc_derived_qa_generation_v1` and `ledger_example_question_generation_v1`
  add a self-check step asking whether a generic LLM could answer the
  question without the doc, with examples of weak/leaky question shapes
  to avoid.

`topic_discovery.md` is now v1 for artifact-schema reasons, not because the
pilot surfaced topic-discovery failures: topic discovery is a first-class
artifact and includes `evidence_pack_hint` values for local subagents.
