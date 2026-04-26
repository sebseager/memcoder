# Prompts

This directory stores versioned prompt templates used to generate MemCoder
artifacts from target repositories.

Initial templates:

- `topic_discovery.md`: discover and justify repository topics.
- `design_doc_generation.md`: generate one prose design document for a topic.
- `doc_derived_qa_generation.md`: generate evaluation QA pairs from a design document.
- `ledger_example_question_generation.md`: generate routing example questions that are
  kept separate from evaluation QAs.

Generated artifacts should record the prompt filename and version used to create
them.
