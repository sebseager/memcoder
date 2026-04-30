# AI Tool Use

AI tools were used during MemCoder development. The project intentionally keeps
that assistance visible because generated code, prompts, synthetic evaluation
artifacts, and documentation can affect reproducibility.

## Tools Used

- Claude Code or Codex-style coding agents were part of the planned artifact
  generation workflow for topic discovery, design documents, and doc-derived QA
  pairs. Generated artifacts record prompt provenance where the schema supports
  it.
- OpenAI `gpt-5.1` is used as the configured LLM-as-judge in
  `config/eval/*.yaml`.
- Local Qwen3-8B is the evaluated base model for `naive`, `in_context`, and
  `shine` prediction conditions.
- Agent workflows using frontier coding models were used to speed up debugging 
  throughout the development process.

## Configuration Notes

- Judge configuration is explicit in each `config/eval/*.yaml` file:
  `judge.provider`, `judge.model`, `judge.prompt`, `rubric_version`,
  `taxonomy_version`, retry count, and concurrency.
- Generation prompts used for artifacts live under `prompts/` and are versioned
  by filename and/or prompt-version fields in generated JSON.
