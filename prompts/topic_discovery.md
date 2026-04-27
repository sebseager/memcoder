# Prompt: Topic Discovery

Version: `topic_discovery_v1`

## Purpose

Discover a small set of documentation topics for a target repository and write a
first-class topic artifact. Each topic must be justified with concrete repository
evidence.

## Inputs

- Repository ID: `{repo_id}`
- Repository path: `{repo_path}`
- Source commit: `{commit}`
- Generator: `{generator}`
- Universal seed topics: `{seed_topics}`
- Files, tree excerpts, or summaries provided by the orchestrator: `{repo_context}`

## Instructions

You are the topic-discovery step in a local orchestrator/subagent workflow for a
code QA experiment. Assume the repository contents stay local. Work only from the
repository context provided by the orchestrator.

Inspect the provided repository context and propose topics that would be useful
for answering developer questions about this repository. Include universal topics
where appropriate, such as setup, overview/purpose, and public interface. Prefer
topics that can be handed to independent subagents with bounded, topic-specific
evidence packs.

For each topic, provide:

- `topic_slug`: a short lowercase identifier using underscores.
- `title`: a short human-readable title.
- `description`: what this topic should cover.
- `evidence`: specific files, directories, or code regions that justify the topic.
- `evidence_pack_hint`: the files, directories, docs pages, or search targets an
  orchestrator should include for the document-generation subagent.
- `priority`: `initial`, `high`, `medium`, or `low`.

Do not invent behavior that is not supported by the repository evidence. Prefer
fewer, better topics over a long list of overlapping topics.

## Output Format

Return JSON:

```json
{
  "repo_id": "{repo_id}",
  "commit": "{commit}",
  "generator": "{generator}",
  "prompt_version": "topic_discovery_v1",
  "seed_topics": ["overview_purpose", "setup_installation", "public_interface"],
  "topics": [
    {
      "topic_slug": "overview_purpose",
      "title": "Overview / Purpose",
      "description": "...",
      "evidence": ["README.md", "kilo.c"],
      "evidence_pack_hint": ["README.md", "top-level source tree", "kilo.c"],
      "priority": "initial"
    }
  ]
}
```
