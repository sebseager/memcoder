# Prompt: Topic Discovery

Version: `topic_discovery_v0`

## Purpose

Discover a small set of documentation topics for a target repository. Each topic
must be justified with concrete repository evidence.

## Inputs

- Repository ID: `{repo_id}`
- Repository path: `{repo_path}`
- Source commit: `{commit}`
- Generator: `{generator}`
- Files or summaries provided by the caller: `{repo_context}`

## Instructions

You are generating topic candidates for a local code QA experiment.

Inspect the provided repository context and propose topics that would be useful
for answering developer questions about this repository. Include universal topics
where appropriate, such as setup, overview/purpose, and public interface.

For each topic, provide:

- `topic_slug`: a short lowercase identifier using underscores.
- `title`: a short human-readable title.
- `description`: what this topic should cover.
- `evidence`: specific files, directories, or code regions that justify the topic.
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
  "prompt_version": "topic_discovery_v0",
  "topics": [
    {
      "topic_slug": "overview_purpose",
      "title": "Overview / Purpose",
      "description": "...",
      "evidence": ["README.md", "kilo.c"],
      "priority": "initial"
    }
  ]
}
```
