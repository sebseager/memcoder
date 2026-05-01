# Filtered Eval Report — naive score ≤ 2 — kilo_easy_v0_20260427T0138

- Source: `kilo_easy_v0_20260427T0138/judgments.jsonl`
- Kept QAs: **27 / 43** (62.8%)

## Score summary on filtered set

| Condition | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| `in_context` | 27 | 4.85 | 88.9% | 0 | 0 | 1 | 2 | 24 |
| `naive` | 27 | 1.67 | 0.0% | 9 | 18 | 0 | 0 | 0 |
| `shine` | 27 | 2.59 | 7.4% | 4 | 11 | 6 | 4 | 2 |

Filter rule: keep only questions where the **naive** condition judge score is ≤ 2. This isolates questions that require document-specific knowledge to answer well.
