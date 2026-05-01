# Filtered Eval Report — naive score ≤ 2 — kilo_easy_v0_20260427T0037

- Source: `kilo_easy_v0_20260427T0037/judgments.jsonl`
- Kept QAs: **26 / 42** (61.9%)

## Score summary on filtered set

| Condition | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| `in_context` | 26 | 4.92 | 92.3% | 0 | 0 | 0 | 2 | 24 |
| `naive` | 26 | 1.65 | 0.0% | 9 | 17 | 0 | 0 | 0 |
| `shine` | 26 | 2.19 | 3.8% | 10 | 7 | 4 | 4 | 1 |

Filter rule: keep only questions where the **naive** condition judge score is ≤ 2. This isolates questions that require document-specific knowledge to answer well.
