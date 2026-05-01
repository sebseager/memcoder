# Filtered Eval Report — naive score ≤ 2 — marimo_easy_v0_20260427T1929

- Source: `marimo_easy_v0_20260427T1929/judgments.jsonl`
- Kept QAs: **20 / 24** (83.3%)

## Score summary on filtered set

| Condition | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| `in_context` | 20 | 4.70 | 80.0% | 0 | 1 | 0 | 3 | 16 |
| `naive` | 20 | 1.90 | 0.0% | 2 | 18 | 0 | 0 | 0 |
| `shine` | 20 | 2.45 | 15.0% | 5 | 8 | 3 | 1 | 3 |

Filter rule: keep only questions where the **naive** condition judge score is ≤ 2. This isolates questions that require document-specific knowledge to answer well.
