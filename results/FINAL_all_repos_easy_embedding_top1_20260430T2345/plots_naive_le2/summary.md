# Filtered Eval Report — naive score ≤ 2 — all_repos_easy_embedding_top1_20260430T2345

- Source: `all_repos_easy_embedding_top1_20260430T2345/judgments.jsonl`
- Kept QAs: **97 / 121** (80.2%)

## Score summary on filtered set

| Condition | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| `in_context` | 110 | 4.95 | 95.5% | 0 | 0 | 1 | 4 | 105 |
| `naive` | 110 | 1.88 | 1.8% | 21 | 86 | 0 | 1 | 2 |
| `shine` | 110 | 2.05 | 6.4% | 34 | 54 | 12 | 3 | 7 |

Filter rule: keep only questions where the **naive** condition judge score is ≤ 2. This isolates questions that require document-specific knowledge to answer well.
