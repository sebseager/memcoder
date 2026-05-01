# Filtered Eval Report — naive score ≤ 2 — all_repos_easy_oracle_20260430T2345

- Source: `all_repos_easy_oracle_20260430T2345/judgments.jsonl`
- Kept QAs: **99 / 121** (81.8%)

## Score summary on filtered set

| Condition | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|---|
| `in_context` | 112 | 4.90 | 92.9% | 0 | 1 | 1 | 6 | 104 |
| `naive` | 112 | 1.91 | 1.8% | 20 | 87 | 2 | 1 | 2 |
| `shine` | 112 | 2.16 | 8.0% | 34 | 50 | 13 | 6 | 9 |

Filter rule: keep only questions where the **naive** condition judge score is ≤ 2. This isolates questions that require document-specific knowledge to answer well.
