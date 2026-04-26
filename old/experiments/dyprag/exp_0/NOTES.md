# Exp 0 — Dataset Characterization: Notes

**Date:** 2026-04-19
**Status:** Complete

## Objective

Tokenize ground-truth relevant files for every SWE-Bench Lite instance using the Qwen3-8B tokenizer, then partition instances into context-constrained vs. unconstrained subsets. Also verify that the full model stack (base model + LoRA + encoder) fits on the available GPU.

## Scripts

- `characterize.py` — Main dataset characterization script. Fetches relevant source files from GitHub at the base commit, tokenizes with `Qwen/Qwen3-8B`, sweeps budget thresholds, selects the best one in the 80–120 constrained-instance range, and produces charts + JSON output.
- `hw_profile.py` — Hardware profiling script. Loads Qwen3-8B in 4-bit, adds a rank-16 LoRA adapter, loads the sentence-transformer encoder, and measures GPU memory usage at each step.

## Key Findings

### Dataset structure

- **300 instances** across **12 repositories** in SWE-Bench Lite (test split).
- Every instance in the test split touches exactly **1 file** in its patch. No multi-file patches exist in this split. This simplifies Phase 0 (no composition needed for oracle LoRAs) but means Phase 2's multi-file composition experiments will need instances from the full SWE-Bench training split or will need to redefine "relevant files" more broadly than "files touched by the patch."

### Token count distribution

| Statistic | Tokens |
|-----------|--------|
| Min       | 164    |
| P25       | 2,772  |
| Median    | 6,338  |
| Mean      | 9,957  |
| P75       | 14,982 |
| P90       | 22,537 |
| P95       | 28,423 |
| Max       | 74,448 |

Heavy right skew — the median is ~6.3K tokens but the mean is ~10K. A few very large files (matplotlib, sympy) pull the tail out to 74K tokens.

### Budget threshold selection

The README specifies a default budget of 4,096 tokens with the constraint that the "context-constrained" subset (instances where relevant files exceed the budget) should contain 80–120 instances.

At 4,096 tokens: **197 constrained** — far too many (two-thirds of the dataset).

A fine-grained sweep found that thresholds from 9,500 to 14,000 all fall in the 80–120 range:

| Budget  | Constrained | Unconstrained |
|---------|-------------|---------------|
| 4,096   | 197         | 103           |
| 8,192   | 129         | 171           |
| 9,500   | 118         | 182           |
| **11,500** | **103**  | **197**       |
| 14,000  | 83          | 217           |

**Selected budget: 11,500 tokens → 103 constrained instances.**

This is near the center of the target range and is a reasonable "context window budget" — it represents roughly one-third of the Qwen3-8B native context length (32,768 tokens) allocated to file content, leaving headroom for the system prompt, problem statement, and generation.

### Per-repo breakdown (at 11,500 budget)

| Repository | Instances | Median tokens | Constrained |
|---|---|---|---|
| django/django | 114 | 4,227 | 33 |
| sympy/sympy | 77 | 10,906 | 38 |
| matplotlib/matplotlib | 23 | 14,338 | 17 |
| scikit-learn/scikit-learn | 23 | 4,526 | 3 |
| pytest-dev/pytest | 17 | 4,730 | 2 |
| sphinx-doc/sphinx | 16 | 6,797 | 4 |
| astropy/astropy | 6 | 5,671 | 2 |
| psf/requests | 6 | 5,234 | 0 |
| pylint-dev/pylint | 6 | 1,544 | 0 |
| pydata/xarray | 5 | 7,950 | 2 |
| mwaskom/seaborn | 4 | 10,115 | 2 |
| pallets/flask | 3 | 4,421 | 0 |

The constrained subset is dominated by **sympy** (38 instances, 37%) and **django** (33 instances, 32%), with **matplotlib** contributing another 17 (16%). Three repos (flask, requests, pylint) contribute zero constrained instances — their files are uniformly small.

### Hardware profiling

GPU: **NVIDIA GeForce RTX 5070 Ti** (17,094 MB total)

| Component | GPU memory |
|---|---|
| Qwen3-8B (4-bit NF4) | 6,411 MB |
| LoRA adapter (rank 16, q/v/up/down_proj) | +106 MB |
| all-MiniLM-L6-v2 encoder | +91 MB |
| **Total after all loads** | **6,622 MB** |
| **Free** | **10,473 MB** |

**Verdict: FITS.** Over 10 GB free after loading all three components. Plenty of headroom for training activations, optimizer state, and batch data. The encoder can remain on GPU for all phases — no need to fall back to CPU-only encoder inference.

LoRA trainable parameters: 26,542,080 (0.32% of total 8.2B parameters).

## Output files

All in `exp_0/results/`:

- `token_counts.json` — Per-instance token counts and file details
- `subsets.json` — Constrained/unconstrained instance ID lists with selected budget
- `summary.json` — Aggregate statistics and per-repo breakdown
- `hw_profile.json` — GPU memory measurements
- `token_distribution.png` — Histogram + CDF of token counts
- `threshold_sweep.png` — Histogram with multiple budget thresholds overlaid
- `per_repo_boxplot.png` — Box plot of token counts by repository

## Implications for Phase 0

1. **Budget threshold raised from 4,096 to 11,500.** The README's default of 4,096 was too aggressive — it classifies 66% of instances as constrained, which would make the "unconstrained" baseline too small. At 11,500, the split is 103/197 (34%/66%), giving a large enough constrained subset for statistical power while keeping the unconstrained baseline meaningful.
2. **Single-file patches only.** Oracle LoRA training in Phase 0 can use one LoRA per instance (no file-level vs. multi-file distinction needed for Lite). The averaging-across-files approach mentioned for Phase 0 Exp 1 condition D is not applicable here — each instance has exactly one relevant file.
3. **Hardware is not a bottleneck.** All three components fit comfortably with 10+ GB free. No architectural compromises needed.
4. **Repo bias in constrained subset.** sympy and django together account for 69% of the constrained subset. Phase 0 results should be reported both overall and per-repo to check whether oracle LoRA effectiveness varies by codebase style.
