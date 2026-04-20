# Experiment 1 — Oracle Ceiling

## Objective

Determine whether injecting source-file knowledge via LoRA fine-tuning can
recover the information lost when context is truncated to a fixed token budget.
The "oracle" LoRA is trained on the pre-patch source file for each SWE-Bench
Lite instance, giving the model perfect knowledge of the file to be edited.

## Conditions

| Cond | Context          | LoRA  | Role                        |
|------|------------------|-------|-----------------------------|
| A    | None             | No    | Floor — no information      |
| B    | Truncated (11.5k)| No    | RAG stand-in baseline       |
| C    | Full file        | No    | Ceiling — best case context |
| D    | Truncated (11.5k)| Oracle| Test — does LoRA help?      |

The gate criterion for proceeding with DyPRAG is:
**(D − B) / (C − B) > 0.5** — i.e., the oracle LoRA recovers at least half
the gap between truncated and full context.

## Setup

- **Model**: Qwen/Qwen3-8B, 4-bit NF4 quantization, bf16 compute dtype
- **LoRA**: rank 16, alpha 32, targets q_proj/v_proj/up_proj/down_proj, dropout 0.05
- **Training**: HuggingFace Trainer, 512-token chunks of the pre-patch source file,
  early stopping (patience 3), step cap scaled with file size (n_tokens/1000×100)
- **Inference**: SEARCH/REPLACE block output format, programmatic conversion to
  unified diff via `difflib.unified_diff`
- **Evaluation**: SWE-Bench Docker harness (swebench 4.1.0)
- **Hardware**: RTX 5070 Ti 16GB
- **Budget**: 11,500 tokens (from exp_0 characterization)
- **Thinking mode**: Disabled (`enable_thinking=False`) — mixing modes corrupts signal

## Pilot Results (n=12)

Selected 12 instances across 4 repos: sympy (4), django (4), matplotlib (2),
sphinx (1), scikit-learn (1).

### Training

All 12 oracle LoRAs trained successfully (~18 min total):
- Training time: 58–149s per instance
- Final train loss: 0.50–1.21
- Best eval loss: 0.75–1.65

### Resolve Rates

| Condition | Resolved | Rate |
|-----------|----------|------|
| A         | 0/12     | 0.0% |
| B         | 0/12     | 0.0% |
| C         | 1/12     | 8.3% |
| D         | 0/12     | 0.0% |

- C resolved `django__django-14238`
- C−B gap: 8.3% (passes the ≥5pp prerequisite)
- **Gate ratio: 0.000 — GATE FAILS**

### Patch Generation Quality

| Condition | Valid patches | Avg raw output length |
|-----------|--------------|----------------------|
| A         | 1/12         | —                    |
| B         | 8/12         | 928 chars            |
| C         | 11/12        | —                    |
| D         | 7/12         | 4,850 chars          |

The oracle LoRA makes the model ~5× more verbose but *less* precise: D produces
fewer valid patches than B despite having the same truncated context.

### Edit Distance to Ground Truth

| Condition | Mean | Median |
|-----------|------|--------|
| A         | 32.3 | 30.0   |
| B         | 30.8 | 26.0   |
| C         | 32.4 | 33.0   |
| D         | 79.2 | 35.0   |

D's higher mean edit distance confirms the LoRA pulls the model away from
correct edits rather than toward them.

## Interpretation

The oracle LoRA trained on raw source file content does not help the model
generate correct patches. The likely explanation:

1. **Memorizing ≠ understanding**: Training on source code teaches the model
   to reproduce file contents, not to reason about what changes are needed.
2. **LoRA interference**: The adapter shifts the model's output distribution
   toward source-code-like text, making it verbose and less likely to produce
   concise, targeted edits.
3. **Floor effect**: With only 1/12 resolved even at the ceiling (C), the task
   is too hard for Qwen3-8B at this scale to draw meaningful conclusions about
   LoRA effectiveness.

## Decision

The gate criterion fails (0.000 < 0.500). Based on pilot results, proceeding
to full-scale (103 instances) is unlikely to change the conclusion.

**Recommendation**: Do not scale to full run. Instead, investigate alternative
oracle LoRA training strategies:
- Train on (problem statement, gold patch) pairs rather than raw source
- Use a larger base model
- Use retrieval-augmented generation with longer context instead of LoRA

## Scripts

| Script              | Purpose                                    |
|---------------------|--------------------------------------------|
| `config.py`         | Shared constants and paths                 |
| `helpers.py`        | Data loading, model loading, utilities     |
| `prompts.py`        | System/user prompt templates               |
| `train_oracle.py`   | Train one oracle LoRA per instance         |
| `generate_patches.py` | Generate patches for all 4 conditions   |
| `evaluate.py`       | Run SWE-Bench Docker evaluation            |
| `analyze.py`        | Compute resolve rates, gate criterion, CIs |
| `run_pilot.py`      | Orchestrator for pilot subset              |

## Key Implementation Decisions

1. **SEARCH/REPLACE format**: Switched from asking the model for unified diffs
   (which had wrong line numbers) to SEARCH/REPLACE blocks that are
   programmatically converted to diffs. This dramatically improved patch
   application success.

2. **bf16 + HF Trainer**: Initial attempts with float16 and manual training
   loops produced NaN losses. Switched to bf16 with HuggingFace Trainer for
   stable training.

3. **512-token chunks**: Reduced from 2048 to avoid OOM on 16GB GPU.

4. **Thinking mode disabled**: `enable_thinking=False` is critical — Qwen3's
   thinking mode produces internal reasoning tokens that corrupt the oracle
   training signal.
