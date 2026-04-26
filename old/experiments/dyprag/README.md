## Phased Experimentation Plan

### Pre-Phase: Architecture Lock-in and Dataset Characterization
*Must complete before any experiment runs. Estimated: Week 1, days 1–3.*

These decisions cannot change after oracle LoRA training begins, because the LoRA configuration defines the hypernetwork's output shape.

**Architecture decisions to lock in:**

**Base model.** `Qwen/Qwen3-8B` (Apache 2.0), loaded in 4-bit quantization via bitsandbytes/Unsloth. Confirmed specs from the HuggingFace model card:

- 8.2B total parameters, 6.95B non-embedding
- 36 transformer layers
- Grouped Query Attention with 32 Q heads and 8 KV heads
- Native context length: 32,768 tokens

There is no Qwen3-Coder at this parameter count. The smallest Qwen3-Coder variant (30B-A3B MoE) requires ~15GB for inference alone, leaving no headroom for training. Qwen3-8B at 4-bit (~5GB) leaves ~11GB for LoRA training overhead, oracle training, and the hypernetwork MLP — verified feasible by Unsloth's own documentation for 16GB GPUs.

**Thinking mode.** Qwen3-8B switches between thinking and non-thinking modes within a single model. Non-thinking mode is activated by passing `enable_thinking=False` to the chat template, or by appending `/no_think` to user prompts. All inference in this project — oracle LoRA training, hypernetwork training, agent loop evaluation, and SWE-bench runs — must use `enable_thinking=False`, set at the chat template level, not the prompt level. Lock this in the training script as a hardcoded constant. If any run accidentally uses thinking mode, the token distribution the LoRA is trained on will not match evaluation, silently corrupting the oracle quality signal.

**LoRA configuration.** Rank 16, target layers: `q_proj`, `v_proj`, `up_proj`, `down_proj`. These are fixed for the entire project and must not change after oracle LoRA training begins.

Due to GQA, `v_proj` has a smaller output dimension than `q_proj`. Specifically, Q projects to 32 heads × head\_dim while V projects to only 8 KV heads × head\_dim, making the v\_proj B matrix roughly one-quarter the size of the q\_proj B matrix. The hypernetwork must output separate weight shapes for each target layer — a single shared output head will not work. Use four separate output MLPs, one per target layer, each predicting the correct A and B matrix shapes for that layer. Verify the exact hidden and intermediate dimensions by inspecting `config.json` before writing the hypernetwork's output layer code.

**Hypernetwork encoder.** Frozen `sentence-transformers/all-MiniLM-L6-v2` (~22M parameters, ~80MB) over 512-token overlapping chunks with mean pooling, producing a 384-dimensional file embedding. This runs comfortably alongside the quantized base model within the 16GB budget.

**Agent loop.** ReAct loop over `Qwen/Qwen3-8B` with `enable_thinking=False`. Tool calls parsed from structured output using the model's native function-calling template. `USE_FILE` routing implemented as a system-prompt-level pattern, not a vocabulary extension.

**Seed.** Set before Phase 0 day one: `SEED=42` for oracle LoRA training initialization, hypernetwork initialization, and agent loop sampling temperature. Document in the paper.

**Exp 0 — Dataset characterization**

For every instance in SWE-Bench Lite, tokenize the ground-truth relevant files and sum their lengths. Produce:

1. A histogram of total relevant-file token counts across instances
2. The context-constrained subset: instances where the sum exceeds 4,096 tokens
3. The unconstrained subset: instances where the sum fits within budget

*Logical gap check:* if the context-constrained subset is fewer than 80 instances, raise the budget threshold to 3,072 or lower it to 6,144 until you have a working set of 80–120. If the full-context baseline (condition C below) barely outperforms truncated-context on the chosen subset, the budget cutoff is too generous and the experiment is uninformative. Check this before training a single LoRA.

Also profile here: can the base model + a rank-16 LoRA + the chunked encoder fit simultaneously on your hardware? If not, the encoder must be frozen at sentence-transformer level for all phases. Discover this now, not in Week 5.

---

### Phase 0 — Oracle Viability
*Weeks 1–3. Question: on context-constrained instances, does oracle LoRA + truncated context recover more than 50% of the resolve-rate gap between truncated and full context?*

**Exp 1 — Oracle ceiling (Weeks 1–2, gates everything downstream)**

Run four conditions on the context-constrained subset:

| Condition | Context | LoRA | Role |
|---|---|---|---|
| A | None | None | Floor |
| B | Truncated (≤ budget) | None | Realistic RAG stand-in |
| C | Full (no truncation) | None | Information ceiling |
| D | Truncated (≤ budget) | Oracle | Your method's oracle |

For multi-file instances in Phase 0, train one oracle LoRA per relevant file and inject the simple average before generation. This isolates the "does parametric injection help at all" question from the composition question, which is Phase 2's job.

Primary measure: resolve rate per condition. Secondary measure: patch edit distance to ground truth (catches near-misses and gives signal even when absolute resolve counts are low).

*Logical gap check:* condition C must exceed condition B by at least 5 percentage points for the experimental setup to be informative. If it doesn't, the budget cutoff is too high — the files that "exceed budget" barely do so and truncation isn't losing much information. Adjust the cutoff and re-run characterization before treating Exp 1 results as meaningful.

**Gate criterion:** D recovers more than 50% of the B→C gap. Formally: `(resolve_D − resolve_B) / (resolve_C − resolve_B) > 0.5`. Report bootstrap 95% CIs on this ratio, not just the point estimate.

*What each outcome means:*

- Gate passes cleanly → proceed to Phase 1 as designed.
- D approaches C but the ratio falls below 0.5 → the oracle LoRAs are under-trained or the averaging across files is losing signal. Re-run with longer oracle training or file-level injection only before concluding.
- D is indistinguishable from B → parametric injection is not adding information beyond truncated context. This is a legitimate negative result. Reframe before Phase 1 (see original risk register).
- D exceeds C → unexpected; verify there's no data leakage between oracle LoRA training and evaluation.

**Capability interference check (run in parallel with Exp 1)**

For each oracle LoRA, run 20 instruction-following prompts unrelated to the target instance before and after injection. If mean performance drops more than 15% consistently, the injection mechanism needs a capability regularization term added to oracle LoRA training before Phase 1. Discover this now, not after training 400 hypernetwork training pairs.

---

**Exp 2 — Retrieval granularity (Week 2–3)**

Using the same context-constrained subset, train oracle LoRAs at three granularities for instances where applicable:

- Function-level: the single function being patched
- File-level: the full relevant file (this is what Exp 1 uses)
- Multi-file combined: all relevant files merged into one LoRA

Compare oracle resolve rate and training loss convergence per granularity. Expected result: file-level balances specificity and coverage. The result here directly determines what unit the hypernetwork is trained to predict in Phase 1 — do not proceed to Phase 1 architecture finalization until this is settled.

*Logical gap:* function-level LoRAs require knowing which function is being patched, which SWE-Bench's ground-truth data provides. Verify this is available for your instance subset before committing to this condition; if not, drop it and compare only file vs. combined.

---

### Phase 1 — Hypernetwork Training
*Weeks 4–9. Question: does a trained hypernetwork produce LoRAs that preserve meaningful task performance, and does the context-efficiency framing hold with predicted rather than oracle LoRAs?*

**Week 4 — Data pipeline and architecture finalization**

Collect (file, oracle\_LoRA) training pairs from the SWE-Bench full training split (not Lite). Enforce a repository-level train/test split — no repository appearing in Lite's test set should appear here. Target 400–500 pairs; profile oracle LoRA training throughput on your hardware in the first two days of this week to know if that's achievable.

*Logical gap:* rank-16 LoRA on Qwen2.5-Coder-7B means each A matrix is 16 × 3584 and each B matrix is 3584 × 16. Across four target layers that's approximately 1.8M output dimensions for the hypernetwork MLP. This is large. Before building the MLP, verify empirically whether a direct MSE regression from a ~768-dimensional file embedding to 1.8M output values converges at all on a toy subset of 20 pairs. If it doesn't, reduce rank to 8 or decompose the output head per layer (four smaller MLPs, one per target layer, each predicting one A/B pair). Do not skip this sanity check — this is the most likely silent failure mode in the entire plan.

**Hypernetwork encoder:** frozen sentence-transformer over 512-token overlapping chunks with mean pooling for Phase 1. Replace with trained cross-attention in Phase 1 Week 7 only if reconstruction fidelity in Exp 3a is unsatisfactory.

---

**Exp 3a — Reconstruction fidelity (Weeks 5–6)**

Train the hypernetwork on the 400-pair training set. On a held-out set of 30 training-split instances (not from Lite), plot resolve rate using predicted LoRAs against resolve rate using oracle LoRAs. This is the hypernetwork's proof of concept.

Do not use cosine similarity in weight space as a proxy for task performance — the relationship between weight-space distance and behavioral distance is nonlinear and not interpretable. Measure task performance directly.

*Logical gap:* running the SWE-Bench test suite for 30 instances × 2 conditions takes real wall-clock time. Profile this during Week 4 so you know if it fits in the Week 5–6 window. If evaluation is too slow, reduce to 15 instances and accept wider CIs; don't skip the direct measurement and fall back to weight-space proxies.

**Gate criterion:** predicted LoRA resolve rate is at least 70% of oracle LoRA resolve rate on the held-out set. If it falls below this, the hypernetwork is underfitting — extend training, reduce rank, or check data pipeline integrity before proceeding.

---

**Exp 3b — Identifier-weighted behavioral loss (Weeks 7–8, optional)**

Implement as a behavioral distillation step after the primary MSE pass: apply predicted LoRA to a set of code completion examples, reweight token-level loss by λ=3 on Python identifier tokens, backpropagate through the hypernetwork. This introduces a generation step inside the training loop.

Profile memory and wall-clock cost before committing. If the cost exceeds available weekly compute, demote to future work. This is a clean 2-condition ablation (MSE-only vs. MSE + behavioral identifier penalty) that adds a paper section if it works and costs nothing if demoted.

*Logical gap:* the behavioral distillation step requires a fixed set of code completion examples that are independent of the target instances. Construct this set during Week 4 alongside the data pipeline, not ad hoc during Week 7.

**Week 9 — Buffer and Phase 1 writeup**

The original plan had no Week 9. Reserve this explicitly for: debugging any reconstruction fidelity issues from Exp 3a, finalizing the composition mechanism design, and writing Phase 0–1 results before Phase 2 begins. Do not start Phase 2 without having written up Phase 0–1 — the writing often surfaces gaps that running more experiments won't fix.

---

### Phase 2 — Composition and Agent Loop
*Weeks 10–13. Question: how should LoRAs from heterogeneous files be composed, and does the full pipeline hold under end-to-end agent evaluation?*

**Routing mechanism (specify before Week 10)**

The `USE_FILE` routing mechanism needs a concrete implementation decision before this phase begins, not during it. The least brittle option given your constraints: use a prompted pattern where the agent loop's system prompt instructs the model to prefix any file read with `<use_file>PATH</use_file>`, parsed by the loop controller, which then injects the corresponding predicted LoRA. This requires no vocabulary extension, degrades gracefully (if the tag is missing, fall back to context only), and makes the routing decision visible and auditable. Define the fallback behavior explicitly — unrouted steps use truncated context with no LoRA.

---

**Exp 4 — Composition strategies (Weeks 10–11)**

On the multi-file subset of SWE-Bench Lite (instances touching 3+ relevant files), compare three strategies using predicted LoRAs:

- Average: average predicted LoRA weight matrices before injection, inject once
- Sequential: inject each file's LoRA at the routing step that returns that file; detach after
- Learned weighted sum: small attention layer trained on Phase 0 oracle pairs that produces file-importance weights given the issue description

*Logical gap:* the learned weighted sum requires training data from Phase 0 oracle pairs. Build and save those weights during Phase 0, not retroactively in Week 10. If you don't have them, this condition drops to a two-way comparison, which is still clean.

Build and verify sequential routing on single-file instances first. If routing fails on single-file, don't test composition.

---

**Exp 5 — End-to-end evaluation (Weeks 12–13)**

Full evaluation on SWE-Bench Lite (both constrained and unconstrained subsets separately). Conditions:

| Condition | Context | LoRA | Notes |
|---|---|---|---|
| B | Truncated | None | Realistic baseline |
| C | Full | None | Information ceiling |
| E | Truncated | Predicted (best composition) | Your method |
| F | Truncated | Predicted (no composition, single avg) | Composition ablation |

Report resolve rate with bootstrap CIs on the context-constrained subset as the primary result, and on the full Lite set as a secondary result. The primary claim lives on the constrained subset — that's where the method has a reason to work.

*Logical gap:* Exp 5 is the paper's primary result table, which means it needs to be reproducible. Lock the random seed for oracle LoRA training, hypernetwork initialization, and agent loop sampling before Phase 0 begins, and document them. Discovering mid-Exp-5 that results aren't reproducible is a bad place to be.

---

### Gap Summary

Five gaps worth keeping visible throughout:

1. **Hypernetwork output dimensionality.** Sanity-check MLP convergence on 20 toy pairs in Week 4 before building the full training pipeline.
2. **Context-constrained subset size.** Verify in Exp 0 that the subset is large enough and that full-context actually outperforms truncated by a meaningful margin. If not, adjust the budget threshold.
3. **Evaluation throughput.** Profile SWE-Bench test suite execution time in Week 4. Every experiment that measures resolve rate is bottlenecked by this.
4. **Learned weighted sum training data.** Build and save from Phase 0 oracle pairs, not retroactively.
5. **Reproducibility.** Lock seeds before Phase 0 day one.

---

### Timeline

| Weeks | Phase | Gate |
|---|---|---|
| 1, days 1–3 | Architecture lock-in + Exp 0: dataset characterization | Constrained subset ≥ 80 instances; C exceeds B by ≥ 5pp |
| 1–2 | Exp 1: Oracle ceiling | D recovers > 50% of B→C gap |
| 2–3 | Exp 2: Retrieval granularity | Determines injection unit for Phase 1 |
| 4 | Data pipeline + MLP convergence check | 400–500 oracle pairs; MLP converges on 20-pair toy test |
| 5–6 | Exp 3a: Reconstruction fidelity | Predicted LoRA resolve ≥ 70% of oracle resolve |
| 7–8 | Exp 3b: Identifier loss (optional) | Cut if compute prohibitive |
| 9 | Buffer + Phase 1 writeup | Phase 0–1 results written before Phase 2 begins |
| 10–11 | Exp 4: Composition strategies | Single-file routing verified before multi-file tested |
| 12–13 | Exp 5: End-to-end evaluation | Primary result table |