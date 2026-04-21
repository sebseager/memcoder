## Bridge Plan: Pre-Flight to Full n=120 Run

---

### Decision 1 — Layer Targeting (Make Now, Not Later)

Adopt FFN-only (`up_proj`, `down_proj`) and document it as a justified design choice, not an untested assumption. The rationale is solid: DyPRAG uses FFN-only because that's where factual and associative knowledge is stored in transformers — attention layers encode relational structure, FFN layers encode content. For code files the same logic applies: what you want to inject is knowledge of what identifiers, patterns, and APIs exist in the file, not attention routing behavior. Citing DyPRAG here is exactly right and saves you a 3-way ablation that would cost a day of compute and likely confirm the prior work anyway.

One thing to record explicitly in your `run_config.json`: the layer targeting decision, the justification, and the DyPRAG citation. If a reviewer asks, you want the answer pre-written.

---

### Step 1 — Pre-Flight Fixes (Do Before Any Further Runs)

These are blocking. Do all four before running anything at scale.

**1a. Seed locking.** Lock `SEED=42` across oracle training init, generation sampling, and bootstrap CI computation. Add a `run_config.json` written at the start of every run containing: seed, model ID, truncation budget, oracle chunk size, behavioral probe settings, layer targets, and timestamp. Make `analyze_stage1.py` read this file and embed it in every output artifact. You do not want to discover at paper-writing time that two runs used different seeds.

**1b. Behavioral probe determinism.** Fix the probe generation to be deterministic and file-keyed before retraining anything. The probes should be the first N functions in the file by AST order (not random selection), generated once and saved to `adapter_metadata.json` alongside the adapter weights. Add a check in `train_oracle.py` that refuses to retrain if saved probes exist but don't match current generation logic — same principle as your existing provenance checks. This matters because the hypernetwork in Stage 2 is regressing onto oracle LoRA weights as targets; if two adapters for similar files were trained on different probe distributions, you've introduced target noise that has nothing to do with file content.

**1c. Identifier overlap diagnostic.** Implement as a post-hoc script `scripts/identifier_overlap.py` that runs after `generate_completions.py`. For each condition D row: extract Python identifiers from the generated completion using `ast`, extract identifiers from the full file, extract identifiers from the truncated context, then compute the set of identifiers used in the completion that appear in the full file but not the truncated context. Record `novel_identifier_count`, `novel_identifier_rate`, and the actual identifier list per row. This is your qualitative evidence mechanism — you want to see D completions using file-internal names that B completions don't. A reviewer will ask this question.

**1d. Per-instance gap filter.** Add a pre-analysis step that computes the B→C BLEU gap per instance from your small pilot data where available. Flag instances with gap < 0.05 as "low-gap" and add a `gap_stratum` field (`low`, `medium`, `high`) to your evaluation CSV. Report recovery ratios stratified by this field in the paper. Do not exclude low-gap instances from the primary analysis — report them separately. If your recovery ratio is 0.7 overall but 0.1 in the low-gap stratum and 0.9 in the high-gap stratum, that's a more interesting and defensible result than a single number.

---

### Step 2 — Scale Probe on n=12 with Qwen3-4B

Before committing the full compute budget to any model, run the small pilot (n=12) on **Qwen3-4B** with thinking off. This is your model scale decision gate.

Why 4B rather than jumping straight to 8B: you need to know whether the recovery ratio is model-size-sensitive before you commit 79 adapter training runs to the 8B. If 4B shows similar recovery to 1.5B, you have evidence the method is robust to scale and 8B will likely be at least as good. If 4B shows significantly better recovery (higher D absolute performance, similar or wider B→C gap), you have evidence that model capacity matters and should use 8B. If 4B shows *worse* recovery despite better absolute completion quality, something interesting is happening (possibly the larger model's stronger priors are harder to override with a rank-16 adapter) and you need to investigate before scaling.

The run is identical to your existing small pilot — same n=12 instances, same scripts, just `--model-id Qwen/Qwen3-4B` with `enable_thinking=False` locked in config. Retrain adapters from scratch for this model (don't reuse 1.5B adapters). Budget roughly one day of compute.

WARNING: This means stage-1 is now runnable with multiple models. The outputs directory should now be organized by model name, for example:

```
stage-1/
  scripts/          ← single source of truth, all model sizes
  outputs/
    qwen2.5-coder-1.5b-instruct/
      oracle_loras/
      completions/
      evaluation/
      analysis/
      capability/
      plots/
      logs/
    qwen3-4b/       ← new runs go here
      oracle_loras/
      ...
    qwen3-8b/       ← if you scale up
      ...
```

---

### Step 3 — Qualitative Gates on the 4B Run

Run these before looking at aggregate metrics.

**Qualitative Gate A — Completion reading.** Pick the 3 instances with the highest B→C BLEU gap in your small pilot. For each, read the B, C, and D completions side by side against the ground truth. You are checking:
- Does D use function names or variable names that appear in the full file but not the truncated context?
- Does D make the same structural choices as the ground truth (similar control flow, similar error handling patterns)?
- Does B fail in a way that D fixes (wrong identifier, wrong return type, hallucinated method name)?

If you can point to two or three concrete examples where D uses a file-internal name that B cannot know, you have your paper's qualitative exhibit. Document this in NOTES.md. If D completions look indistinguishable from B completions on these instances, the adapter still isn't transferring the right knowledge and you need to diagnose further before scaling.

**Qualitative Gate B — Identifier overlap spot check.** Run `identifier_overlap.py` on the 4B D completions. For the same 3 high-gap instances, check whether `novel_identifier_rate` is positive. You want to see the diagnostic confirming what you observed by eye in Gate A. If the diagnostic shows zero novel identifiers on instances where you visually observed file-internal names being used, there's a bug in the diagnostic to fix before the full run.

**Gate: proceed if at least 2 of 3 high-gap instances show visible qualitative evidence of file-knowledge transfer in D completions.** This is a subjective but necessary check — metrics can lie at small n, direct reading cannot.

---

### Step 4 — Quantitative Comparison: 1.5B vs 4B

Compare the small pilot metrics side by side across both model sizes:

| Metric | 1.5B | 4B | Direction you want |
|---|---|---|---|
| C BLEU (ceiling) | 0.308 | ? | Higher = harder task, more room |
| B BLEU (baseline) | 0.074 | ? | Lower relative to C = bigger gap |
| D BLEU (oracle) | 0.207 | ? | Higher = better injection |
| Recovery ratio (BLEU) | 0.739 | ? | ≥ 0.6 |
| D pass@1 | 0.083 | ? | ≥ 0.083 |
| Novel identifier rate | TBD | ? | > 0 on high-gap instances |

**Quantitative Gate — Model selection decision:**

- If 4B recovery ratio ≥ 0.6 AND 4B absolute D performance ≥ 1.5B D performance → use 4B for full run. 8B adds compute cost without clear justification.
- If 4B recovery ratio ≥ 0.6 AND 4B C BLEU substantially higher than 1.5B C BLEU (ceiling is harder) → use 8B. The task is harder at larger model scale, so the experiment is more interesting and the result more credible to reviewers.
- If 4B recovery ratio < 0.4 despite good absolute C performance → investigate before scaling. Do not run 8B until you understand why a larger model shows weaker recovery.
- If 4B and 1.5B recovery ratios are within 0.1 of each other → the method is robust to scale, use whichever fits your compute budget for n=120. 4B is probably the right call unless you have a specific reason to claim 8B performance.

One practical note: the hypernetwork in Stage 2 predicts LoRA weights as regression targets. Larger model LoRAs have larger weight matrices and potentially noisier oracle targets (more capacity means the oracle may fit training pairs in multiple equally-valid ways). If you see high variance in oracle adapter quality at 8B during training loss inspection, that's a signal the hypernetwork will have a harder regression problem, not just a bigger one.

---

### Step 5 — Final Pre-Flight Check Before n=120

Once model is selected and all Step 1 fixes are confirmed, run this checklist before launching the full run:

- `run_config.json` written and verified (seed, model, all hyperparams)
- Behavioral probes deterministic and saved in adapter metadata for all 79 file keys
- Layer targets locked to `up_proj`, `down_proj` in config
- `identifier_overlap.py` runs cleanly end-to-end on small pilot outputs
- Gap stratum field populated for all 120 instances
- `max_new_token_hit_rate` from small pilot is under 0.1 (if not, raise `MAX_NEW_TOKENS` before full run — getting cut off at n=120 scale will silently depress D metrics)
- Resumable run confirmed: kill the run halfway through, restart, verify no adapters are re-trained unnecessarily

If all green, you are go to launch full n=120. Expected output is a recovery ratio with a CI lower bound above 0, stratified by gap stratum, with an identifier overlap diagnostic that gives you the qualitative story. That's your Stage 1 paper result.