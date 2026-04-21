## Fix Stage 1: Step-by-Step with Gates

---

### Step 1 — Verify the Current Failure Mode Precisely

Before touching any code, confirm exactly what the oracle LoRA has learned.

For one pilot instance, run this manually:

```python
# With oracle adapter loaded, prompt it with just the raw file continuation task
prompt = "<first 500 tokens of the file>"
# Then with the actual evaluation prompt format
prompt = "<truncated context>\n\ndef function_name(args):"
```

Compare logit distributions (or just top-5 next tokens) between B and D on both prompt types. You're checking whether the adapter has learned *anything at all* task-relevant, or whether it's purely shifting the model toward file-continuation behavior.

**Gate 1:** If B and D produce near-identical logits on the evaluation prompt format, the training distribution mismatch is confirmed and Step 2 is the right fix. If they diverge meaningfully but D output is still worse, stop: the problem is different (adapter interference, merging issue) and needs separate investigation before proceeding.

---

### Step 2 — Rewrite the Oracle Training Data Construction

This is the core fix. In `train_oracle.py`, replace the raw file LM training sequence with supervised (prompt → completion) pairs extracted from the file itself.

For example, training examples could be made as follows:

```python
for each function f in file (parsed via tree-sitter):
    prompt = truncated_file_context_excluding_f + f.signature
    target = f.body

    training_example = {
        "input": prompt,
        "output": target,
        # mask prompt tokens in loss — only supervise on target
    }
```

The truncated context should use the *same* truncation logic as your generation pipeline — `truncate_to_budget()` from `helpers.py` — so the training distribution exactly matches evaluation. This is the key invariant: the model should see at training time exactly the kind of prompt it will see at evaluation time.

Loss masking is important: set labels to `-100` for all prompt tokens, supervise only on the target body tokens. If you train on the full concatenated sequence without masking, the LoRA will spend capacity modeling the truncated context rather than learning to complete functions from it.

If the file has N functions, you get N training pairs. For short files (fewer than 3 functions) consider including cross-file imports as additional context in the prompt to give the LoRA more to work with.

**Gate 2:** After implementing, inspect one training batch manually before running any training. Print the decoded input tokens, confirm the prompt ends at the function signature, confirm the target is only the body, confirm loss mask is applied correctly. Do not proceed to Step 3 until this looks right on paper — a subtle masking bug here will silently poison every oracle adapter.

---

### Step 3 — Retrain One Oracle Adapter and Inspect Directly

Pick the single pilot instance where the B vs C BLEU gap was largest (most to gain). Delete its existing adapter and retrain with the new objective.

After training, run two direct inspections before any automated evaluation:

**Inspection A — Loss sanity.** Check that training loss actually descends to a low value (below 0.5 ideally). If it stays high (above 1.5), the adapter isn't fitting the training pairs — check learning rate, number of epochs, and whether your functions are long enough to give meaningful supervision signal.

**Inspection B — Manual completion.** With the adapter loaded, run the exact evaluation prompt for that instance and print the raw output. Read it. Does it look like it's trying to complete the function in the style of the file? Does it use variable names, helper functions, or patterns from the file that condition B doesn't? You're looking for qualitative evidence that something was learned before you trust any metric.

**Gate 3:** Training loss below 1.0 AND manual output visibly different from condition B in a plausible direction. If training loss is low but output is unchanged, suspect the adapter isn't being loaded correctly for generation (see Step 4's swap check). If training loss is high, tune epochs/lr before proceeding.

---

### Step 4 — Verify Adapter Loading and Swap Behavior

Your Entry 10 noted 1 `loaded_initial` and 11 `swapped`. This needs to be clean before trusting any D results.

Add explicit logging to `generate_completions.py` that for each instance prints:

```
instance_id | adapter_key | load_status | active_before_generate | first_5_output_tokens
```

Specifically verify:
- The adapter is fully active (merged or set as active PEFT adapter) *before* `model.generate()` is called, not after
- The adapter being loaded matches the file key for that instance (not a stale swap from a previous instance)
- After generation, if you're unloading/swapping, the unload completes before the next instance begins

Run this on two instances only — one where the adapter was `loaded_initial` and one that was `swapped` — and confirm both produce different output from B.

**Gate 4:** Zero instances where the adapter key doesn't match the expected file, and zero instances where the active adapter state is ambiguous at generation time. If swaps are unreliable, simplify: load a fresh model for each condition D instance rather than swapping. Slower but correct.

---

### Step 5 — Rerun Tiny Pilot (n=4) with New Objective

With the training fix confirmed on one adapter and loading verified, retrain all 4 tiny pilot adapters and rerun the full tiny pilot.

Look for:
- D BLEU > B BLEU on at least 3/4 instances
- D outputs inspected manually — should use file-internal names/patterns that B misses
- Recovery ratio `(D−B)/(C−B)` on BLEU > 0.3 at n=4 (noisy but directionally right)

**Gate 5:** D BLEU strictly greater than B BLEU in aggregate. This is the minimum bar to proceed. If D is still at or below B after the training fix, go back to Gate 3 and re-examine whether training loss is actually low — the most likely remaining cause is too few training pairs per file (files with only 1–2 short functions won't produce enough supervision signal).

---

### Step 6 — Rerun Small Pilot (n=12)

Once Gate 5 passes, rerun the small pilot with all 12 adapters retrained.

At n=12 you can start reading the recovery ratio more seriously. Targets:
- BLEU recovery ratio > 0.3 (weak but acceptable to proceed)
- BLEU recovery ratio > 0.5 (strong, proceed with confidence)
- At least 1 pass@1 resolve in condition D where B gets 0 (even one is meaningful signal at this scale)

Also check:
- `max_new_token_hit_rate` — if this climbs above 0.2 for condition D, your generations are getting cut off and `MAX_NEW_TOKENS=1024` may still be too low for some instances
- Syntax valid rate for D should be ≥ B — if D is producing more syntactically invalid outputs than B, the adapter is actively hurting generation coherence and the behavioral loss term (Step 7) may be needed sooner

**Gate 6:** BLEU recovery ratio > 0.3 at n=12 with bootstrap CI lower bound above 0. This is the Stage 1 pass condition. Proceed to Stage 2 hypernetwork training only after this gate clears.

---

### Step 7 — If Gate 6 Fails: Two Targeted Interventions

If n=12 recovery is still near zero after the training fix, there are two targeted things to try before concluding Stage 1 is broken.

**Intervention A — Tighten truncation budget.** Change `TRUNCATION_BUDGET` from 2048 to 1024. The B→C gap may be too small at 2048 — files that "exceed budget" may only slightly exceed it, meaning truncation isn't losing much. A tighter budget makes the oracle's job more clearly meaningful and widens the gap the LoRA needs to recover.

**Intervention B — Add a behavioral regularization pass.** After the supervised training on (prompt→body) pairs, run a second short pass where you apply the adapter to 5 completion probes from the *same file* and add cross-entropy loss on the correct continuations. This reinforces that the adapter should activate helpfully on completion prompts specifically, not just fit the training pairs in weight space.

Run each intervention independently, not together, so you can attribute any improvement. Rerun tiny pilot (n=4) after each one — you don't need n=12 to see if the direction is right.

**Gate 7:** Either intervention produces D > B on BLEU at n=4. If neither does, the base model (currently 1.5B) may genuinely be too small to express the injected knowledge usefully, and the right move is to run the tiny pilot on Qwen3-8B before concluding anything about the method. That's a compute decision, but at that point it's the right one.