## Experimental Plan

---

Stage 0 — Dataset Construction (Days 1–3)

[REWRITTEN FROM EARLIER VERSION -- NOW USES SWEBENCH INSTEAD OF GITHUB SCRAPING]

Goal: Build a contamination-safe completion dataset from SWE-bench instances with pre-verified, executable test harnesses.

Source instances from SWE-bench Verified (not Lite). Filter to Python-only instances where the gold patch touches exactly one file — this keeps the completion target unambiguous and avoids multi-file entanglement confounding your LoRA attribution. Target 80–120 instances after filtering.
Contamination cut. Drop any instance whose repo first-commit date falls before Qwen3-8B's estimated training cutoff (Oct 28, 2024). If this leaves no or few repos, stop and ask me what to do. Produce the contamination figure: plot cutoff as a horizontal line against issue dates. All retained instances should sit above it. Save this for the paper.
Construct the completion instances. For each retained SWE-bench instance, the pre-patch state of the touched file is your input; the gold patch body gives you the ground-truth completion. Specifically:

Check out the repo at the pre-patch commit using the SWE-bench instance metadata
Use tree-sitter to locate the function(s) modified by the gold patch within the touched file
Keep only functions satisfying all three: body references names defined elsewhere in the same file (class attributes, module-level constants, other functions); body is 10–80 lines; the containing file exceeds 2048 tokens when fully read
Record: the full pre-patch file, the masked version (signature only), the ground-truth body from the gold patch, and the cross-file context (imported files)


Verify test harness executability upfront, before any model runs. For each instance, use the SWE-bench Docker harness to confirm that: (a) the pre-patch state produces a failing FAIL_TO_PASS test, and (b) applying the gold patch flips it to passing. Discard any instance where either check fails — environment issues, flaky tests, or gold patch errors will silently corrupt your results later and you want zero ambiguity in what passing means. This is the key advantage over scraping GitHub directly: you are curating from a set that has already been through human verification, and you are re-verifying executability yourself before committing to the instance.
Define your evaluation contract now, once. Pass@1 means: complete the masked function, apply it as a patch to the pre-patch repo state using the same patch format SWE-bench uses, run the FAIL_TO_PASS tests via the Docker harness, record binary pass/fail. BLEU-4 is not a fallback metric here — if an instance lacks a runnable harness after Step 4, discard it rather than falling back to BLEU. Execution-based signal only. This contract carries through Stages 1–3 unchanged.
Record instance metadata for later use: repo name, file path, pre-patch commit SHA, token count of the full file, FAIL_TO_PASS test IDs, and whether cross-file context was needed for the gold patch. The token count feeds directly into Stage 1's truncation analysis.


The rest of the plan flows cleanly from this: Stage 1's conditions B/C/D run the same Docker harness, Stage 3 adds condition E, and the recovery ratio (E − B) / (C − B) is computed over binary pass/fail — which makes the primary claim much harder to dismiss. BLEU is a secondary metric only. The "close but wrong" failure mode you're worried about is exactly what execution-based eval catches that BLEU doesn't.

---

### Stage 1 — Oracle Ceiling (Days 4–7)

**Goal:** Confirm that oracle LoRA on truncated context recovers full-context performance before touching the hypernetwork.

5. Train oracle LoRAs for each unique file in your dataset using your existing `train_oracle.py`. Each LoRA is trained on the full file text. This is a batch run — you've already automated it.

6. Run three conditions on all instances:

   | Condition | Context | LoRA | Role |
   |---|---|---|---|
   | B | Truncated ≤ 2048 tok | None | Baseline |
   | C | Full file in context | None | Ceiling |
   | D | Truncated ≤ 2048 tok | Oracle | Oracle upper bound |

7. Evaluate with pass@1 (execute against repo test suite) as primary metric. BLEU-4 as secondary for instances where no test suite is available.

8. Compute the recovery ratio: `(D − B) / (C − B)`. If this is below 0.3, stop and diagnose — either tighten the truncation budget to 1024 tokens to widen the B→C gap, or inspect whether oracle LoRA training loss actually converged. Do not proceed to Stage 2 until D meaningfully exceeds B.

9. Run capability interference check: 20 instruction-following probes before/after oracle LoRA injection per file. Flag any adapter causing >15% drop. You already have `capability_interference.py` — just rerun it on this dataset.

---

### Stage 2 — Hypernetwork Training (Days 8–16)

**Goal:** Train a hypernetwork that predicts file LoRAs from source text alone.

10. **Toy sanity check first (Day 8, ~2 hours).** Take 15 (file, oracle LoRA) pairs. Flatten each oracle LoRA's weight matrices. Train a single-layer MLP from the file's sentence-transformer embedding to the flattened weights using MSE loss. Check that loss descends. If it doesn't, reduce rank from 16 to 8 before building anything else.

11. Collect training pairs from the SWE-bench *training split* (not Lite) to supplement your dataset pairs. Target 80–120 total pairs. Enforce repo-level split: no repo in your test set may appear here.

12. Implement code-aware chunking using tree-sitter: chunk files at function boundaries rather than fixed token counts. Feed chunks through frozen `all-MiniLM-L6-v2` with mean pooling to get a 384-dim file embedding. This is your differentiator from DyPRAG's prose chunking — call it out explicitly in the paper.

13. Build four separate output MLPs, one per target layer (`q_proj`, `v_proj`, `up_proj`, `down_proj`). Each takes the 384-dim embedding and outputs the flattened A and B matrices for that layer. Verify output shapes against Qwen3-8B's `config.json` before writing any training code.

14. Train with MSE on flattened weight matrices as primary loss. Add behavioral regularization: apply the predicted LoRA to 10 fixed held-out completion probes and add cross-entropy on ground-truth tokens at λ=0.1. Run this as an ablation (MSE-only vs MSE + behavioral) — it's a clean one-section paper contribution.

15. Cache all predicted LoRAs for your test instances to disk after training. Hypernetwork inference should take <1 second per file on your GPU — measure and record this, it's a selling point.

---

### Stage 3 — Evaluation (Days 17–20)

**Goal:** Establish the full results table and the core recovery ratio using predicted LoRAs.

16. Run four conditions on held-out test instances (repo-disjoint from training):

   | Condition | Context | LoRA | Role |
   |---|---|---|---|
   | B | Truncated | None | Baseline |
   | C | Full | None | Ceiling |
   | D | Truncated | Oracle | Oracle upper bound |
   | E | Truncated | Predicted | Your method |

17. Report pass@1 with bootstrap 95% CIs on all four. Primary claim: `(E − B) / (C − B)` recovery ratio with CI. Secondary: hypernetwork inference latency vs. context tokens saved.

18. Run the two ablations: E-nocode (fixed-size token chunking instead of tree-sitter) to isolate the code-aware chunking contribution; and MSE-only vs MSE+behavioral from Step 14.

---

### Stage 4 — Demo Part 1: Static LoRA Assistant (Days 21–23)

**Goal:** A live Gradio app showing LoRA injection improving completion on a real, pre-chosen repo.

19. Pick a demo repo that isn't in your training or test set — `httpx` or `rich` are good choices. Pre-generate predicted LoRAs for 10–15 files in it overnight using your trained hypernetwork. Cache to disk. This is the "known files" pool.

20. Build a Gradio app with three panels:
    - **Left:** file selector showing the demo repo's file tree, plus a drag-and-drop zone for arbitrary files
    - **Middle:** tree-sitter-parsed function list from the selected file; user clicks a function to mask it
    - **Right:** side-by-side completion output — Baseline (truncated, no LoRA) vs. With LoRA (truncated + predicted), with differing tokens highlighted

21. Add a metadata strip below the output: "LoRA generated in Xs · 0 extra context tokens used · File: Y tokens (budget: 2048)." The contrast between file size and context tokens used is the visual punchline.

22. For the demo narrative, curate one hero example in advance: a file well above 2048 tokens containing a function that calls several file-internal helpers. Baseline visibly hallucinates wrong function names or wrong return patterns. LoRA version gets it right. Practice showing this first before opening to audience input.

---

### Stage 5 — Demo Part 2: Periodic LoRA Refresh (Days 24–25)

**Goal:** Show the system staying in sync with a live codebase without retraining.

23. Set up a small local Git repo (or use a public one with a clearly bounded change history). Pick a file that undergoes a meaningful edit between two commits — a new helper function added, an internal API renamed, a constant changed.

24. In the Gradio app, add a second tab: **"Version Tracking."** It shows:
    - A two-commit diff of the chosen file (V1 → V2), displayed with syntax highlighting
    - A "Refresh LoRA" button that re-runs the hypernetwork on the updated file (~0.5s)
    - Before/after completion outputs for a function that depends on the changed code

25. The demo flow for Part 2:
    - Show V1 of the file. Run completion. LoRA correctly reflects V1's API.
    - Display the commit diff — a new helper was added, or a constant was renamed.
    - Hit "Refresh LoRA." Timer shown on screen.
    - Run the same completion again. LoRA now reflects V2 — it uses the new helper, or the new constant name.
    - Baseline (no LoRA, truncated) fails both times identically. The LoRA is the only thing that changed.

26. Add a label to this tab: **"No retraining. No context overhead. LoRA refreshed from file diff in Xs."** This is the slide-worthy moment and directly motivates the continual learning framing in the paper's discussion section — use the term *parametric version tracking* rather than continual learning to avoid reviewer comparisons to the catastrophic forgetting literature.
