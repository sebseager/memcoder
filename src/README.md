## Experimental Plan

---

### Stage 0 — Dataset Construction (Days 1–3)

**Goal:** Build a contamination-safe completion dataset using RepoBench's task structure but fresh data.

1. Query GitHub API for Python repos with first commit after November 2024, 50–500 stars, at least 20 Python files, actively maintained (last push within 60 days). Target 15–20 repos.

2. For each repo, use tree-sitter to extract function-level completion instances. Keep only functions that satisfy all three of:
   - Body references names defined elsewhere in the same file (class attributes, module-level constants, other functions) — this ensures truncation actually hurts
   - Body is 10–80 lines (short enough to be a clean target, long enough to be non-trivial)
   - The file containing it exceeds your truncation budget (2048 tokens) when fully read — this is the core experimental condition

3. For each qualifying function, record: the full file, the masked version (signature only), the ground-truth body, and the cross-file context (other files imported). Target 80–120 instances total across your repos.

4. Produce the contamination figure: plot Qwen3-8B's estimated cutoff as a horizontal line against each repo's first-commit date. All repos should sit above the line. Save this — it goes in the paper.

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
