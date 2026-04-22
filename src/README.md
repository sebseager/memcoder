## Experimental Plan

---

### Stage 0 — Dataset Construction (Days 1–3)

[TOTAL REWRITE FROM EARLIER VERSION -- NOW USES SWEBENCH INSTEAD OF GITHUB SCRAPING.]

**Goal:** Build a contamination-safe completion dataset from SWE-rebench instances with pre-verified, pre-built Docker harnesses.

1. **Load from `nebius/SWE-rebench-leaderboard`.** This is the leaderboard split, not the full corpus — the key advantage being that every instance already has a pre-built Docker image on Docker Hub under the `swerebench` namespace, so there is no environment build step to debug. Install the nebius SWE-bench fork (`SWE-rebench/SWE-bench-fork`) rather than upstream SWE-bench — it handles the `install_config` field and the `--namespace swerebench` flag correctly; the upstream harness will silently fail on these instances.

2. **Filter for contamination safety and task structure.** Apply the following filters in sequence:
   - `created_at >= 2025-01-01` (well past Qwen3-8B's October 2024 cutoff — all leaderboard instances satisfy this by construction, but make it explicit in your filtering code so it's auditable)
   - `meta.num_modified_files == 1` (single-file gold patch; multi-file patches make LoRA attribution ambiguous)
   - `meta.has_test_patch == True` (confirms FAIL_TO_PASS tests exist)
   - Pool across months (pull from the monthly splits using `created_at` prefix filtering) until you have a candidate set of 300–400 instances before your function-level filters below

3. **Construct completion instances.** For each candidate instance, check out the repo at `base_commit` using the `environment_setup_commit` field for dependency resolution. Use tree-sitter to enumerate all functions across all Python files in the repo — you are not constrained to the file touched by the gold patch, which you can ignore entirely from here on. For each function, apply the following filters:
   - The containing file exceeds 2048 tokens when fully read
   - Body is 10–80 lines
   - Body references names defined elsewhere in the same file (class attributes, module-level constants, other functions) — this ensures truncation actually hurts and the function isn't self-contained

   From the functions passing all three, pick the best candidate per instance (prefer functions with more intra-file references). Record: the full file, the masked version (signature + docstring if present, body wiped to `pass`), the ground-truth body, the file path, `instance_id`, `base_commit`, `docker_image`, token count of the full file.

4. **Verify that wiping each candidate function actually breaks the test suite.** Inside the instance's Docker container, apply the masked version of the file (body replaced with `pass`) and run the harness:
   ```
   python -m swebench.harness.run_evaluation \
     --dataset_name nebius/SWE-rebench-leaderboard \
     --predictions_path <masked_patch> \
     --instance_ids <instance_id> \
     --cache_level instance \
     --run_id verify-wipe \
     --namespace "swerebench"
   ```
   If the suite still passes with the body wiped, the function isn't exercised by the tests — discard it and try the next candidate function in that instance. If no function in the instance survives this check, discard the instance. This is your most important filter: it guarantees that every instance in your dataset has a direct execution signal tied to the specific function you're completing. Target 80–120 instances after this pass.

5. **Define your evaluation contract now, once.** Pass@1 means: complete the masked function body, apply it as a patch to `base_commit`, run the FAIL_TO_PASS tests inside the Docker container via the harness, record binary pass/fail. The `test_cmd` in `install_config` gives you the exact pytest invocation per instance — record it so the eval is reproducible independently of the harness. No BLEU fallback anywhere in the pipeline. This contract carries through Stages 1–3 unchanged.

6. **Produce the contamination figure.** Plot Qwen3-8B's October 2024 cutoff as a horizontal line against each instance's `created_at` date. All retained instances should sit above it by construction. Save this for the paper — and because you're sourcing from SWE-rebench, you can cite their contamination-control methodology directly rather than having to argue the point yourself.

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

7. Evaluate with pass@1 (execute against repo test suite) as primary metric. BLEU-4 as secondary.

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
