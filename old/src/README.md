# Experimental Plan — Hypernetwork LoRAs for Code Context

**Target:** demo-able system in 21 days, venue-targetable paper on the same work. Backbone Qwen3-8B throughout. Spine is SHINE IFT'd on code-completion triples; D2L on Gemma-2-2b held in reserve.

## Stage 0 — Dataset construction (Days 1–3)

Build the evaluation set from SWE-rebench with pre-verified Docker harnesses.

Load `nebius/SWE-rebench-leaderboard`; install `SWE-rebench/SWE-bench-fork` (handles `install_config` and `--namespace swerebench`; upstream silently fails on these instances). Filter in sequence: `created_at >= 2025-01-01` (auditable even though the leaderboard satisfies this by construction), `num_modified_files == 1`, `has_test_patch == True`. Pool across monthly splits to reach 300–400 candidates before function-level filtering.

For each candidate, check out the repo at `base_commit` (using `environment_setup_commit` for deps). Tree-sitter-enumerate all functions across all Python files — you are not restricted to the file touched by the gold patch. Keep functions where (i) containing file >2,048 tokens, (ii) body 10–80 lines, (iii) body references ≥1 name defined elsewhere in the same file. Per instance, pick the candidate with most intra-file references. Record: full file, masked version (body → `pass`), ground-truth body, path, `instance_id`, `base_commit`, `docker_image`, file token count, **and the per-file call graph** (needed for Stage 1 slicing and for the Stage 5 demo).

Wipe-verify inside the Docker container: apply the masked file, run the harness, confirm FAIL_TO_PASS tests fail. Discard any instance where the test suite still passes with the body wiped. **This is your execution signal — no instance enters the dataset without it.** Target 80–120 instances post-filter.

**Define evaluation contract once:** pass@1 = complete masked body, apply as patch to `base_commit`, run FAIL_TO_PASS tests via the harness (use `test_cmd` from `install_config` to reproduce independently), binary pass/fail. Bootstrap 95% CIs for all reporting. No BLEU anywhere.

**Contamination figure:** plot Qwen3-8B's Oct-2024 cutoff against `created_at`; cite SWE-rebench's methodology rather than re-arguing it.

**Gate:** if <80 instances survive, extend month coverage before proceeding. Stage 1 is useless without sufficient held-out signal.

## Stage 1 — Diagnostic baseline (Day 4)

Three conditions on 20 instances:

| Condition | Context | LoRA |
|---|---|---|
| B | Truncated ≤ 1024 tok (file head) | None |
| C | Full file in context | None |
| D | BM25 retrieval ≤ 1024 tok (chunks within repo) | None |

**Gate:** C − B must exceed 0.15 pass@1. If not, the base model already solves these with truncation and there's nothing for a hypernetwork to recover — drop truncation budget to 512 tokens and retest. D exists to keep you honest: retrieval-only baselines will exist in reviewer reports, and you want to know how far you are from BM25.

## Stage 1a — SHINE infrastructure verification and zero-shot (Day 5)

**Morning (critical path):** clone `Yewei-Liu/SHINE` into `vendors` directory, pull the Qwen3-8B hypernetwork checkpoint, run their `inference.ipynb` on one of their example QA tasks end-to-end. Must succeed on your 5070 in 8-bit base weights. **Gate (hard):** if the checkpoint isn't released, gated, or doesn't load on your hardware, trigger contingency C immediately — don't spend the afternoon debugging their repo.

**Afternoon:** run SHINE zero-shot on 20 of your Stage 0 instances. Context is a tree-sitter slice — target function's signature, its transitively-called functions (one hop via the call graph), module-level constants referenced, class attributes used — capped at 1,024 tokens. Question is the function signature + docstring; expected answer is the body. Record pass@1. This is **E-zero** for your results table.

**Log slice-coverage diagnostic:** across your instances, what fraction of intra-file references ended up in the 1,024-token slice? Target median ≥70%. If below, your slicer is lossy and Stage 1b will be held back by context loss rather than method limitations.

## Stage 1b — IFT SHINE on code (Days 6–9)

Build ~500 (code-slice, signature+docstring, body) triples from SWE-rebench **training** split (repo-disjoint from test). Same Stage 0 function-selection filters; same tree-sitter slicing as Stage 1a.

Train using SHINE's IFT recipe (paper Eq. 12): cross-entropy on answer tokens only, LoRA generated from the slice-as-context. Only the M2P Transformer, Meta LoRA, and initial memory embeddings update; Qwen3-8B base frozen. Start with 1 epoch at lr=1e-5 matching their 1QA stage; base in 8-bit.

Evaluate on 20 held-out instances. Primary metric is recovery ratio `ρ = (E − B) / (C − B)`.

**Gates and branches:**
- **ρ > 0.25 → Path A (strong).** Proceed to Stage 2 to scale.
- **0.10 ≤ ρ ≤ 0.25 → Path B (weak).** Scale dataset to 1,500 triples, run a 2nd epoch, remeasure. If improvement plateaus, accept and proceed to Stage 2 with a narrower claim.
- **ρ < 0.10 → Path C (fail).** Trigger contingency: D2L on Gemma-2-2b.

## Stage 2 — Scale the winning recipe (Days 10–13)

Conditional on Stage 1b outcome.

- **Path A or B:** scale IFT dataset to 1,500–3,000 triples. Run the one ablation that earns a paper section — *code-aware slicing via call graph vs naive truncation-to-1024*. Everything else (rank sweeps, Meta LoRA size) is a stretch and skipped unless a day is recovered from earlier stages.
- **Path C:** port SakanaAI's D2L to Gemma-2-2b, train end-to-end with context distillation (teacher: full-file context; student: hypernetwork LoRA, no context). 200–400 training pairs. This is a weaker base model and a weaker publication, but you still have a demo.

## Stage 3 — Evaluation (Days 14–16)

Five conditions on held-out instances:

| Condition | Context | LoRA | Role |
|---|---|---|---|
| B | Truncated 1024 | None | Baseline |
| C | Full file | None | Ceiling |
| D | BM25 retrieval 1024 | None | Retrieval baseline |
| E-zero | Slice 1024 | Pretrained SHINE | Zero-shot baseline |
| E | Slice 1024 | SHINE-IFT (your method) | Main claim |

Primary result: `ρ = (E − B) / (C − B)` with bootstrap 95% CIs. Secondary: hypernetwork latency per file (target <1s; SHINE reports 0.3s); context tokens saved vs condition C.

**Capability-interference check:** 20 instruction-following probes on 10 randomly-sampled predicted LoRAs, before/after injection. Flag any adapter causing >15% drop on the probes. This is non-optional — if predicted LoRAs routinely break instruction-following, your Stage 5 demo won't work and you need to know before building it.

**Ablation:** code-aware slicing vs naive truncation, same SHINE-IFT checkpoint, same test set. One ablation only.

## Stage 4 — Static assistant demo (Days 17–19)

Pick a hero repo outside train and test — `httpx` or `rich`. Pre-generate and cache predicted LoRAs for 10–15 files. Gradio app with three panels: file tree + drag-and-drop (left), tree-sitter function list (middle), side-by-side completion output B vs E with differing tokens highlighted (right). Metadata strip: "LoRA generated in Xs · 0 extra context tokens · File: Y tokens (budget: 1024)."

Curate one hero example in advance: a file where baseline visibly hallucinates a helper name and LoRA gets it right. Rehearse this before opening to audience input.

## Stage 5 — Parametric version tracking demo (Days 20–21)

Add a "Version Tracking" tab. Pick a file with a meaningful commit-to-commit change (new helper added, internal API renamed, constant changed) and a function in the same file that depends on the changed code. Show:

1. V1 of the file. Run completion. LoRA reflects V1.
2. Display the commit diff with syntax highlighting.
3. "Refresh LoRA" button → re-run hypernetwork on V2, show timer.
4. Run the same completion. LoRA now reflects V2; baseline unchanged both times.

Label: **"No retraining. No context overhead. LoRA refreshed in Xs."** In the paper, call this *parametric version tracking*, not continual learning.

---

## Paths to demo

Each path terminates in something you can show.

**Path A — full demo (assumes Stage 1b: ρ > 0.25).** Stages 4 + 5 as written. Strong empirical headline, live version-tracking visual. Paper claim: *hypernetwork-generated file-LoRAs recover X% of full-context performance at Y% of the context cost, update in Zs from code diffs.*

**Path B — reduced demo (assumes 0.10 ≤ ρ ≤ 0.25).** Stage 4 as written but curate hero examples harder — run completion on 50 instances, cherry-pick the 5–10 where LoRA clearly wins, use those for live demo. Stage 5 runs as a scripted segment rather than live (V1→V2 is pre-recorded if interactive updates are unreliable on random inputs). Paper claim is softer — *demonstrates parametric memory for code in principle* — with the ablation doing the heavy lifting.

**Path C — contingency demo (Stage 1b failed, D2L worked).** Gemma-2-2b base means worse pass@1 absolute numbers, but (E−B)/(C−B) can still be meaningful because baseline is also worse. Drop SWE-rebench as primary eval; switch to CrossCodeEval or a custom micro-benchmark where Gemma-2-2b is stronger. Stage 4 demo works unchanged. Stage 5 works unchanged. Paper is now a short-paper or workshop submission rather than a TiSE main track.

**Path D — concept demo (both 1b and contingency failed).** You still have infrastructure: SWE-rebench pipeline, tree-sitter slicer, Gradio app, hypernetwork loading. The demo becomes *"here is what parametric version tracking would look like"* with SHINE zero-shot predictions and no strong empirical claim. The live "Refresh LoRA in 0.3s" visual still works — it's fast because of architecture, not because of training. Paper is a position paper or deferred. This is the floor, not the target.

---

## Decision log

- **Drop DyPRAG pilot entirely.** SHINE's IFT recipe supersedes the two-stage oracle-then-translator approach.
- **Drop oracle-LoRA training entirely.** SHINE authors argue (and your Phase 1 confirmed) that segment-wise MLP hypernetworks targeting pre-trained oracle weights don't capture global parameter dependencies. Stage 1 of the original plan is gone.
- **Keep D2L on Gemma-2-2b as contingency only.** Don't schedule it; trigger only on Stage 1b failure.
- **One ablation in the paper, not two.** Code-aware slicing is your defensible methodological contribution. MSE-vs-behavioral-regularization is out — no MSE anywhere now that oracles are gone.
- **Frame SHINE as prior work, not competitor.** Extension to code + execution-signal evaluation + version-tracking demo are the three contributions. Cite them prominently.