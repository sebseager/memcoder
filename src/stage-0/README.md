# Stage 0 - SWE-rebench Dataset Construction

This stage builds a contamination-safe, execution-grounded completion dataset from
`nebius/SWE-rebench-leaderboard`.

## Pipeline

1. `scripts/filter_leaderboard.py`
   - Loads explicit month splits from `nebius/SWE-rebench-leaderboard`.
   - Applies hard filters:
     - `created_at >= 2025-01-01`
     - `meta.num_modified_files == 1`
     - `meta.has_test_patch == true`
   - Writes:
     - `outputs/01_filtered_instances.jsonl`
     - `outputs/01_filter_counts_by_month.csv`
     - `outputs/01_filter_counts_by_month.png`
     - `outputs/01_filter_summary.json`

2. `scripts/extract_function_candidates.py`
   - Checks out each repo at `base_commit` (cache under `stage-0/cache/repos`).
   - Uses tree-sitter to enumerate Python functions across repository files.
   - Keeps functions where:
     - containing file token count > 2048,
     - body length is 10-80 lines,
     - body references names defined elsewhere in the same file.
   - Excludes files touched by the gold patch.
   - Ranks candidates by external-reference richness.
   - Writes:
     - `outputs/02_function_candidates.jsonl`
     - `outputs/02_ranked_instance_candidates.jsonl`
     - `outputs/02_candidate_stats.csv`
     - `outputs/02_candidate_count_hist.png`
     - `outputs/02_candidate_summary.json`

3. `scripts/prepare_candidate_attempts.py`
   - Builds top-k wipe attempts per instance by masking candidate function body to `pass`.
   - Writes:
     - `outputs/03_candidate_attempts.jsonl`
     - `outputs/03_attempts_summary.json`

4. `scripts/verify_wipe.py`
   - Verifies candidate relevance in Docker harness by applying:
     - `gold_patch + mask_patch`
   - Accepts candidates where the combined patch is applied and no longer resolves tests.
   - Writes:
     - `outputs/04_verify_runs.jsonl`
     - `outputs/04_verified_instances.jsonl`
     - `outputs/04_verify_summary.json`

5. `scripts/finalize_instances.py`
   - Materializes final artifacts per accepted instance:
     - full file
     - masked file
     - ground-truth body
     - function source
     - mask patch
   - Writes:
     - `outputs/instances.jsonl`
     - `outputs/instances.csv`
     - `outputs/instances/<instance_id>/...`
     - `outputs/05_final_summary.json`

6. `scripts/plot_contamination.py`
   - Generates contamination table and figure against the model cutoff line.
   - Writes:
     - `outputs/contamination.csv`
     - `outputs/contamination.png`
     - `outputs/06_contamination_summary.json`

## Environment

Use the shared `src/.venv` managed by `uv`.

```bash
cd /home/seb/Developer/Classes/continual-learning/src
bash stage-0/scripts/setup_env.sh
source .venv/bin/activate
```

## Runs

Tiny:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
bash scripts/run_stage0.sh --mode tiny
```

Small:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
bash scripts/run_stage0.sh --mode small
```

Full:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
bash scripts/run_stage0.sh --mode full
```
