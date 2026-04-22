# Stage 0 Lab Notebook

## Entry 2026-04-22 1

- Objective: replace prior Stage 0 GitHub-scraping approach with SWE-rebench leaderboard construction and executable wipe verification.
- Confirmed environment baseline:
  - `uv` available at `/home/seb/.local/bin/uv`
  - shared venv exists at `src/.venv`
  - Python `3.12.13`
- Confirmed leaderboard schema includes required fields:
  - `meta.num_modified_files`, `meta.has_test_patch`
  - `base_commit`, `environment_setup_commit`
  - `install_config.test_cmd`
  - `docker_image`
- Implemented modular Stage 0 scripts and orchestration in `src/stage-0/scripts`.

## Entry 2026-04-22 2

- Validation run:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
python -m compileall stage-0/scripts/extract_function_candidates.py stage-0/scripts/verify_wipe.py
```

## Entry 2026-04-22 3

- Updated verification artifact placement:
  - `scripts/verify_wipe.py` now relocates SWE-rebench run report JSON files (e.g., `stage0-wipe-r*.stage0-verify-r*.json`) into `stage-0/outputs/verify_reports/`.
