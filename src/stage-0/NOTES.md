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
