# Stage 0 - Dataset Construction

This stage builds a contamination-safe function completion dataset from recently active Python repositories.

## Environment

The project uses the `uv` virtual environment located at `src/.venv`.

Pinned package versions used for this run are captured in `requirements.lock.txt`.

## Scripts

- `scripts/discover_repos.py`: queries GitHub API and selects candidate repos.
- `scripts/clone_repos.py`: clones selected repos into `data/repos/`.
- `scripts/setup_repo_env.py`: creates repo-local `.venv` environments and installs dependencies.
- `scripts/score_test_coverage.py`: runs baseline pytest once per repo and filters/weights repos.
- `scripts/build_instances.py`: extracts completion instances and keeps only testable instances.
- `scripts/plot_contamination.py`: creates the contamination cutoff figure.
- `scripts/run_stage0.sh`: executes all Stage 0 scripts in sequence.

## Typical Run

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
python scripts/discover_repos.py --max-search-pages 3 --per-page 40 --sleep-seconds 0.2
python scripts/clone_repos.py
python scripts/setup_repo_env.py
python scripts/score_test_coverage.py
python scripts/build_instances.py
python scripts/plot_contamination.py
```

## Outputs

- `outputs/repo_candidates.json`
- `outputs/env_setup.json`
- `outputs/test_coverage.json`
- `outputs/instances.jsonl`
- `outputs/instances_summary.json`
- `outputs/contamination_first_commit.csv`
- `outputs/contamination_first_commit.png`
- `outputs/instances/<instance_id>/...`
