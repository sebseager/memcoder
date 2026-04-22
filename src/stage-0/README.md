# Stage 0 - Dataset Construction

This stage builds a contamination-safe function completion dataset from recently created Python repositories.

## Environment

The project uses the `uv` virtual environment located at `src/.venv`.

Pinned package versions used for this run are captured in `requirements.lock.txt`.

## Scripts

- `scripts/discover_repos.py`: queries GitHub API and selects candidate repos.
- `scripts/clone_repos.py`: clones selected repos into `data/repos/`.
- `scripts/build_instances.py`: extracts completion instances from cloned repos.
- `scripts/plot_contamination.py`: creates the contamination cutoff figure.
- `scripts/run_stage0.sh`: executes all Stage 0 scripts in sequence.

## Typical Run

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
python scripts/discover_repos.py --max-search-pages 3 --per-page 40 --sleep-seconds 0.2
python scripts/clone_repos.py
python scripts/build_instances.py
python scripts/plot_contamination.py
```

## Outputs

- `outputs/repo_candidates.json`
- `outputs/instances.jsonl`
- `outputs/instances_summary.json`
- `outputs/contamination_first_commit.csv`
- `outputs/contamination_first_commit.png`
- `outputs/instances/<instance_id>/...`
