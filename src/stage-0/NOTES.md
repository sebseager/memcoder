# Stage 0 Lab Notebook

## Entry 2026-04-21 1

- Objective: start Stage 0 dataset construction pipeline in `src/stage-0`.
- Created a `uv` environment in `src/.venv`.
- Initial attempt used Python 3.14 and failed because `tree-sitter-languages` has no `cp314` wheels.
- Recreated `.venv` with Python 3.12 using:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
rm -rf .venv
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install requests pandas matplotlib tree-sitter tree-sitter-languages tiktoken python-dateutil
```

- Observation: Python 3.12 resolves dependency compatibility for tree-sitter tooling.

## Entry 2026-04-21 2

- Set up Stage 0 workspace directories:

```bash
mkdir -p src/stage-0/scripts
mkdir -p src/stage-0/data/repos
mkdir -p src/stage-0/outputs
```

- Implemented modular scripts:
  - `scripts/discover_repos.py`
  - `scripts/clone_repos.py`
  - `scripts/build_instances.py`
  - `scripts/plot_contamination.py`
  - `scripts/run_stage0.sh`
  - `scripts/config.py`

- Design note: scripts are intentionally single-purpose so each experiment step can be rerun independently.

## Entry 2026-04-21 3

- Ran repository discovery with a strict contamination cutoff and active-maintenance filter.
- Initial runs surfaced two issues:
  - JSON serialization failed on `date` objects in config output.
  - Candidate yield was too low when query/sort settings were too narrow.
- Fixes applied:
  - Added `config_as_json_dict` in `scripts/config.py` for ISO-date serialization.
  - Updated timestamps to timezone-aware UTC.
  - Improved GitHub query to include `created:>=2024-11-01` and changed sort to stars.
  - Reduced API pressure by checking oldest commit only after passing the Python-file count filter and early-stopping once enough repos were qualified.

- Final discovery command:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
python scripts/discover_repos.py --max-search-pages 1 --per-page 30 --sleep-seconds 0.2 --require-cutoff-pass
```

- Discovery results from `outputs/repo_candidates.json`:
  - raw candidates: 30
  - qualified: 15
  - selected: 15 (meets minimum target of 15)
  - contamination condition: 15/15 pass first-commit cutoff after 2024-11-01

## Entry 2026-04-21 4

- Cloned selected repositories:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
python scripts/clone_repos.py
```

- Clone summary:
  - cloned: 15
  - skipped: 0

## Entry 2026-04-21 5

- First extraction run failed on parser initialization due version mismatch:
  - `tree-sitter-languages` was incompatible with `tree-sitter==0.25.x`.
- Fix:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
uv pip install 'tree-sitter<0.22'
```

- Re-ran extraction and identified one robustness bug where a path ending in `.py` was a directory.
- Patched `scripts/build_instances.py` to skip non-file paths (`if not path.is_file(): continue`).

- Final extraction command:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
python scripts/build_instances.py
```

- Extraction results from `outputs/instances_summary.json`:
  - candidate instances: 2115
  - selected instances: 120
  - meets Stage 0 target (80-120): yes (upper bound reached)
  - selected instances spread across all 15 repos

## Entry 2026-04-21 6

- Generated contamination figure and CSV:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
cd stage-0
python scripts/plot_contamination.py
```

- Outputs:
  - `outputs/contamination_first_commit.png`
  - `outputs/contamination_first_commit.csv`
- Result: 15/15 selected repos are above the cutoff line.

## Entry 2026-04-21 7

- Validation:
  - Ran diagnostics on all Stage 0 Python scripts; no static errors reported.
- Saved an environment lock snapshot for reproducibility:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
source .venv/bin/activate
uv pip freeze > stage-0/requirements.lock.txt
```

- Key dependency pins to preserve compatibility:
  - `tree-sitter==0.21.3`
  - `tree-sitter-languages==1.10.2`
- Stage conclusion:
  - Stage 0 is complete with contamination-safe selected repos and a full function-level completion dataset artifact set.
