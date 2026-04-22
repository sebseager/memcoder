Implement a repo "weighting" system with the following factors in mind. Use them to select a set of high-quality repos for the rest of this experiment.

## Step 1: Extend `discover_repos.py` filters

Add these as hard filters during discovery, before you clone anything.

**Has a test suite.** Check that the repo has a `tests/` or `test_*.py` pattern in the file tree via the GitHub Trees API (one call per candidate). Repos without any test files are disqualified immediately.

**Has a `pyproject.toml` or `setup.py`.** This is a proxy for "installable with uv." Repos that are just loose scripts rarely have real test infrastructure.

**CI is green on HEAD.** Check the GitHub Commits API for the latest commit's status. If the repo's own CI is red on main, your local test runs will inherit those failures and poison your eval signal.

**No obvious external service dependencies.** Scan `requirements` and `pyproject.toml` for known service-dependent packages: `boto3`, `redis`, `psycopg2`, `pymongo`, `stripe`, `twilio`, `sendgrid`, `celery` (with a broker). These don't disqualify automatically but flag the repo for manual review or downweighting.

---

## Step 2: Add a `setup_repo_env.py` script

Run this after `clone_repos.py`, before `build_instances.py`. For each repo:

```
1. uv venv data/repos/<repo-slug>/.venv --python 3.12
2. uv pip install --python data/repos/<repo-slug>/.venv -e ".[dev,test]" 
   fallback: uv pip install -r requirements-dev.txt / requirements.txt
3. Record install success/failure to outputs/env_setup.json
4. Disqualify repos where install fails
```

Key detail: install with `--no-build-isolation` as a fallback for repos with unusual build setups, but log when you had to do it. Repos that need exotic build steps are risky for reproducibility.

---

## Step 3: Add a `score_test_coverage.py` script

This is the most important addition. For each qualified repo, run the test suite once against the **unmodified** repo and record a baseline:

```bash
cd data/repos/<repo-slug>
.venv/bin/pytest --tb=no -q --timeout=30 \
  --ignore=tests/integration \
  --ignore=tests/e2e \
  -x  # stop on first unexpected error (not failure)
```

Record:
- `baseline_pass_count` / `baseline_fail_count` / `baseline_error_count`
- `baseline_exit_code`
- wall-clock runtime

**Disqualify** the repo if:
- exit code is 2 (collection error — test suite is broken)
- runtime > 120s (too slow for per-instance eval loops)
- `baseline_error_count > 5` (too noisy)

**Downweight** repos where `baseline_fail_count > 0` (pre-existing failures contaminate your delta signal). You can still use them, but you need to track the baseline failure set and exclude those tests when evaluating completions.

---

## Step 4: Extend `build_instances.py` with a coverage filter

After you extract candidate functions, for each one patch the file with the **ground truth body** and re-run the test suite, then diff against baseline:

```
delta_tests = tests_passing_after - tests_passing_before
```

Add to each instance's metadata:
- `gt_patch_test_delta`: number of tests that change state
- `affected_tests`: list of test node IDs that changed
- `testable`: bool — `gt_patch_test_delta > 0`

**Only select instances where `testable = True`.** If patching with the ground truth doesn't move any test, the instance has no execution signal and is useless for eval. This is the single most important quality filter.

This does make `build_instances.py` much slower (one pytest run per candidate function). Run it with `--workers N` via `multiprocessing.Pool` partitioned by repo, so repos run sequentially but you parallelize across repos.

---

## Step 5: Snapshot repo state

After selection, for each repo record the exact commit hash you ran against:

```bash
git -C data/repos/<repo-slug> rev-parse HEAD > data/repos/<repo-slug>/.eval-commit
```

Store this in `instances_summary.json`. When you re-run evals in later stages, `git checkout <hash>` before running tests so environment drift doesn't change your baseline.

---

## Step 6: Update instance metadata schema

Add these fields to every selected instance's `metadata.json`:

```json
{
  "env_python": "3.12.x",
  "env_install_method": "editable" | "requirements",
  "baseline_pass_count": 47,
  "baseline_fail_count": 0,
  "baseline_runtime_seconds": 12.4,
  "affected_tests": ["tests/test_foo.py::test_bar", ...],
  "gt_patch_test_delta": 3,
  "testable": true,
  "repo_commit": "abc123..."
}
```

This makes the eval loop in later stages trivial — you already know exactly which tests to run and what the baseline is.

---

## What this gives you

| Filter | Removes |
|---|---|
| Has tests + installable | Repos that were never locally runnable |
| CI green on HEAD | Repos with pre-existing breakage |
| No ext. services (flagged) | Flaky network-dependent tests |
| Baseline runtime < 120s | Slow repos that make eval loops impractical |
| `testable = True` | Functions with no execution signal |

You'll lose some of your 15 repos and some of your 120 instances, but what remains will have clean, trustworthy execution signal — which is more valuable than larger noisy numbers for gating Stage 2+.

The main thing you **don't** get that SWE-bench has is manual validation that the affected tests are *semantically* testing the target function. That's the expensive part. For your purposes, "tests that change state when you patch the function" is a sufficient proxy.