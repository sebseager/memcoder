# Stage 1a Lab Notebook

## Entry 2026-04-24 1

- Objective: implement Stage 1a afternoon pipeline for SHINE zero-shot evaluation on Stage 0 SWE-rebench instances.
- Split execution by resource boundary:
  - `run_stage1a.py` / `run_stage1a.sh`: HPC-friendly SHINE inference, no Docker.
  - `score_stage1a.py` / `score_stage1a.sh`: local scoring and SWE-rebench Docker harness pass@1.
- Source notebook used as implementation reference:
  - `stage-1a/inference.ipynb`
- Stage 1a contract from `README.md`:
  - Use 20 Stage 0 instances.
  - Context: code-aware slice capped at 1024 tokens.
  - Question: target function signature + docstring.
  - Expected answer: missing function body.
  - Metric: pass@1 via SWE-rebench harness.
  - Diagnostic: median slice coverage target >= 70%.

## Entry 2026-04-24 2

- Implemented inference script:
  - `stage-1a/scripts/run_stage1a.py`
  - `stage-1a/scripts/run_stage1a.sh`
- Key behavior:
  - Loads SHINE / Qwen3-8B hypernetwork stack from `vendor/SHINE`.
  - Reads Stage 0 instances from `stage-0/outputs/stage1_instances.jsonl`.
  - Builds per-instance code slices from:
    - target function signature/docstring,
    - one-hop callees,
    - module-level constants,
    - class attributes,
    - external reference index fallback.
  - Writes predictions to `stage-1a/outputs/stage1a_predictions.jsonl`.
  - Logs per-instance and aggregate slice coverage diagnostics.
- Wrapper usage:

```bash
cd /home/seb/Developer/Classes/continual-learning/src/stage-1a/scripts
./run_stage1a.sh --mode full
```

- Other useful modes:

```bash
./run_stage1a.sh --mode tiny
./run_stage1a.sh --mode small
./run_stage1a.sh --max-instances 20
```

## Entry 2026-04-24 3

- Initial HPC inference issue:
  - `Slice coverage median = 0.000`.
  - Several predictions contained Qwen think-mode/prose instead of body-only code.
- Root causes:
  - Artifact hydration was incomplete when Stage 0 artifact files were missing or not inlined.
  - Empty/incomplete `full_file`, `function_source`, `masked_function`, and `ground_truth_body` degraded both slicing and prompting.
  - The Qwen3 chat template needed explicit thinking suppression.
- Fixes applied to `run_stage1a.py`:
  - Added robust artifact hydration:
    - read Stage 0 artifacts when present,
    - fall back to repo checkout at `base_commit`,
    - derive `function_source`, `ground_truth_body`, and `masked_function` from the checked-out full file.
  - Added external-reference coverage fallback sections so diagnostics reflect references actually present in the final selected slice.
  - Changed coverage computation to check regex presence in final slice text.
  - Strengthened generation prompt:
    - body-only,
    - no markdown,
    - no explanation,
    - no think mode.
  - Set `enable_thinking=False` in `tokenizer.apply_chat_template`.
  - Hardened answer extraction for unclosed `<think>` blocks, markdown fences, and prose prefixes.
- Result:
  - Latest prediction file shows median slice coverage `1.0`.
  - Slice coverage gate is met.

## Entry 2026-04-24 4

- Implemented scoring script:
  - `stage-1a/scripts/score_stage1a.py`
  - `stage-1a/scripts/score_stage1a.sh`
- Key behavior:
  - Reads `stage1a_predictions.jsonl`.
  - Computes exact match, BLEU-4 diagnostic, syntax-valid rate, and pass@1.
  - Builds predicted patches by combining:
    - Stage 0 `gold_patch`, and
    - generated replacement for the masked target body.
  - Runs local SWE-rebench harness through:

```bash
python -m swebench.harness.run_evaluation
```

- Wrapper usage:

```bash
cd /home/seb/Developer/Classes/continual-learning/src/stage-1a/scripts
./score_stage1a.sh --force
```

- Exact-only diagnostic mode:

```bash
./score_stage1a.sh --pass-at-1-mode exact_only --force
```

## Entry 2026-04-24 5

- First scoring issue:
  - Scoring finished almost immediately.
  - Summary reported all rows as `syntax_invalid`.
  - Harness did not actually execute tests.
- Root cause:
  - Syntax validation was running on raw model output before normalization.
  - Raw model outputs often had wrong indentation or prose wrappers.
- Fixes applied to `score_stage1a.py`:
  - Added `normalize_body_prediction`.
  - Reindents predictions before syntax validation.
  - Keeps raw prediction in memory for diagnostics.
  - Logs syntax-valid counts before and after normalization.
  - Logs harness submission skip reasons:
    - `syntax_invalid`,
    - `missing_full_file`,
    - `noop_patch`.
- Observation after fix:
  - 17/17 predictions changed during normalization.
  - 10/17 became syntactically valid enough for harness submission.
  - 7/17 remained syntax-invalid.

## Entry 2026-04-24 6

- Second scoring issue:
  - Harness crashed with `KeyError: 'RDFLib/pySHACL'`.
- Root cause:
  - `src/pyproject.toml` depended on PyPI `swebench>=4.1.0`.
  - PyPI `swebench` does not support the SWE-rebench leaderboard `install_config` / `image_name` workflow used by Stage 0.
  - Stage 0 had relied on the SWE-rebench fork, but the project metadata allowed `uv sync` to replace it with the upstream package.
- Source fix:
  - Updated `src/pyproject.toml` to source `swebench` from:

```text
https://github.com/SWE-rebench/SWE-bench-fork.git
```

  - Updated `src/uv.lock`.
  - Synced `src/.venv`.
  - Added a fail-fast check in `score_stage1a.sh` that verifies the installed harness has `install_config` support.
- Verification:

```bash
cd /home/seb/Developer/Classes/continual-learning/src
uv sync
source .venv/bin/activate
python - <<'PY'
import inspect
from swebench.harness.test_spec.test_spec import make_test_spec
assert "install_config" in inspect.getsource(make_test_spec)
print("SWE-rebench harness fork installed")
PY
```

## Entry 2026-04-24 7

- Third scoring issue:
  - Harness ran successfully, but all 10 submitted instances were unresolved.
  - Submitted example patches showed scorer patch-construction errors for methods.
- Example:
  - `RDFLib__pySHACL-285` target method `make_v_result` was patched as a top-level `def make_v_result(...)`.
- Root cause:
  - The scorer rebuilt the full function replacement from dedented `masked_function` artifacts.
  - This lost original file context such as class indentation, decorators, and the exact signature block.
- Source fix in `score_stage1a.py`:
  - Added AST lookup against the original `full_file`.
  - Patch construction now locates the real `FunctionDef` / `AsyncFunctionDef`.
  - Only `node.body` is replaced.
  - Original decorators, signature, class indentation, and surrounding file text are preserved.
  - Syntax validation now parses the actual patched full file rather than a synthetic function string.
  - Harness subprocess failures now raise an error instead of being recorded as model failures.

## Entry 2026-04-24 8

- Current Stage 1a status:
  - Inference pipeline exists and runs on HPC.
  - Scoring pipeline exists and runs locally with Docker when the SWE-rebench fork is installed.
  - Slice coverage diagnostic passes:
    - latest observed median: `1.0`.
  - Harness integration is healthy when the dataset cache and Docker environment are accessible:
    - submitted instances complete,
    - no empty patches,
    - no harness-reported instance errors.
  - Prediction quality is poor:
    - exact match: `0.0`,
    - BLEU-4 diagnostic: near zero,
    - pass@1: `0.0` on the observed run,
    - 7/17 predictions remain syntax-invalid,
    - 10/17 syntactically valid predictions are semantically wrong.
- Representative prediction failures:
  - `Materials-Consortia__optimade-python-tools-2250`:
    - predicted a one-line `ENTRY_TYPE = ...` body instead of the endpoint response logic.
  - `RDFLib__pySHACL-285`:
    - predicted only `RDFNode = type(rdflib.URIRef(...))`.
  - `SpikeInterface__spikeinterface-3829`:
    - predicted a long repeated identifier list.
  - `angr__claripy-603`:
    - generated code for a nested `__call__`, not the expected operation-construction body.
- Interpretation:
  - After the scoring fixes, the main Stage 1a failure is prediction quality, not harness execution.
  - This is consistent with Stage 1a being an untrained SHINE zero-shot baseline on code-completion triples.

## Entry 2026-04-24 9

- Important caveat about latest local outputs:
  - A sandboxed rerun hit a local Hugging Face cache permission error before Docker:

```text
PermissionError: ... /home/seb/.cache/huggingface/datasets/...lock
```

  - That kind of failure should now stop future scoring runs because `score_stage1a.py` raises on nonzero harness exit.
  - If `stage1a_score.summary.json` shows `missing_report_exit_1`, rerun scoring outside the restricted sandbox/local permission issue.

## Commands to Reproduce

### Environment

```bash
cd /home/seb/Developer/Classes/continual-learning/src
uv sync
source .venv/bin/activate
```

Verify the correct SWE-rebench harness fork:

```bash
python - <<'PY'
import inspect
from swebench.harness.test_spec.test_spec import make_test_spec
assert "install_config" in inspect.getsource(make_test_spec)
print("SWE-rebench harness fork installed")
PY
```

### Run SHINE Inference

Run this on the HPC / GPU machine:

```bash
cd /home/seb/Developer/Classes/continual-learning/src/stage-1a/scripts
./run_stage1a.sh --mode full
```

Useful debug modes:

```bash
./run_stage1a.sh --mode tiny
./run_stage1a.sh --mode small
```

Primary output:

```text
src/stage-1a/outputs/stage1a_predictions.jsonl
```

### Score Locally

Run this where Docker and SWE-rebench images are available:

```bash
cd /home/seb/Developer/Classes/continual-learning/src/stage-1a/scripts
./score_stage1a.sh --force
```

Primary outputs:

```text
src/stage-1a/outputs/stage1a_score.summary.json
src/stage-1a/outputs/stage1a_score.per_instance.csv
src/stage-1a/outputs/harness_predictions/
src/stage-1a/outputs/harness_reports/
src/stage-1a/logs/run_evaluation/
```

## Files Added or Modified

- `stage-1a/scripts/run_stage1a.py`
- `stage-1a/scripts/run_stage1a.sh`
- `stage-1a/scripts/score_stage1a.py`
- `stage-1a/scripts/score_stage1a.sh`
- `pyproject.toml`
- `uv.lock`
- `stage-1a/NOTES.md`

## Final Observations

- Stage 1a now produces a valid E-zero baseline artifact.
- The infrastructure work is valuable for Stage 1b:
  - SHINE loading path,
  - code-slice construction,
  - artifact hydration,
  - patch construction,
  - pass@1 harness evaluation,
  - per-instance diagnostics.
- The zero-shot SHINE checkpoint is not enough for these code-body reconstruction tasks.
- Stage 1b should focus on IFT triples with the same `(slice, signature+docstring, body)` contract and should reuse this scorer unchanged.
