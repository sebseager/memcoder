#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1A_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(cd "$STAGE1A_DIR/.." && pwd)"

PASS_AT_1_MODE="swebench_harness"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pass-at-1-mode)
      PASS_AT_1_MODE="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$SRC_DIR"
if [[ ! -d ".venv" ]]; then
  echo "Missing src/.venv. Create it first (e.g. with uv venv)."
  exit 1
fi
source .venv/bin/activate

if [[ "$PASS_AT_1_MODE" == "swebench_harness" ]]; then
  if ! python -c "import swebench" >/dev/null 2>&1; then
    echo "Missing swebench in src/.venv."
    echo "Install before scoring: uv sync"
    echo "Or: uv pip install --upgrade 'git+https://github.com/SWE-rebench/SWE-bench-fork.git'"
    exit 1
  fi
  if ! python - <<'PY' >/dev/null 2>&1
import inspect
from swebench.harness.test_spec.test_spec import make_test_spec

src = inspect.getsource(make_test_spec)
raise SystemExit(0 if "install_config" in src else 1)
PY
  then
    echo "Installed swebench is not the SWE-rebench harness fork."
    echo "Fix src/.venv before scoring:"
    echo "  cd $SRC_DIR && uv sync"
    echo "Or:"
    echo "  cd $SRC_DIR && uv pip install --upgrade 'git+https://github.com/SWE-rebench/SWE-bench-fork.git'"
    exit 1
  fi
  if ! command -v docker >/dev/null 2>&1; then
    echo "Missing docker binary (required for harness pass@1)."
    exit 1
  fi
fi

cd "$STAGE1A_DIR"
echo "Scoring Stage 1a predictions locally: pass_at_1_mode=$PASS_AT_1_MODE"
python scripts/score_stage1a.py --pass-at-1-mode "$PASS_AT_1_MODE" "${EXTRA_ARGS[@]}"
echo "Stage 1a scoring complete."
