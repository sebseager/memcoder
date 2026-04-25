#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1A_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(cd "$STAGE1A_DIR/.." && pwd)"

MODE="full"
MAX_INSTANCES=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --max-instances)
      MAX_INSTANCES="$2"
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

if [[ -z "$MAX_INSTANCES" ]]; then
  if [[ "$MODE" == "tiny" ]]; then
    MAX_INSTANCES=4
  elif [[ "$MODE" == "small" ]]; then
    MAX_INSTANCES=10
  elif [[ "$MODE" == "full" ]]; then
    MAX_INSTANCES=20
  else
    echo "Invalid mode: $MODE (expected tiny|small|full)"
    exit 1
  fi
fi

cd "$SRC_DIR"
if [[ ! -d ".venv" ]]; then
  echo "Missing src/.venv. Create it first (e.g. with uv venv)."
  exit 1
fi
source .venv/bin/activate

cd "$STAGE1A_DIR"
echo "Running Stage 1a inference (HPC-friendly, no docker harness): mode=$MODE max_instances=$MAX_INSTANCES"
python scripts/run_stage1a.py --max-instances "$MAX_INSTANCES" "${EXTRA_ARGS[@]}"
echo "Stage 1a inference complete."
