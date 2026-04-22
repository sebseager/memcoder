#!/usr/bin/env bash
set -euo pipefail

MODE="tiny"
TARGET_FINAL=""
MAX_FILTERED=""
MAX_INSTANCES=""
TOP_K=""
VERIFY_WORKERS=""
VERIFY_BATCH_SIZE=""
VERIFY_TIMEOUT="1800"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --target-final)
      TARGET_FINAL="$2"
      shift 2
      ;;
    --max-filtered)
      MAX_FILTERED="$2"
      shift 2
      ;;
    --max-instances)
      MAX_INSTANCES="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --verify-workers)
      VERIFY_WORKERS="$2"
      shift 2
      ;;
    --verify-batch-size)
      VERIFY_BATCH_SIZE="$2"
      shift 2
      ;;
    --verify-timeout)
      VERIFY_TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(cd "$STAGE_DIR/.." && pwd)"

cd "$STAGE_DIR"

if [[ "$MODE" == "tiny" ]]; then
  : "${MAX_FILTERED:=40}"
  : "${MAX_INSTANCES:=20}"
  : "${TOP_K:=3}"
  : "${TARGET_FINAL:=6}"
  : "${VERIFY_WORKERS:=1}"
  : "${VERIFY_BATCH_SIZE:=4}"
elif [[ "$MODE" == "small" ]]; then
  : "${MAX_FILTERED:=160}"
  : "${MAX_INSTANCES:=120}"
  : "${TOP_K:=4}"
  : "${TARGET_FINAL:=24}"
  : "${VERIFY_WORKERS:=2}"
  : "${VERIFY_BATCH_SIZE:=8}"
elif [[ "$MODE" == "full" ]]; then
  : "${MAX_FILTERED:=0}"
  : "${MAX_INSTANCES:=0}"
  : "${TOP_K:=5}"
  : "${TARGET_FINAL:=100}"
  : "${VERIFY_WORKERS:=4}"
  : "${VERIFY_BATCH_SIZE:=20}"
else
  echo "Invalid mode: $MODE (expected tiny|small|full)"
  exit 1
fi

cd "$SRC_DIR"

bash "$STAGE_DIR/scripts/setup_env.sh"
source .venv/bin/activate

cd "$STAGE_DIR"

FILTER_ARGS=()
if [[ "$MAX_FILTERED" != "0" ]]; then
  FILTER_ARGS+=(--max-candidates "$MAX_FILTERED")
fi

EXTRACT_ARGS=(--top-k-per-instance "$TOP_K")
if [[ "$MAX_INSTANCES" != "0" ]]; then
  EXTRACT_ARGS+=(--max-instances "$MAX_INSTANCES")
fi

python scripts/filter_leaderboard.py "${FILTER_ARGS[@]}"
python scripts/extract_function_candidates.py "${EXTRACT_ARGS[@]}"
python scripts/prepare_candidate_attempts.py --top-k-per-instance "$TOP_K"
python scripts/verify_wipe.py \
  --target-final-instances "$TARGET_FINAL" \
  --max-workers "$VERIFY_WORKERS" \
  --batch-size "$VERIFY_BATCH_SIZE" \
  --timeout-seconds "$VERIFY_TIMEOUT"
python scripts/verify_gold_pass.py
python scripts/finalize_instances.py
python scripts/plot_contamination.py

echo "Stage 0 complete. Outputs in src/stage-0/outputs"
