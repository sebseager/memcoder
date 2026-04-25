#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1B_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(cd "$STAGE1B_DIR/.." && pwd)"

MODE="${MODE:-build}"
MAX_TRIPLES="${MAX_TRIPLES:-500}"
HELDOUT="${HELDOUT:-20}"

cd "$SRC_DIR"
if [[ -d ".venv" ]]; then
  source .venv/bin/activate
fi

case "$MODE" in
  build)
    python "$STAGE1B_DIR/scripts/build_stage1b_dataset.py" \
      --max-triples "$MAX_TRIPLES" \
      --heldout "$HELDOUT" \
      "$@"
    ;;
  train)
    "$STAGE1B_DIR/scripts/train_stage1b.sh" "$@"
    ;;
  infer)
    "$STAGE1B_DIR/scripts/run_stage1b_inference.sh" "$@"
    ;;
  eval)
    python "$STAGE1B_DIR/scripts/evaluate_stage1b_light.py" "$@"
    ;;
  full-eval|full_eval)
    python "$STAGE1B_DIR/scripts/evaluate_stage1b_full.py" "$@"
    ;;
  all)
    python "$STAGE1B_DIR/scripts/build_stage1b_dataset.py" \
      --max-triples "$MAX_TRIPLES" \
      --heldout "$HELDOUT" \
      "$@"
    "$STAGE1B_DIR/scripts/train_stage1b.sh"
    "$STAGE1B_DIR/scripts/run_stage1b_inference.sh" --force
    python "$STAGE1B_DIR/scripts/evaluate_stage1b_light.py"
    ;;
  *)
    echo "Invalid MODE=$MODE. Expected build|train|infer|eval|full-eval|all." >&2
    exit 1
    ;;
esac
