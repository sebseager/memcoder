#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1B_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(cd "$STAGE1B_DIR/.." && pwd)"

NAME="${NAME:-stage1b_code_ift_qwen3_8b}"
INSTANCES_JSONL="${INSTANCES_JSONL:-$STAGE1B_DIR/outputs/heldout_instances.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$STAGE1B_DIR/outputs/stage1b_predictions.jsonl}"
OUTPUT_META_JSON="${OUTPUT_META_JSON:-$STAGE1B_DIR/outputs/stage1b_run_meta.json}"
MODEL_PATH="${MODEL_PATH:-$SRC_DIR/stage-1a/models/Qwen3-8B}"
SHINE_DIR="${SHINE_DIR:-$(cd "$SRC_DIR/.." && pwd)/vendor/SHINE}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

if [[ -z "$CHECKPOINT_DIR" ]]; then
  TRAIN_ROOT="$STAGE1B_DIR/checkpoints/$NAME/train"
  if [[ ! -d "$TRAIN_ROOT" ]]; then
    TRAIN_ROOT="$STAGE1B_DIR/shine_work/checkpoints/$NAME/train"
  fi
  if [[ ! -d "$TRAIN_ROOT" ]]; then
    echo "No Stage 1b train checkpoint root found for NAME=$NAME" >&2
    exit 1
  fi
  shopt -s nullglob
  candidates=("$TRAIN_ROOT"/checkpoint-epoch-* "$TRAIN_ROOT"/checkpoint-*)
  shopt -u nullglob
  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "No checkpoint-* directories found under $TRAIN_ROOT" >&2
    exit 1
  fi
  IFS=$'\n' candidates=($(printf '%s\n' "${candidates[@]}" | sort -V))
  unset IFS
  CHECKPOINT_DIR="${candidates[-1]}"
fi

cd "$SRC_DIR/stage-1a"
echo "Running Stage 1b inference with checkpoint: $CHECKPOINT_DIR"
python scripts/run_stage1a.py \
  --instances-jsonl "$INSTANCES_JSONL" \
  --output-jsonl "$OUTPUT_JSONL" \
  --output-meta-json "$OUTPUT_META_JSON" \
  --base-model-path "$MODEL_PATH" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --vendor-shine-dir "$SHINE_DIR" \
  --max-instances "${MAX_INSTANCES:-20}" \
  --context-max-tokens "${CONTEXT_MAX_TOKENS:-1024}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --max-conversation-length "${MAX_CONVERSATION_LENGTH:-4096}" \
  "$@"
