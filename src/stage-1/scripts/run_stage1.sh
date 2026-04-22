#!/usr/bin/env bash
set -euo pipefail

MODE="tiny"
MODEL_ID=""
SEED=42
TRUNC_BUDGET=2048
MAX_NEW_TOKENS=1024
TEMPERATURE=0.0
TOP_P=1.0
BEHAVIORAL_PROBES=5
BEHAVIORAL_EPOCHS=1
BEHAVIORAL_LR_MULT=0.5
ORACLE_CHUNK_SIZE=3072

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --model-id)
      MODEL_ID="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --trunc-budget)
      TRUNC_BUDGET="$2"
      shift 2
      ;;
    --max-new-tokens)
      MAX_NEW_TOKENS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --behavioral-probes)
      BEHAVIORAL_PROBES="$2"
      shift 2
      ;;
    --behavioral-epochs)
      BEHAVIORAL_EPOCHS="$2"
      shift 2
      ;;
    --behavioral-lr-mult)
      BEHAVIORAL_LR_MULT="$2"
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

cd "$SRC_DIR"

if [[ ! -d ".venv" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Missing uv binary; install uv before running Stage 1"
    exit 1
  fi
  echo "Creating src/.venv with uv"
  uv venv .venv
fi

source .venv/bin/activate
cd "$STAGE_DIR"

if [[ "$MODE" == "tiny" ]]; then
  MAX_INSTANCES=4
  TRAIN_MAX_EPOCHS=2
  TRAIN_MIN_EPOCHS=1
  CAP_MAX_ADAPTERS=4
  if [[ -z "$MODEL_ID" ]]; then
    MODEL_ID="Qwen/Qwen2.5-Coder-1.5B-Instruct"
  fi
elif [[ "$MODE" == "small" ]]; then
  MAX_INSTANCES=12
  TRAIN_MAX_EPOCHS=4
  TRAIN_MIN_EPOCHS=2
  CAP_MAX_ADAPTERS=12
  if [[ -z "$MODEL_ID" ]]; then
    MODEL_ID="Qwen/Qwen2.5-Coder-1.5B-Instruct"
  fi
elif [[ "$MODE" == "full" ]]; then
  MAX_INSTANCES=""
  TRAIN_MAX_EPOCHS=8
  TRAIN_MIN_EPOCHS=2
  CAP_MAX_ADAPTERS=""
  if [[ -z "$MODEL_ID" ]]; then
    MODEL_ID="Qwen/Qwen3-8B"
  fi
else
  echo "Invalid mode: $MODE (expected tiny|small|full)"
  exit 1
fi

echo "Running Stage 1 mode=$MODE model=$MODEL_ID"
export STAGE1_MODE="$MODE"

MODEL_SLUG="$(python - <<'PY' "$MODE" "$MODEL_ID"
import re
import sys

mode = sys.argv[1]
model_id = sys.argv[2]
mode_slug = re.sub(r"[^a-z0-9._-]+", "-", mode.strip().lower()).strip("-")
model_slug = re.sub(r"[^a-z0-9._-]+", "-", model_id.strip().lower()).strip("-")
model_slug = model_slug or "model"
print(f"{mode_slug}.{model_slug}" if mode_slug else model_slug)
PY
)"
MODEL_OUTPUT_DIR="outputs/${MODEL_SLUG}"
MODEL_LOGS_DIR="${MODEL_OUTPUT_DIR}/logs"
mkdir -p "$MODEL_LOGS_DIR"

python scripts/init_run_config.py \
  --mode "$MODE" \
  --model-id "$MODEL_ID" \
  --seed "$SEED" \
  --trunc-budget "$TRUNC_BUDGET" \
  --chunk-size "$ORACLE_CHUNK_SIZE" \
  --behavioral-probes "$BEHAVIORAL_PROBES" \
  --behavioral-epochs "$BEHAVIORAL_EPOCHS" \
  --behavioral-lr-mult "$BEHAVIORAL_LR_MULT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P"

SUBSET_INSTANCE_IDS_FILE=""
SUBSET_FILE_KEYS_FILE=""
if [[ -n "$MAX_INSTANCES" ]]; then
  SUBSET_INSTANCE_IDS_FILE="${MODEL_LOGS_DIR}/${MODE}_instance_ids.txt"
  SUBSET_FILE_KEYS_FILE="${MODEL_LOGS_DIR}/${MODE}_file_keys.txt"
  export STAGE0_INSTANCES_JSONL="$SRC_DIR/stage-0/outputs/instances.jsonl"
  export SUBSET_INSTANCE_IDS_FILE
  export SUBSET_FILE_KEYS_FILE
  export MAX_INSTANCES
  python - <<'PY'
import json
import os
import re
from pathlib import Path

src = Path(os.environ["STAGE0_INSTANCES_JSONL"])
ids_path = Path(os.environ["SUBSET_INSTANCE_IDS_FILE"])
keys_path = Path(os.environ["SUBSET_FILE_KEYS_FILE"])
max_instances = int(os.environ["MAX_INSTANCES"])

rows = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

subset = rows[:max_instances]

ids_path.write_text("\n".join(r["instance_id"] for r in subset) + "\n", encoding="utf-8")

seen = set()
keys = []
for r in subset:
    raw = f"{r['repo']}__{r['file_path']}"
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    if key not in seen:
        seen.add(key)
        keys.append(key)

keys_path.write_text("\n".join(keys) + "\n", encoding="utf-8")
print(f"Subset instances: {len(subset)}")
print(f"Subset unique file keys: {len(keys)}")
PY
fi

TRAIN_ARGS=(
  --model-id "$MODEL_ID"
  --seed "$SEED"
  --trunc-budget "$TRUNC_BUDGET"
  --chunk-size "$ORACLE_CHUNK_SIZE"
  --max-epochs "$TRAIN_MAX_EPOCHS"
  --min-epochs "$TRAIN_MIN_EPOCHS"
  --behavioral-probes "$BEHAVIORAL_PROBES"
  --behavioral-epochs "$BEHAVIORAL_EPOCHS"
  --behavioral-lr-mult "$BEHAVIORAL_LR_MULT"
)
if [[ -n "$SUBSET_FILE_KEYS_FILE" ]]; then
  TRAIN_ARGS+=(--file-keys-file "$SUBSET_FILE_KEYS_FILE")
fi
python scripts/train_oracle.py "${TRAIN_ARGS[@]}"

GEN_ARGS=(
  --model-id "$MODEL_ID"
  --seed "$SEED"
  --trunc-budget "$TRUNC_BUDGET"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
)
if [[ -n "$SUBSET_INSTANCE_IDS_FILE" ]]; then
  GEN_ARGS+=(--instance-ids-file "$SUBSET_INSTANCE_IDS_FILE")
elif [[ -n "$MAX_INSTANCES" ]]; then
  GEN_ARGS+=(--max-instances "$MAX_INSTANCES")
fi

python scripts/generate_completions.py --condition B "${GEN_ARGS[@]}" --force
python scripts/generate_completions.py --condition C "${GEN_ARGS[@]}" --force
python scripts/generate_completions.py --condition D "${GEN_ARGS[@]}" --force

python scripts/identifier_overlap.py --model-id "$MODEL_ID" --trunc-budget "$TRUNC_BUDGET" --condition D

python "$SRC_DIR/stage-0/scripts/sync_prebuilt_images.py" --operation copy --quiet
python scripts/evaluate_completions.py --condition all --model-id "$MODEL_ID" --pass-at-1-mode docker_pytest
python scripts/analyze_stage1.py --model-id "$MODEL_ID" --seed "$SEED"

CAP_ARGS=(--model-id "$MODEL_ID" --seed "$SEED")
if [[ -n "$CAP_MAX_ADAPTERS" ]]; then
  CAP_ARGS+=(--max-adapters "$CAP_MAX_ADAPTERS")
fi
if [[ -n "$SUBSET_FILE_KEYS_FILE" ]]; then
  CAP_ARGS+=(--file-keys-file "$SUBSET_FILE_KEYS_FILE")
fi
python scripts/capability_interference.py "${CAP_ARGS[@]}"

echo "Stage 1 run complete. Outputs in stage-1/${MODEL_OUTPUT_DIR}"
