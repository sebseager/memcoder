#!/usr/bin/env bash
set -euo pipefail

MODE="pilot"
MODEL_ID=""

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
source .venv/bin/activate
cd "$STAGE_DIR"

if [[ "$MODE" == "pilot" ]]; then
  MAX_INSTANCES=4
  TRAIN_MAX_EPOCHS=2
  TRAIN_MIN_EPOCHS=1
  CAP_MAX_ADAPTERS=4
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
  echo "Invalid mode: $MODE (expected pilot|full)"
  exit 1
fi

echo "Running Stage 1 mode=$MODE model=$MODEL_ID"

PILOT_INSTANCE_IDS_FILE=""
PILOT_FILE_KEYS_FILE=""
if [[ "$MODE" == "pilot" ]]; then
  mkdir -p outputs/logs
  PILOT_INSTANCE_IDS_FILE="outputs/logs/pilot_instance_ids.txt"
  PILOT_FILE_KEYS_FILE="outputs/logs/pilot_file_keys.txt"
  export STAGE0_INSTANCES_JSONL="$SRC_DIR/stage-0/outputs/instances.jsonl"
  export PILOT_INSTANCE_IDS_FILE
  export PILOT_FILE_KEYS_FILE
  export MAX_INSTANCES
  python - <<'PY'
import json
import os
import re
from pathlib import Path

src = Path(os.environ["STAGE0_INSTANCES_JSONL"])
ids_path = Path(os.environ["PILOT_INSTANCE_IDS_FILE"])
keys_path = Path(os.environ["PILOT_FILE_KEYS_FILE"])
max_instances = int(os.environ["MAX_INSTANCES"])

rows = []
with src.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

pilot = rows[:max_instances]

ids_path.write_text("\n".join(r["instance_id"] for r in pilot) + "\n", encoding="utf-8")

seen = set()
keys = []
for r in pilot:
    raw = f"{r['repo']}__{r['file_path']}"
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    if key not in seen:
        seen.add(key)
        keys.append(key)

keys_path.write_text("\n".join(keys) + "\n", encoding="utf-8")
print(f"Pilot instances: {len(pilot)}")
print(f"Pilot unique file keys: {len(keys)}")
PY
fi

TRAIN_ARGS=(--model-id "$MODEL_ID" --max-epochs "$TRAIN_MAX_EPOCHS" --min-epochs "$TRAIN_MIN_EPOCHS")
if [[ -n "$PILOT_FILE_KEYS_FILE" ]]; then
  TRAIN_ARGS+=(--file-keys-file "$PILOT_FILE_KEYS_FILE")
fi
python scripts/train_oracle.py "${TRAIN_ARGS[@]}"

GEN_ARGS=(--model-id "$MODEL_ID")
if [[ -n "$PILOT_INSTANCE_IDS_FILE" ]]; then
  GEN_ARGS+=(--instance-ids-file "$PILOT_INSTANCE_IDS_FILE")
elif [[ -n "$MAX_INSTANCES" ]]; then
  GEN_ARGS+=(--max-instances "$MAX_INSTANCES")
fi

python scripts/generate_completions.py --condition B "${GEN_ARGS[@]}" --force
python scripts/generate_completions.py --condition C "${GEN_ARGS[@]}" --force
python scripts/generate_completions.py --condition D "${GEN_ARGS[@]}" --force

python scripts/evaluate_completions.py --condition all
python scripts/analyze_stage1.py

CAP_ARGS=(--model-id "$MODEL_ID")
if [[ -n "$CAP_MAX_ADAPTERS" ]]; then
  CAP_ARGS+=(--max-adapters "$CAP_MAX_ADAPTERS")
fi
python scripts/capability_interference.py "${CAP_ARGS[@]}"

echo "Stage 1 run complete. Outputs in stage-1/outputs"
