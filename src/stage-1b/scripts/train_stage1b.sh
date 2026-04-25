#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1B_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(cd "$STAGE1B_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/.." && pwd)"
SHINE_DIR="${SHINE_DIR:-$REPO_ROOT/vendor/SHINE}"

NAME="${NAME:-stage1b_code_ift_qwen3_8b}"
BASE_NAME="${BASE_NAME:-8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-18900}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-250}"
EVAL_STEPS="${EVAL_STEPS:-250}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
CONTEXT_MAX_LEN="${CONTEXT_MAX_LEN:-1024}"
CONVERSATION_MAX_LEN="${CONVERSATION_MAX_LEN:-4096}"
LORA_R="${LORA_R:-8}"
METALORA_R="${METALORA_R:-128}"
METANET_LAYERS="${METANET_LAYERS:-4}"
MODEL_PATH="${MODEL_PATH:-$SRC_DIR/stage-1a/models/Qwen3-8B}"
DATA_JSON="${DATA_JSON:-$STAGE1B_DIR/data/ift_c1qa_code_train.json}"
WORK_DIR="${WORK_DIR:-$STAGE1B_DIR/shine_work}"
PRETRAIN_IFTPWC="${PRETRAIN_IFTPWC:-$SRC_DIR/stage-1a/checkpoints/$BASE_NAME/iftpwc}"

if [[ ! -d "$SHINE_DIR" ]]; then
  echo "Missing SHINE_DIR: $SHINE_DIR" >&2
  exit 1
fi
if [[ ! -f "$DATA_JSON" ]]; then
  echo "Missing Stage 1b SHINE data JSON: $DATA_JSON" >&2
  echo "Run scripts/build_stage1b_dataset.py first." >&2
  exit 1
fi
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing model path: $MODEL_PATH" >&2
  echo "Run the Stage 1a checkpoint/model download first." >&2
  exit 1
fi
if [[ ! -d "$PRETRAIN_IFTPWC" ]]; then
  echo "Missing Stage 1a SHINE iftpwc checkpoint dir: $PRETRAIN_IFTPWC" >&2
  echo "This script points to it instead of copying it." >&2
  exit 1
fi

mkdir -p "$WORK_DIR" "$WORK_DIR/data" "$WORK_DIR/checkpoints/$NAME" "$WORK_DIR/tensorboard" "$STAGE1B_DIR/checkpoints"

# Build a writable SHINE overlay. Source files are symlinked; data and checkpoints
# stay inside Stage 1b so reruns never edit vendor/SHINE or prior stages.
for entry in "$SHINE_DIR"/*; do
  base="$(basename "$entry")"
  case "$base" in
    data|checkpoints|tensorboard|tmp_metatrain_*.txt|meta_train_parallel.py)
      ;;
    *)
      ln -sfn "$entry" "$WORK_DIR/$base"
      ;;
  esac
done

cp "$SHINE_DIR/meta_train_parallel.py" "$WORK_DIR/meta_train_parallel.py"
python - "$WORK_DIR/meta_train_parallel.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = '''    elif cfg.data.source == "ift-c1qa":
        data_path = os.path.join("data", "ift_c1qa.json")
        train_ds = IFTC1QADataset(data_path, use_exceed=False, max_context_len=cfg.data.context_max_length, max_conversation_len=cfg.data.conversation_max_length)
        val_dataset = load_dataset(os.path.join("data", "squad"), split="validation")
        val_dataset = val_dataset.shuffle(seed=42).select(range(1000))
        val_ds = GroupedSquadDataset(val_dataset, tokenizer, 512, name="Validation", sep="\\n\\n")
        train_collator = IFTCollator(tokenizer, cfg.data.context_max_length, cfg.data.conversation_max_length, cfg=cfg)
        val_collator = SquadCollator(tokenizer=tokenizer, conversation_max_length=cfg.data.conversation_max_length, context_max_length=cfg.data.context_max_length, metatrain=True, cfg=cfg)
'''
new = '''    elif cfg.data.source == "ift-c1qa":
        data_path = os.path.join("data", "ift_c1qa.json")
        train_ds = IFTC1QADataset(data_path, use_exceed=False, max_context_len=cfg.data.context_max_length, max_conversation_len=cfg.data.conversation_max_length)
        val_ds = train_ds
        train_collator = IFTCollator(tokenizer, cfg.data.context_max_length, cfg.data.conversation_max_length, cfg=cfg)
        val_collator = IFTCollator(tokenizer, cfg.data.context_max_length, cfg.data.conversation_max_length, cfg=cfg)
'''
if old not in text:
    raise SystemExit("Could not patch SHINE ift-c1qa validation branch")
path.write_text(text.replace(old, new), encoding="utf-8")
PY

ln -sfn "$DATA_JSON" "$WORK_DIR/data/ift_c1qa.json"
ln -sfn "$PRETRAIN_IFTPWC" "$WORK_DIR/checkpoints/$NAME/iftpwc"
ln -sfn "$WORK_DIR/checkpoints/$NAME" "$STAGE1B_DIR/checkpoints/$NAME"

if [[ -d "$SHINE_DIR/data/squad" ]]; then
  ln -sfn "$SHINE_DIR/data/squad" "$WORK_DIR/data/squad"
fi

while true; do
  if ! nc -z 127.0.0.1 "$MASTER_PORT"; then
    break
  fi
  MASTER_PORT=$((MASTER_PORT + 1))
done

export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"

cd "$WORK_DIR"
echo "Training Stage 1b SHINE IFT: name=$NAME gpus=$NUM_GPUS data=$DATA_JSON"
echo "Outputs: $WORK_DIR/checkpoints/$NAME/train (also linked from $STAGE1B_DIR/checkpoints/$NAME)"

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port="$MASTER_PORT" \
  meta_train_parallel.py \
  --config-name Qwen3-8B \
  name="$NAME" \
  mode=train \
  paths.model_path="$MODEL_PATH" \
  model.tokenizer_from="$MODEL_PATH" \
  model.model_from="$MODEL_PATH" \
  data.source=ift-c1qa \
  data.context_max_length="$CONTEXT_MAX_LEN" \
  data.conversation_max_length="$CONVERSATION_MAX_LEN" \
  data.train_batch_size=1 \
  data.eval_batch_size=1 \
  data.num_workers="${NUM_WORKERS:-2}" \
  run.gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
  run.use_gradient_checkpoint="${USE_GRADIENT_CHECKPOINT:-False}" \
  run.use_amp="${USE_AMP:-false}" \
  optim.num_epochs="$NUM_EPOCHS" \
  optim.learning_rate="$LEARNING_RATE" \
  optim.warmup_steps="$WARMUP_STEPS" \
  eval.eval_steps="$EVAL_STEPS" \
  save.save_steps="$SAVE_STEPS" \
  resume_global_step="${RESUME_GLOBAL_STEP:--1}" \
  metanetwork.type=transformer \
  metanetwork.transformer_cfg.num_layers="$METANET_LAYERS" \
  metanetwork.method=rl \
  model.lora_r="$LORA_R" \
  model.metalora_r="$METALORA_R" \
  model.ift_additional_metalora_r=-1
