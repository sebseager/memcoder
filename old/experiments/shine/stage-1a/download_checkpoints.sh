#!/bin/bash
set -e

# Confirm with the user
read -p "This will download ~25GB in your PWD. Continue? (y/n): " confirm
if [ "$confirm" != "y" ]; then
  echo "Aborting."
  exit 1
fi

# Dependencies
uv pip install huggingface==0.0.1 modelscope==1.31.0 transformers==4.57.1 \
  datasets==4.4.1 scikit-learn==1.7.2 hydra-core==1.3.2 tensorboard==2.20.0 \
  openai==2.6.1 rouge==1.0.1 seaborn==0.13.2 matplotlib==3.10.7 \
  multiprocess==0.70.16

BASE="checkpoints/8gpu_8lora_128metalora_lr5e-5_grouppretrain_1150"

# Backbone
mkdir -p models
modelscope download --model Qwen/Qwen3-8B --local_dir models/Qwen3-8B

# Hypernetwork checkpoints
mkdir -p "$BASE/pretrain"
huggingface-cli download Yewei-Liu/SHINE-Pretrain \
  --local-dir "$BASE/pretrain/checkpoint-epoch-1"

mkdir -p "$BASE/iftpwc"
huggingface-cli download Yewei-Liu/SHINE-ift_mqa \
  --local-dir "$BASE/iftpwc/checkpoint-epoch-2"

mkdir -p "$BASE/train"
huggingface-cli download Yewei-Liu/SHINE-ift_mqa_1qa \
  --local-dir "$BASE/train/checkpoint-epoch-1"

echo "All done."