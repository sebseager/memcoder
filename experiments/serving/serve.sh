#!/usr/bin/env bash
# Start vLLM serving Qwen3-4B with runtime LoRA hot-loading enabled.
# Adapters produced by make_lora.py can be attached via POST /v1/load_lora_adapter
# without restarting the server.
set -euo pipefail

cd "$(dirname "$0")/.."  # memcoder/experiments/

# Required to expose load_lora_adapter / unload_lora_adapter endpoints.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-8000}"
MAX_LORAS="${MAX_LORAS:-4}"
MAX_LORA_RANK="${MAX_LORA_RANK:-8}"    # per the Doc-to-LoRA paper
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# vllm-metal requires Python >=3.12; the experiments env is 3.10. Launch vLLM
# in an isolated env uv manages for us so the 3.10 env stays intact.
uv run --isolated --python 3.12 --with vllm-metal -- \
    vllm serve "$MODEL" \
    --enable-lora \
    --max-loras "$MAX_LORAS" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT"
