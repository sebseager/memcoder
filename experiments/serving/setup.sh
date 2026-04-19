#!/usr/bin/env bash
# One-time setup: install serving-side deps into the experiments/ uv env and
# fetch the Doc-to-LoRA Qwen3-4B hypernetwork checkpoint.
# Re-run is safe — all steps are idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."  # memcoder/experiments/

echo "[1/2] Installing serving deps into the experiments (3.10) env..."
# vllm-metal requires Python >=3.12 and is launched from serve.sh in its own
# isolated env via `uv run --python 3.12 --with vllm-metal`. The 3.10 env only
# needs the client-side deps that make_lora.py / fallback_server.py use.
uv pip install fastapi 'uvicorn[standard]' requests

# Matches the checkpoint layout expected by experiments/doc-to-lora/checkpoint_config.py:
# TRAINED_D2L_DIR = <script dir>/trained_d2l/<run>/<checkpoint-NNN>/pytorch_model.bin
DEST="doc-to-lora/trained_d2l"
echo "[2/2] Downloading SakanaAI/doc-to-lora qwen_4b_d2l into $DEST ..."
mkdir -p "$DEST"
uv run huggingface-cli download SakanaAI/doc-to-lora \
    --local-dir "$DEST" \
    --include "qwen_4b_d2l/*"

echo
echo "Done. Next:"
echo "  bash serving/serve.sh                              # start vLLM"
echo "  uv run python serving/make_lora.py --help          # generate + hot-load"
