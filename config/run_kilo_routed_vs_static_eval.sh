#!/usr/bin/env bash
set -euo pipefail

# Run from repo root.
cd "$(dirname "$0")/.."

export PYTHONPATH="${PWD}/vendor/SHINE:${PWD}:${PYTHONPATH:-}"

uv run python scripts/run_routed_lora_eval.py \
  --config config/shine_eval_demo.yaml \
  --shine-root vendor/SHINE \
  --model-from Qwen/Qwen3-8B \
  --tokenizer-from Qwen/Qwen3-8B \
  --skip-generate-loras \
  --lora-dir artifacts/antirez__kilo/easy/loras \
  --output artifacts/antirez__kilo/easy/lora_routed_vs_static_results.jsonl \
  --routing-results artifacts/antirez__kilo/easy/routing_results.qwen3.overview_purpose_1.top1.jsonl \
  --routing-results artifacts/antirez__kilo/easy/routing_results.qwen3.raw_terminal_input_1.top1.jsonl \
  --routing-results artifacts/antirez__kilo/easy/routing_results.qwen3.rows_editing_persistence_1.top1.jsonl \
  --include-routed-top1 \
  --include-static-loras \
  --qwen-cuda 0
