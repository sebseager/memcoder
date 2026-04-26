#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export D2L_CHECKPOINT_RUN="qwen_4b_d2l"
export D2L_CHECKPOINT_STEP="checkpoint-20000"

uv run python run_all.py "$@"
