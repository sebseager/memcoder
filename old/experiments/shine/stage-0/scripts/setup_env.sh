#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$(cd "$STAGE_DIR/.." && pwd)"

cd "$SRC_DIR"

if [[ ! -d ".venv" ]]; then
  if ! command -v uv >/dev/null 2>&1; then
    echo "Missing uv binary; install uv before running Stage 0"
    exit 1
  fi
  echo "Creating src/.venv with uv"
  uv venv .venv
fi

source .venv/bin/activate

uv pip install --upgrade \
  datasets \
  pandas \
  matplotlib \
  transformers

# Pin to SWE-rebench harness fork for install_config + namespace behavior.
uv pip install --upgrade "git+https://github.com/SWE-rebench/SWE-bench-fork.git"

# tree-sitter-languages 1.10.2 requires tree-sitter<0.22.
uv pip install --upgrade "tree-sitter<0.22" tree-sitter-languages==1.10.2

echo "Stage 0 environment ready in src/.venv"
