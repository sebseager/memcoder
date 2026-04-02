#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$SCRIPT_DIR"

[ -d .venv ] || uv venv --python 3.12

export CUDA_HOME="/usr/local/cuda-12.8"

# Blackwell-compatible nightly
uv pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

uv pip install peft accelerate einops jaxtyping opt-einsum datasets
uv pip install "transformers==4.51.3"
uv pip install -e "$REPO_ROOT/vendor/doc-to-lora" --no-deps

uv pip install ninja packaging
uv pip install flash-attn --no-build-isolation

uv run python demo.py