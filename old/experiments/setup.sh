#!/bin/bash
set -e

uv venv --python 3.10
uv pip install --pre torch \
  --index-url https://download.pytorch.org/whl/nightly/cu128
uv pip install -r pyproject.toml