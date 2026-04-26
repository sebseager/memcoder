#!/usr/bin/env bash

# This script builds flash-attn from source for anyone (un)lucky enough 
# to have a 50-series NVIDIA GPU, for which there aren't any flash-attn 
# wheels available yet.

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export FLASH_ATTN_CUDA_ARCHS="120"
export MAX_JOBS=4

uv pip install flash-attn==2.8.3 --no-build-isolation --no-deps
