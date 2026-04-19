"""Checkpoint selection helpers for lora-recall experiments.

Selection order:
  1) D2L_CHECKPOINT_PATH (file path or checkpoint directory)
  2) D2L_CHECKPOINT_RUN (+ optional D2L_CHECKPOINT_STEP)
  3) Default gemma_demo checkpoint
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINED_D2L_DIR = SCRIPT_DIR.parent / "doc-to-lora" / "trained_d2l"
DEFAULT_CHECKPOINT_RUN = "gemma_demo"
DEFAULT_CHECKPOINT_STEP = "checkpoint-80000"


def _normalize_checkpoint_step(step: str) -> str:
    step = step.strip()
    if not step:
        return DEFAULT_CHECKPOINT_STEP
    if step.startswith("checkpoint-"):
        return step
    if step.isdigit():
        return f"checkpoint-{step}"
    return step


def _resolve_from_run(run_name: str, checkpoint_step: str) -> Path:
    step = _normalize_checkpoint_step(checkpoint_step)
    return TRAINED_D2L_DIR / run_name / step / "pytorch_model.bin"


def resolve_checkpoint_path() -> Path:
    raw_path = os.environ.get("D2L_CHECKPOINT_PATH", "").strip()
    if raw_path:
        checkpoint_path = Path(raw_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "pytorch_model.bin"
        return checkpoint_path

    run_name = os.environ.get("D2L_CHECKPOINT_RUN", "").strip()
    if run_name:
        checkpoint_step = os.environ.get("D2L_CHECKPOINT_STEP", DEFAULT_CHECKPOINT_STEP)
        return _resolve_from_run(run_name, checkpoint_step)

    return _resolve_from_run(DEFAULT_CHECKPOINT_RUN, DEFAULT_CHECKPOINT_STEP)


def read_max_ctx_chunk_len(checkpoint_path: str | Path) -> int:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent.parent
    args_path = checkpoint_dir / "args.yaml"
    if args_path.exists():
        with args_path.open(encoding="utf-8") as f:
            args = yaml.safe_load(f) or {}
        return int(args.get("max_ctx_chunk_len", -1))
    return -1
