from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

STAGE1_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE1_DIR.parent
STAGE0_DIR = SRC_DIR / "stage-0"

# Inputs from Stage 0
INSTANCES_JSONL = STAGE0_DIR / "outputs" / "instances.jsonl"
INSTANCES_ARTIFACTS_DIR = STAGE0_DIR / "outputs" / "instances"
PREBUILT_IMAGE_MANIFEST = (
    STAGE0_DIR / "outputs" / "prebuilt_images" / "image_manifest.json"
)

# Outputs
OUTPUTS_ROOT_DIR = STAGE1_DIR / "outputs"
RUN_CONFIG_FILENAME = "run_config.json"

# Model defaults
MODEL_ID = "Qwen/Qwen3-8B"
ENABLE_THINKING = False
SEED = 42

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Bridge Decision 1: FFN-only adapters, following DyPRAG's knowledge-localization rationale.
LORA_TARGET_MODULES = ["up_proj", "down_proj"]
LAYER_TARGETING_DECISION = "ffn_only"
LAYER_TARGETING_JUSTIFICATION = (
    "Use FFN-only LoRA targets for file-knowledge injection; attention layers are left "
    "untouched to reduce behavioral side effects."
)
LAYER_TARGETING_REFERENCE = (
    "DyPRAG prior uses FFN-focused adaptation for knowledge content storage."
)

# Oracle training defaults
ORACLE_CHUNK_SIZE = 3072
ORACLE_BATCH_SIZE = 1
ORACLE_GRAD_ACCUM = 8
ORACLE_LR = 2e-4
ORACLE_MIN_EPOCHS = 2
ORACLE_MAX_EPOCHS = 8
ORACLE_PATIENCE = 2
ORACLE_MIN_CHUNK_TOKENS = 64
BEHAVIORAL_PROBES = 5
BEHAVIORAL_EPOCHS = 1
BEHAVIORAL_LR_MULT = 0.5

# Generation defaults
TRUNCATION_BUDGET_TOKENS = 2048
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.0
TOP_P = 1.0

# Conditions for Stage 1
CONDITIONS = ["B", "C", "D"]
CONDITION_LABELS = {
    "B": "Truncated context, no LoRA",
    "C": "Full context, no LoRA",
    "D": "Truncated context + oracle LoRA",
}

# Capability interference probes
CAPABILITY_DROP_THRESHOLD_PCT = 15.0

# Evaluation bootstrap
BOOTSTRAP_SAMPLES = 1000

# Gap stratification thresholds (Bridge Step 1d)
LOW_GAP_BLEU_THRESHOLD = 0.05
HIGH_GAP_BLEU_THRESHOLD = 0.20


@dataclass(frozen=True)
class Stage1Paths:
    root: Path
    oracle_lora: Path
    completions: Path
    evaluation: Path
    analysis: Path
    capability: Path
    plots: Path
    logs: Path
    run_config: Path


def model_id_to_slug(model_id: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_id.strip().lower()).strip("-")
    return slug or "model"


def mode_to_slug(mode: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", mode.strip().lower()).strip("-")
    return slug


def build_output_slug(model_id: str, mode: str | None = None) -> str:
    model_slug = model_id_to_slug(model_id)
    if mode is None:
        mode = os.environ.get("STAGE1_MODE", "")
    mode_slug = mode_to_slug(mode)
    if mode_slug:
        return f"{mode_slug}.{model_slug}"
    return model_slug


def get_stage1_paths(model_id: str = MODEL_ID, mode: str | None = None) -> Stage1Paths:
    model_root = OUTPUTS_ROOT_DIR / build_output_slug(model_id, mode=mode)
    return Stage1Paths(
        root=model_root,
        oracle_lora=model_root / "oracle_loras",
        completions=model_root / "completions",
        evaluation=model_root / "evaluation",
        analysis=model_root / "analysis",
        capability=model_root / "capability",
        plots=model_root / "plots",
        logs=model_root / "logs",
        run_config=model_root / RUN_CONFIG_FILENAME,
    )


# Backward-compatible default aliases.
DEFAULT_PATHS = get_stage1_paths(MODEL_ID)
ORACLE_LORA_DIR = DEFAULT_PATHS.oracle_lora
COMPLETIONS_DIR = DEFAULT_PATHS.completions
EVAL_DIR = DEFAULT_PATHS.evaluation
ANALYSIS_DIR = DEFAULT_PATHS.analysis
CAPABILITY_DIR = DEFAULT_PATHS.capability
PLOTS_DIR = DEFAULT_PATHS.plots
LOGS_DIR = DEFAULT_PATHS.logs
