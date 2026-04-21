from __future__ import annotations

from pathlib import Path

STAGE1_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE1_DIR.parent
STAGE0_DIR = SRC_DIR / "stage-0"

# Inputs from Stage 0
INSTANCES_JSONL = STAGE0_DIR / "outputs" / "instances.jsonl"
INSTANCES_ARTIFACTS_DIR = STAGE0_DIR / "outputs" / "instances"

# Outputs
OUTPUTS_DIR = STAGE1_DIR / "outputs"
ORACLE_LORA_DIR = OUTPUTS_DIR / "oracle_loras"
COMPLETIONS_DIR = OUTPUTS_DIR / "completions"
EVAL_DIR = OUTPUTS_DIR / "evaluation"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
CAPABILITY_DIR = OUTPUTS_DIR / "capability"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Model defaults
MODEL_ID = "Qwen/Qwen3-8B"
ENABLE_THINKING = False
SEED = 42

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj"]

# Oracle training defaults
ORACLE_CHUNK_SIZE = 512
ORACLE_BATCH_SIZE = 1
ORACLE_GRAD_ACCUM = 8
ORACLE_LR = 2e-4
ORACLE_MIN_EPOCHS = 2
ORACLE_MAX_EPOCHS = 8
ORACLE_PATIENCE = 2
ORACLE_MIN_CHUNK_TOKENS = 64

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
