"""
Exp 1 — Oracle Ceiling: Shared configuration.

All locked-in architecture decisions from the README live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP1_DIR = Path(__file__).resolve().parent
EXP0_DIR = EXP1_DIR.parent / "exp_0"
RESULTS_DIR = EXP1_DIR / "results"
ORACLE_LORA_DIR = RESULTS_DIR / "oracle_loras"
PATCHES_DIR = RESULTS_DIR / "patches"
EVAL_DIR = RESULTS_DIR / "eval_reports"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
LOSS_CURVES_DIR = RESULTS_DIR / "loss_curves"

# exp_0 outputs
SUBSETS_PATH = EXP0_DIR / "results" / "subsets.json"
TOKEN_COUNTS_PATH = EXP0_DIR / "results" / "token_counts.json"
FILE_CACHE_DIR = EXP0_DIR / ".file_cache"

# ---------------------------------------------------------------------------
# Model & LoRA (locked in — do not change after oracle training begins)
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-8B"
SEED = 42
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj"]
LORA_DROPOUT = 0.05
ENABLE_THINKING = False  # hardcoded — never use thinking mode

# ---------------------------------------------------------------------------
# Budget (from exp_0 characterization)
# ---------------------------------------------------------------------------
BUDGET_TOKENS = 11500

# ---------------------------------------------------------------------------
# Oracle LoRA training
# ---------------------------------------------------------------------------
ORACLE_LR = 2e-4
ORACLE_CHUNK_SIZE = 512  # tokens per training chunk
ORACLE_MIN_EPOCHS = 3
ORACLE_MAX_EPOCHS = 20
ORACLE_PATIENCE = 3  # early-stop patience (epochs without improvement)
ORACLE_LOSS_THRESHOLD = 0.01  # min loss delta to count as improvement
ORACLE_BATCH_SIZE = 1
ORACLE_GRAD_ACCUM = 4

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.0  # greedy for reproducibility (SEED controls init only)
TOP_P = 1.0

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = ["A", "B", "C", "D"]
CONDITION_LABELS = {
    "A": "No context, no LoRA (floor)",
    "B": "Truncated context, no LoRA (RAG stand-in)",
    "C": "Full context, no LoRA (ceiling)",
    "D": "Truncated context + oracle LoRA",
}

# ---------------------------------------------------------------------------
# SWE-bench
# ---------------------------------------------------------------------------
SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Lite"
SWEBENCH_SPLIT = "test"
