"""
Exp 1 — Oracle Ceiling: Shared configuration.

All locked-in architecture decisions from the README live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXP0_DIR = Path(__file__).resolve().parents[1] / "exp_0"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ORACLE_LORA_DIR = RESULTS_DIR / "oracle_loras"
PATCHES_DIR = RESULTS_DIR / "patches"
EVAL_DIR = RESULTS_DIR / "eval_reports"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
LOSS_CURVES_DIR = RESULTS_DIR / "loss_curves"
SWEBENCH_DIR = RESULTS_DIR / "swebench"
CAPABILITY_DIR = RESULTS_DIR / "capability_checks"

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
ORACLE_LOSS_THRESHOLD = 0.02  # min loss delta to count as improvement
ORACLE_BATCH_SIZE = 1
ORACLE_GRAD_ACCUM = 4

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 8192
GENERATION_RESERVE_TOKENS = 2048  # keep headroom for output during prompt packing
GENERATION_MAX_ATTEMPTS = 3
GENERATION_TOKEN_SCHEDULE_BD = [1024, 2048, 4096]
GENERATION_TOKEN_SCHEDULE_C = [1024, 2048, 8192]
TEMPERATURE = 0.2  # low-temp sampling; reproducible with fixed SEED
TOP_P = 0.95

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
