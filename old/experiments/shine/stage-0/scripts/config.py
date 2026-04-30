from __future__ import annotations

from pathlib import Path

STAGE0_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE0_DIR.parent

OUTPUTS_DIR = STAGE0_DIR / "outputs"
CACHE_DIR = STAGE0_DIR / "cache"
REPOS_CACHE_DIR = CACHE_DIR / "repos"
VERIFY_TMP_DIR = CACHE_DIR / "verify_tmp"

DATASET_NAME = "nebius/SWE-rebench-leaderboard"
CUTOFF_DATE = "2025-01-01"
TOKENIZER_ID = "Qwen/Qwen3-8B"

MIN_FILE_TOKENS = 2048
MIN_BODY_LINES = 10
MAX_BODY_LINES = 80
TOP_CANDIDATES_PER_INSTANCE = 5

DEFAULT_TARGET_CANDIDATES = 400
DEFAULT_MIN_TARGET_CANDIDATES = 300
DEFAULT_TARGET_FINAL_INSTANCES = 100

DEFAULT_VERIFY_TIMEOUT_SECONDS = 1800
DEFAULT_VERIFY_MAX_WORKERS = 2
DEFAULT_DOCKER_NAMESPACE = "swerebench"


def ensure_stage0_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VERIFY_TMP_DIR.mkdir(parents=True, exist_ok=True)
