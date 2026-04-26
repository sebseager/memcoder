"""
Exp 1 — Oracle Ceiling: Shared helpers for loading data and model.

Keeps train_oracle.py and generate_patches.py DRY.
"""

import json
import re
from functools import lru_cache
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
from memcoder.old.experiments.dyprag.exp_1.config import (
    BUDGET_TOKENS,
    FILE_CACHE_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MODEL_ID,
    SEED,
    SUBSETS_PATH,
    SWEBENCH_DATASET,
    SWEBENCH_SPLIT,
    TOKEN_COUNTS_PATH,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_subsets() -> dict:
    """Load the exp_0 subsets.json."""
    with open(SUBSETS_PATH) as f:
        return json.load(f)


def load_token_counts() -> list[dict]:
    """Load per-instance token data from exp_0."""
    with open(TOKEN_COUNTS_PATH) as f:
        return json.load(f)


def get_instance_data(instance_id: str, token_counts: list[dict]) -> dict:
    """Look up a single instance by ID."""
    for rec in token_counts:
        if rec["instance_id"] == instance_id:
            return rec
    raise KeyError(f"Instance {instance_id} not found in token_counts")


def load_swebench_instance(instance_id: str) -> dict:
    """Load a single SWE-Bench Lite instance from HuggingFace."""
    from datasets import load_dataset

    ds = load_dataset(SWEBENCH_DATASET, split=SWEBENCH_SPLIT)
    for row in ds:
        if row["instance_id"] == instance_id:
            return row
    raise KeyError(f"Instance {instance_id} not in SWE-bench Lite test split")


@lru_cache(maxsize=1)
def load_swebench_dataset() -> dict:
    """Load full SWE-Bench Lite test split as {instance_id: row}."""
    from datasets import load_dataset

    ds = load_dataset(SWEBENCH_DATASET, split=SWEBENCH_SPLIT)
    return {row["instance_id"]: row for row in ds}


def extract_patched_files(patch: str) -> list[str]:
    """Extract file paths from a unified diff patch."""
    return re.findall(r"diff --git a/(.*?) b/", patch)


def fetch_file_content(repo: str, commit: str, filepath: str) -> str | None:
    """Fetch a file from GitHub at a specific commit, with local caching."""
    cache_key = f"{repo}/{commit}/{filepath}".replace("/", "__")
    cache_path = FILE_CACHE_DIR / cache_key
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="replace")

    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{filepath}"
    req = Request(url, headers={"User-Agent": "dyprag-exp1"})
    try:
        with urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="replace")
        FILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
        return content
    except HTTPError as e:
        print(f"  WARN: HTTP {e.code} fetching {url}")
        return None
    except Exception as e:
        print(f"  WARN: error fetching {url}: {e}")
        return None


def get_file_content_for_instance(
    instance_id: str, token_counts: list[dict], swebench_data: dict
) -> tuple[str, str]:
    """Return (file_path, file_content) for an instance.

    All SWE-Bench Lite test instances have exactly one patched file (per exp_0).
    """
    rec = get_instance_data(instance_id, token_counts)
    row = swebench_data[instance_id]
    fpath = rec["files"][0]["path"]
    content = fetch_file_content(row["repo"], row["base_commit"], fpath)
    if content is None:
        raise RuntimeError(f"Could not fetch file for {instance_id}: {fpath}")
    return fpath, content


def truncate_to_budget(text: str, tokenizer, budget: int = BUDGET_TOKENS) -> str:
    """Truncate text to fit within token budget, preserving complete lines."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= budget:
        return text
    # Decode the truncated tokens back to text
    truncated = tokenizer.decode(tokens[:budget], skip_special_tokens=True)
    # Try to end at a complete line
    last_newline = truncated.rfind("\n")
    if last_newline > len(truncated) * 0.8:  # only if we don't lose too much
        truncated = truncated[: last_newline + 1]
    return truncated + "\n# ... [truncated] ..."


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_base_model(device_map="auto"):
    """Load Qwen3-8B in 4-bit quantization. Returns (model, tokenizer)."""
    torch.manual_seed(SEED)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    return model, tokenizer


def prepare_model_for_lora_training(model):
    """Apply PEFT's recommended preparation for k-bit LoRA training."""
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model


def make_lora_config():
    """Return the project-wide LoRA config."""
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )


def cycle_lora(peft_model, new_config=None, adapter_name: str = "default"):
    """Drop the current adapter and attach a fresh one in-place.

    This is the canonical pattern for training N adapters sequentially on a
    single base model without re-loading weights.

    Returns the same peft_model object (now with a clean, untrained adapter).
    """
    peft_model.delete_adapter(adapter_name)
    peft_model.add_adapter(adapter_name, new_config or make_lora_config())
    peft_model.set_adapter(adapter_name)
    return peft_model


def load_model_with_lora(adapter_path=None, device_map="auto"):
    """Load base model, optionally with a saved LoRA adapter.

    If adapter_path is None, adds a fresh (untrained) LoRA via add_adapter().
    """
    model, tokenizer = load_base_model(device_map=device_map)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    else:
        # add_adapter() is the non-mutating alternative to get_peft_model().
        # It returns None — the adapter is attached in-place.
        model.add_adapter(make_lora_config(), adapter_name="default")
        model.set_adapter("default")
    return model, tokenizer
