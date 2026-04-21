from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from config import (
    ENABLE_THINKING,
    INSTANCES_JSONL,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MODEL_ID,
    ORACLE_MIN_CHUNK_TOKENS,
    SEED,
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class FileRecord:
    file_key: str
    repo: str
    file_path: str
    full_file: str


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_instances(instances_path: Path = INSTANCES_JSONL) -> list[dict]:
    rows = []
    with instances_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def make_file_key(repo: str, file_path: str) -> str:
    raw = f"{repo}__{file_path}"
    # Keep keys filesystem-safe and deterministic.
    key = re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
    return key


def build_file_records(instances: list[dict]) -> list[FileRecord]:
    by_key: dict[str, FileRecord] = {}
    for row in instances:
        repo = row["repo"]
        file_path = row["file_path"]
        key = make_file_key(repo, file_path)
        if key not in by_key:
            by_key[key] = FileRecord(
                file_key=key,
                repo=repo,
                file_path=file_path,
                full_file=row["full_file"],
            )
    return sorted(by_key.values(), key=lambda x: x.file_key)


def select_instances(instances: list[dict], max_instances: int | None) -> list[dict]:
    if max_instances is None:
        return instances
    return instances[:max_instances]


def select_file_records(
    file_records: list[FileRecord], max_files: int | None
) -> list[FileRecord]:
    if max_files is None:
        return file_records
    return file_records[:max_files]


def truncate_to_budget(text: str, tokenizer, budget_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget_tokens:
        return text
    clipped = tokenizer.decode(ids[:budget_tokens], skip_special_tokens=True)
    cut = clipped.rfind("\n")
    if cut > 0:
        clipped = clipped[: cut + 1]
    return clipped + "\n# ... [truncated to token budget] ...\n"


def make_lora_config() -> LoraConfig:
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )


def load_model_and_tokenizer(model_id: str = MODEL_ID, use_4bit: bool = True):
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_cfg,
            dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    return model, tokenizer


def prepare_model_for_oracle_training(model):
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()
    return model


def cycle_lora_adapter(
    peft_model: PeftModel, adapter_name: str = "default"
) -> PeftModel:
    peft_model.delete_adapter(adapter_name)
    peft_model.add_adapter(adapter_name, make_lora_config())
    peft_model.set_adapter(adapter_name)
    return peft_model


def make_chat_prompt(
    masked_function: str, context_text: str, function_name: str
) -> tuple[str, str]:
    system_prompt = (
        "You complete Python function bodies. "
        "Return only the function body lines with correct indentation. "
        "Do not include markdown fences."
    )
    user_prompt = (
        f"Target function name: {function_name}\n\n"
        "Masked function:\n"
        f"{masked_function}\n\n"
        "Relevant file context:\n"
        f"{context_text}\n\n"
        "Output only the missing function body for the masked function."
    )
    return system_prompt, user_prompt


def generate_text(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def normalize_body_prediction(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    # If the model returns an entire function, keep only body lines.
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("def "):
        lines = lines[1:]

    # Drop obvious role/preamble artifacts.
    while lines and lines[0].strip().lower().startswith(("assistant", "here is")):
        lines = lines[1:]

    # Ensure body indentation exists.
    norm = []
    for line in lines:
        if not line.strip():
            norm.append("")
            continue
        if line.startswith("    "):
            norm.append(line.rstrip())
        else:
            norm.append(f"    {line.rstrip()}")
    return "\n".join(norm).rstrip()


def make_chunk_dataset_records(
    text: str,
    tokenizer,
    chunk_size: int,
) -> list[dict]:
    pad_id = tokenizer.pad_token_id
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        if len(chunk) < ORACLE_MIN_CHUNK_TOKENS:
            continue
        pad_n = chunk_size - len(chunk)
        chunks.append(
            {
                "input_ids": chunk + [pad_id] * pad_n,
                "attention_mask": [1] * len(chunk) + [0] * pad_n,
                "labels": chunk + [-100] * pad_n,
            }
        )
    return chunks
