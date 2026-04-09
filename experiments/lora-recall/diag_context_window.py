"""Diagnostic 2: Context window / token length check.

1. Measures the token length of each document through the context encoder's
   tokenizer and compares it to the model's max_position_embeddings.
2. Feeds a deliberately tiny 3-sentence document with a single made-up fact,
   then probes for that fact — to rule out truncation as a cause.

Usage:
  cd experiments/lora-recall
  uv run python diag_context_window.py
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "vendor" / "doc-to-lora" / "src"),
)

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_D2L_ROOT = PROJECT_ROOT / "vendor" / "doc-to-lora"

CHECKPOINT_PATH = str(
    Path(__file__).parents[1]
    / "doc-to-lora"
    / "trained_d2l"
    / "gemma_demo"
    / "checkpoint-80000"
    / "pytorch_model.bin"
)

DOCS_DIR = Path(__file__).parent / "docs"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MAX_NEW_TOKENS = 100


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


TINY_DOC = (
    "The Zylophone library provides a single helper function called compute_blorb. "
    "compute_blorb takes exactly two arguments: a frobulation index (int) and a "
    "squench factor (float). It returns a NexusResult whose status field is always "
    "the string 'nominal' on success."
)

TINY_PROBES = [
    {
        "question": "What function does the Zylophone library provide?",
        "answer": "compute_blorb",
    },
    {
        "question": "How many arguments does compute_blorb take?",
        "answer": "two",
    },
    {
        "question": "What type does compute_blorb return?",
        "answer": "NexusResult",
    },
    {
        "question": "What is the status field value on success?",
        "answer": "nominal",
    },
]


def normalize(text: str) -> str:
    import re

    text = text.lower().strip()
    text = re.sub(r'["\']', "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(predicted: str, gold: str) -> bool:
    return normalize(gold) in normalize(predicted)


def main():
    print("=" * 70)
    print("DIAGNOSTIC 2: Context window & token length check")
    print("=" * 70)

    # --- Load model ---
    print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
    state_dict = torch.load(CHECKPOINT_PATH, weights_only=False)
    state_dict["ctx_encoder_args"].quantize_ctx_encoder = False

    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_sequence_packing=False,
        use_flash_attn=True,
    )
    model.reset()
    with pushd(VENDOR_D2L_ROOT):
        tokenizer = get_tokenizer(model.base_model.name_or_path)

    # Get context encoder tokenizer (may differ from generation tokenizer)
    with pushd(VENDOR_D2L_ROOT):
        ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    # --- Part A: Check token lengths ---
    print("\n" + "-" * 50)
    print("Part A: Token length analysis")
    print("-" * 50)

    max_pos = model.ctx_encoder.config.max_position_embeddings
    print(f"\nContext encoder max_position_embeddings: {max_pos}")
    print(f"Context encoder model: {model.ctx_encoder.base_model.name_or_path}")

    results = {"max_position_embeddings": max_pos, "documents": {}}

    modules = ["flask_sessions", "click_types", "marshmallow_validate"]
    variants = ["variant_a", "variant_b"]

    for module_name in modules:
        for variant in variants:
            doc_path = DOCS_DIR / module_name / f"{variant}.txt"
            if not doc_path.exists():
                continue
            doc_text = doc_path.read_text()
            token_ids = ctx_tokenizer.encode(doc_text)
            n_tokens = len(token_ids)
            exceeds = n_tokens > max_pos
            key = f"{module_name}/{variant}"

            results["documents"][key] = {
                "chars": len(doc_text),
                "tokens": n_tokens,
                "exceeds_max": exceeds,
            }

            flag = " ⚠️ EXCEEDS MAX" if exceeds else ""
            print(f"  {key}: {len(doc_text)} chars, {n_tokens} tokens{flag}")

    # Also check the vendor's demo doc
    sakana_path = VENDOR_D2L_ROOT / "data" / "sakana_wiki.txt"
    if sakana_path.exists():
        doc_text = sakana_path.read_text()
        token_ids = ctx_tokenizer.encode(doc_text)
        n_tokens = len(token_ids)
        results["documents"]["sakana_wiki (vendor demo)"] = {
            "chars": len(doc_text),
            "tokens": n_tokens,
            "exceeds_max": n_tokens > max_pos,
        }
        print(f"  sakana_wiki (vendor demo): {len(doc_text)} chars, {n_tokens} tokens")

    # --- Part B: Tiny document probe ---
    print("\n" + "-" * 50)
    print("Part B: Tiny document probe (3 sentences, made-up fact)")
    print("-" * 50)

    tiny_tokens = ctx_tokenizer.encode(TINY_DOC)
    print(f"\nTiny doc length: {len(TINY_DOC)} chars, {len(tiny_tokens)} tokens")
    print(f"Document:\n  {TINY_DOC}\n")

    model.reset()
    with pushd(VENDOR_D2L_ROOT):
        model.internalize(TINY_DOC)

    tiny_results = []
    for probe in TINY_PROBES:
        question = probe["question"]
        chat = [{"role": "user", "content": question}]
        chat_ids = tokenizer.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(input_ids=chat_ids, max_new_tokens=MAX_NEW_TOKENS)

        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        input_text = tokenizer.decode(chat_ids[0], skip_special_tokens=True)
        generated = full_response[len(input_text) :].strip()

        match = exact_match(generated, probe["answer"])
        status = "✓" if match else "✗"
        print(f"  {status} Q: {question}")
        print(f"      Gold: {probe['answer']}")
        print(f"      Got:  {generated[:150]}")

        tiny_results.append(
            {
                "question": question,
                "gold_answer": probe["answer"],
                "generated": generated,
                "exact_match": match,
            }
        )

    model.reset()

    correct = sum(1 for r in tiny_results if r["exact_match"])
    total = len(tiny_results)
    print(f"\nTiny doc recall: {correct}/{total} = {correct / total:.0%}")

    results["tiny_doc"] = {
        "document": TINY_DOC,
        "tokens": len(tiny_tokens),
        "recall": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "probes": tiny_results,
    }

    # --- Save ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"diag2_context_window_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
