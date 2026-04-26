"""Diagnostic 4: Verify model.reset() fully clears LoRA weights.

Runs the baseline probes, then internalizes a document, then resets
and runs the baseline again.  If scores differ, reset() is leaking.

Usage:
  cd experiments/lora-recall
  uv run python diag_reset_verify.py
"""

import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from checkpoint_config import read_max_ctx_chunk_len, resolve_checkpoint_path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "vendor" / "doc-to-lora" / "src"),
)

from chunk_helper import generate_with_chunks, internalize_chunked
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.utils import get_layers, get_peft_modules

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_D2L_ROOT = PROJECT_ROOT / "vendor" / "doc-to-lora"

CHECKPOINT_PATH = str(resolve_checkpoint_path())

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


# Fixed set of probes to run in both baseline passes
CONSISTENCY_PROBES = [
    "What is the capital of France?",
    "What programming language was created by Guido van Rossum?",
    "What is 7 times 8?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'["\']', "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def run_fixed_probes(model, tokenizer, label):
    """Run the consistency probes and return raw outputs."""
    print(f"\n  Running probes: {label}")
    results = []
    for q in CONSISTENCY_PROBES:
        chat = [{"role": "user", "content": q}]
        chat_ids = tokenizer.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                input_ids=chat_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        input_text = tokenizer.decode(chat_ids[0], skip_special_tokens=True)
        generated = full_response[len(input_text) :].strip()

        print(f"    Q: {q}")
        print(f"    A: {generated[:120]}")
        results.append({"question": q, "generated": generated})

    return results


def check_weight_norms(model, label):
    """Snapshot the norms of LoRA-targeted layers."""
    layers = get_layers(model.base_model)
    norms = {}
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            name = f"layer{layer_idx}.{module_info['name']}"
            w = module_info["module"].weight
            norms[name] = float(torch.norm(w).item())

    print(f"\n  Weight norms ({label}): first 5:")
    for k, v in list(norms.items())[:5]:
        print(f"    {k}: {v:.6f}")

    return norms


def main():
    print("=" * 70)
    print("DIAGNOSTIC 4: Reset verification")
    print("=" * 70)

    # Load model
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

    results = {}

    # --- Phase 1: Fresh baseline ---
    print("\n" + "─" * 50)
    print("Phase 1: Fresh baseline (no prior internalization)")
    print("─" * 50)

    norms_before = check_weight_norms(model, "fresh baseline")
    baseline_1 = run_fixed_probes(model, tokenizer, "baseline #1")

    # --- Phase 2: Internalize a large document ---
    print("\n" + "─" * 50)
    print("Phase 2: Internalize a document")
    print("─" * 50)

    doc_path = DOCS_DIR / "flask_sessions" / "variant_a.txt"
    doc_text = doc_path.read_text()
    print(f"  Internalizing: {doc_path} ({len(doc_text)} chars)")

    max_chunk_len = read_max_ctx_chunk_len(CHECKPOINT_PATH)
    print(f"  max_ctx_chunk_len from checkpoint config: {max_chunk_len}")
    with pushd(VENDOR_D2L_ROOT):
        n_chunks = internalize_chunked(model, doc_text, max_chunk_len=max_chunk_len)
    print(f"  Internalized: {n_chunks} chunk(s)")

    norms_during = check_weight_norms(model, "with LoRA active")

    # Check if the forward is patched
    layers = get_layers(model.base_model)
    patched_count = 0
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            if (
                hasattr(module_info["module"], "patched_forward")
                and module_info["module"].patched_forward
            ):
                patched_count += 1
    print(f"\n  Patched forward hooks active: {patched_count}")

    # Ask a doc-specific question while LoRA is active
    chat = [{"role": "user", "content": "What exception does NullSession raise?"}]
    chat_ids = tokenizer.apply_chat_template(
        chat,
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = generate_with_chunks(
            model,
            input_ids=chat_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    input_text = tokenizer.decode(chat_ids[0], skip_special_tokens=True)
    with_lora_answer = full_response[len(input_text) :].strip()
    print("\n  With LoRA — Q: What exception does NullSession raise?")
    print(f"  With LoRA — A: {with_lora_answer[:200]}")

    # --- Phase 3: Reset and re-run baseline ---
    print("\n" + "─" * 50)
    print("Phase 3: After model.reset()")
    print("─" * 50)

    model.reset()

    norms_after = check_weight_norms(model, "after reset")

    # Check patched forward hooks cleared
    patched_count_after = 0
    for layer_idx in model.hypernet.layer_indices:
        for module_info in get_peft_modules(layers[layer_idx], model.peft_config):
            if (
                hasattr(module_info["module"], "patched_forward")
                and module_info["module"].patched_forward
            ):
                patched_count_after += 1
    print(f"\n  Patched forward hooks after reset: {patched_count_after}")
    if patched_count_after > 0:
        print("  ⚠️  LEAK: Forward hooks NOT fully cleared!")

    baseline_2 = run_fixed_probes(model, tokenizer, "baseline #2 (after reset)")

    # --- Phase 4: Compare ---
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # Compare weight norms
    norm_diffs = {}
    max_diff = 0.0
    for k in norms_before:
        diff = abs(norms_after.get(k, 0) - norms_before[k])
        norm_diffs[k] = diff
        max_diff = max(max_diff, diff)

    print(f"\n  Max weight norm difference (before vs after reset): {max_diff:.8f}")
    if max_diff > 1e-5:
        print("  ⚠️  LEAK: Base weights were modified in-place!")
    else:
        print("  ✓  Base weights unchanged.")

    # Compare outputs
    print("\n  Output comparison:")
    all_match = True
    for r1, r2 in zip(baseline_1, baseline_2):
        match = normalize(r1["generated"]) == normalize(r2["generated"])
        status = "✓" if match else "✗"
        if not match:
            all_match = False
        print(f"    {status} Q: {r1['question']}")
        if not match:
            print(f"        Before: {r1['generated'][:100]}")
            print(f"        After:  {r2['generated'][:100]}")

    if all_match:
        print("\n  ✓  CONCLUSION: model.reset() fully restores baseline behavior.")
    else:
        print("\n  ⚠️  CONCLUSION: model.reset() does NOT fully restore baseline.")
        print("     There may be residual weight contamination.")

    results = {
        "weight_norm_max_diff": max_diff,
        "weight_leak_detected": max_diff > 1e-5,
        "forward_hooks_leaked": patched_count_after > 0,
        "outputs_match": all_match,
        "baseline_1": baseline_1,
        "baseline_2": baseline_2,
        "with_lora_answer": with_lora_answer,
    }

    # --- Save ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"diag4_reset_verify_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
