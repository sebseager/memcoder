"""Diagnostic 6: Compositional stability — multi-LoRA merging.

Tests whether internalizing two documents simultaneously (as separate
context chunks) retains facts from *both*, or whether the second document's
LoRA catastrophically overwrites the first.

Conditions per document pair (A, B):
  solo_A   — internalize only doc A,  run A-probes
  solo_B   — internalize only doc B,  run B-probes
  merged   — internalize A+B together, run A-probes AND B-probes

Metrics:
  retention_A = merged_recall_A / solo_recall_A
  retention_B = merged_recall_B / solo_recall_B

Usage:
  cd experiments/lora-recall
  uv run python diag6_compositional.py
"""

import csv
import gc
import json
import os
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
from score_helper import score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_D2L_ROOT = PROJECT_ROOT / "vendor" / "doc-to-lora"


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = str(resolve_checkpoint_path())

DOCS_DIR = Path(__file__).parent / "docs"
PROBES_PATH = Path(__file__).parent / "probes.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODULES = ["flask_sessions", "click_types", "marshmallow_validate"]
# All unique pairs of modules
DOC_PAIRS = [
    ("flask_sessions", "click_types"),
    ("flask_sessions", "marshmallow_validate"),
    ("click_types", "marshmallow_validate"),
]
# Use variant_a for the compositional test
VARIANT = "variant_a"

MAX_NEW_TOKENS = 100


def _is_cuda_oom(exc: Exception) -> bool:
    accelerator_error = getattr(torch, "AcceleratorError", None)
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if accelerator_error is not None and isinstance(exc, accelerator_error):
        text = str(exc).lower()
        return "out of memory" in text or "cudaerrormemoryallocation" in text
    text = str(exc).lower()
    return (
        ("cuda" in text and "out of memory" in text)
        or "cuda out of memory" in text
        or "cudaerrormemoryallocation" in text
    )


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_generation_attempts() -> list[tuple[int, bool]]:
    candidates = [
        (MAX_NEW_TOKENS, True),
        (min(MAX_NEW_TOKENS, 64), True),
        (min(MAX_NEW_TOKENS, 64), False),
        (min(MAX_NEW_TOKENS, 48), False),
        (min(MAX_NEW_TOKENS, 32), False),
    ]
    attempts = []
    seen = set()
    for max_new_tokens, use_cache in candidates:
        key = (max_new_tokens, use_cache)
        if max_new_tokens <= 0 or key in seen:
            continue
        seen.add(key)
        attempts.append(key)
    return attempts


def generate_with_retry(model, tokenizer, chat_ids: torch.Tensor, question: str):
    attempts = _build_generation_attempts()
    for idx, (max_new_tokens, use_cache) in enumerate(attempts, start=1):
        try:
            generate_kwargs = {
                "input_ids": chat_ids,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": use_cache,
            }
            if tokenizer.eos_token_id is not None:
                generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
            with torch.no_grad():
                return generate_with_chunks(model, **generate_kwargs)
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            _clear_cuda_cache()
            if idx == len(attempts):
                raise RuntimeError(
                    f"CUDA OOM while generating for probe: {question[:80]}"
                ) from exc
            next_max_new_tokens, next_use_cache = attempts[idx]
            print(
                f"    [OOM] Retrying with "
                f"max_new_tokens={next_max_new_tokens}, use_cache={next_use_cache}"
            )


# ---------------------------------------------------------------------------
# Multi-doc internalization
# ---------------------------------------------------------------------------
def internalize_multi_docs(
    model,
    docs: list[str],
    max_chunk_len: int = -1,
) -> int:
    """Internalize multiple documents by generating LoRAs independently, then
    concatenating along the chunk dimension.

    Each document is internalized separately (so generate_weights always sees
    batch=1, which this checkpoint requires).  The resulting per-document LoRA
    weight tensors are concatenated along dim-0 (the chunk axis) and stored on
    the model.  At generation time, ``combine_lora`` merges them along the rank
    dimension — exactly the path the vendor code uses for multi-chunk contexts.

    Returns the total number of context chunks.
    """
    all_loras: list[dict] = []
    total_chunks = 0

    for doc_text in docs:
        model.reset()
        _clear_cuda_cache()
        with pushd(VENDOR_D2L_ROOT):
            n = internalize_chunked(model, doc_text, max_chunk_len=max_chunk_len)
        all_loras.append(model.generated_loras)
        total_chunks += n

    # Concatenate along the chunk dimension (dim 0)
    merged: dict[str, dict[str, torch.Tensor]] = {}
    for module_name in all_loras[0]:
        merged[module_name] = {}
        for matrix_key in ("A", "B"):
            tensors = [lora[module_name][matrix_key] for lora in all_loras]
            merged[module_name][matrix_key] = torch.cat(tensors, dim=0)

    model.generated_loras = merged
    model._n_ctx_chunks = total_chunks

    return total_chunks


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def load_model():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
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
    return model, tokenizer


def run_probes(model, tokenizer, probes: list[dict]) -> list[dict]:
    """Run probe questions against the currently-internalized LoRA."""
    results = []
    for probe in probes:
        question = probe["question"]
        chat = [{"role": "user", "content": question}]
        chat_ids = tokenizer.apply_chat_template(
            chat,
            add_special_tokens=False,
            return_attention_mask=False,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        output = generate_with_retry(model, tokenizer, chat_ids, question)
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        input_text = tokenizer.decode(chat_ids[0], skip_special_tokens=True)
        generated = full_response[len(input_text) :].strip()

        del output, chat_ids
        _clear_cuda_cache()

        result = score(generated, probe["answer"])
        results.append(
            {
                "id": probe["id"],
                "question": question,
                "gold_answer": probe["answer"],
                "generated": generated,
                "correct": result.correct,
                "score_method": result.method,
                "category": probe["category"],
            }
        )

        status = "✓" if result.correct else "✗"
        print(f"  {status} [{probe['id']}] {question[:60]}...")
        if not result.correct:
            print(f"    Expected: {probe['answer']}")
            print(f"    Got:      {generated[:120]}")

    return results


def compute_stats(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    cat_rates = {
        cat: {
            "recall": v["correct"] / v["total"] if v["total"] > 0 else 0,
            "correct": v["correct"],
            "total": v["total"],
        }
        for cat, v in categories.items()
    }
    return {
        "overall_recall": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "per_category": cat_rates,
    }


def _gc_and_clear():
    """Force Python GC and clear CUDA cache to reclaim memory between phases."""
    gc.collect()
    _clear_cuda_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DIAGNOSTIC 6: Compositional stability (multi-LoRA merge)")
    print("=" * 70)

    with open(PROBES_PATH) as f:
        probes_by_module = json.load(f)

    model, tokenizer = load_model()
    max_chunk_len = read_max_ctx_chunk_len(CHECKPOINT_PATH)
    print(f"max_ctx_chunk_len from checkpoint config: {max_chunk_len}")

    # --- Phase 1: solo runs (once per module, cached) ---
    solo_cache: dict[str, dict] = {}
    docs: dict[str, str] = {}

    for mod in MODULES:
        doc_path = DOCS_DIR / mod / f"{VARIANT}.txt"
        if not doc_path.exists():
            print(f"\nSkipping {mod}: {doc_path} not found")
            continue

        docs[mod] = doc_path.read_text()
        probes = probes_by_module[mod]

        print(f"\n{'=' * 70}")
        print(f"solo: internalize only {mod}")
        print(f"{'=' * 70}")
        _gc_and_clear()
        model.reset()
        with torch.no_grad(), pushd(VENDOR_D2L_ROOT):
            n = internalize_chunked(model, docs[mod], max_chunk_len=max_chunk_len)
        print(f"  Internalized {mod}: {len(docs[mod])} chars, {n} chunk(s)")
        results = run_probes(model, tokenizer, probes)
        stats = compute_stats(results)
        print(
            f"  solo recall: {stats['correct']}/{stats['total']}"
            f" = {stats['overall_recall']:.1%}"
        )
        solo_cache[mod] = {"results": results, "stats": stats}

    model.reset()
    _gc_and_clear()

    # --- Phase 2: merged runs (one per pair) ---
    all_results: dict = {}

    for mod_a, mod_b in DOC_PAIRS:
        if mod_a not in docs or mod_b not in docs:
            print(f"\nSkipping pair ({mod_a}, {mod_b}): missing docs")
            continue

        pair_key = f"{mod_a}+{mod_b}"
        probes_a = probes_by_module[mod_a]
        probes_b = probes_by_module[mod_b]

        print(f"\n{'=' * 70}")
        print(f"[{pair_key}] merged: internalize {mod_a} + {mod_b}")
        print(f"{'=' * 70}")
        _gc_and_clear()
        model.reset()
        with torch.no_grad(), pushd(VENDOR_D2L_ROOT):
            n_merged = internalize_multi_docs(
                model, [docs[mod_a], docs[mod_b]], max_chunk_len=max_chunk_len
            )
        print(
            f"  Internalized both: {len(docs[mod_a]) + len(docs[mod_b])} chars total,"
            f" {n_merged} chunk(s)"
        )

        print(f"\n  Probing {mod_a} facts on merged LoRA:")
        merged_a_results = run_probes(model, tokenizer, probes_a)
        merged_a_stats = compute_stats(merged_a_results)

        print(f"\n  Probing {mod_b} facts on merged LoRA:")
        merged_b_results = run_probes(model, tokenizer, probes_b)
        merged_b_stats = compute_stats(merged_b_results)

        solo_a_stats = solo_cache[mod_a]["stats"]
        solo_b_stats = solo_cache[mod_b]["stats"]

        retention_a = (
            merged_a_stats["overall_recall"] / solo_a_stats["overall_recall"]
            if solo_a_stats["overall_recall"] > 0
            else float("nan")
        )
        retention_b = (
            merged_b_stats["overall_recall"] / solo_b_stats["overall_recall"]
            if solo_b_stats["overall_recall"] > 0
            else float("nan")
        )

        print(f"\n  --- Retention for {pair_key} ---")
        print(
            f"  {mod_a}: solo={solo_a_stats['overall_recall']:.1%}"
            f"  merged={merged_a_stats['overall_recall']:.1%}"
            f"  retention={retention_a:.1%}"
        )
        print(
            f"  {mod_b}: solo={solo_b_stats['overall_recall']:.1%}"
            f"  merged={merged_b_stats['overall_recall']:.1%}"
            f"  retention={retention_b:.1%}"
        )

        all_results[pair_key] = {
            "solo_a": {
                "module": mod_a,
                "results": solo_cache[mod_a]["results"],
                "stats": solo_a_stats,
            },
            "solo_b": {
                "module": mod_b,
                "results": solo_cache[mod_b]["results"],
                "stats": solo_b_stats,
            },
            "merged_a": {
                "module": mod_a,
                "results": merged_a_results,
                "stats": merged_a_stats,
            },
            "merged_b": {
                "module": mod_b,
                "results": merged_b_results,
                "stats": merged_b_stats,
            },
            "retention_a": retention_a,
            "retention_b": retention_b,
        }

        model.reset()

    # --- Save results ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"diag6_compositional_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary CSV
    summary_csv = RESULTS_DIR / f"diag6_compositional_{ts}_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "pair",
                "module",
                "condition",
                "recall",
                "correct",
                "total",
                "retention",
            ]
        )
        for pair_key, pair_data in all_results.items():
            mod_a = pair_data["solo_a"]["module"]
            mod_b = pair_data["solo_b"]["module"]

            sa = pair_data["solo_a"]["stats"]
            sb = pair_data["solo_b"]["stats"]
            ma = pair_data["merged_a"]["stats"]
            mb = pair_data["merged_b"]["stats"]

            writer.writerow(
                [
                    pair_key,
                    mod_a,
                    "solo",
                    f"{sa['overall_recall']:.4f}",
                    sa["correct"],
                    sa["total"],
                    "",
                ]
            )
            writer.writerow(
                [
                    pair_key,
                    mod_a,
                    "merged",
                    f"{ma['overall_recall']:.4f}",
                    ma["correct"],
                    ma["total"],
                    f"{pair_data['retention_a']:.4f}",
                ]
            )
            writer.writerow(
                [
                    pair_key,
                    mod_b,
                    "solo",
                    f"{sb['overall_recall']:.4f}",
                    sb["correct"],
                    sb["total"],
                    "",
                ]
            )
            writer.writerow(
                [
                    pair_key,
                    mod_b,
                    "merged",
                    f"{mb['overall_recall']:.4f}",
                    mb["correct"],
                    mb["total"],
                    f"{pair_data['retention_b']:.4f}",
                ]
            )
    print(f"Summary CSV saved to {summary_csv}")

    # Print summary table
    print("\n" + "=" * 70)
    print("COMPOSITIONAL STABILITY SUMMARY")
    print("=" * 70)
    print(f"{'Pair':<40} {'Module':<25} {'Solo':>8} {'Merged':>8} {'Retention':>10}")
    print("-" * 91)
    for pair_key, pair_data in all_results.items():
        sa = pair_data["solo_a"]["stats"]
        ma = pair_data["merged_a"]["stats"]
        sb = pair_data["solo_b"]["stats"]
        mb = pair_data["merged_b"]["stats"]

        print(
            f"  {pair_key:<38} {pair_data['solo_a']['module']:<23}"
            f" {sa['overall_recall']:>7.1%} {ma['overall_recall']:>7.1%}"
            f" {pair_data['retention_a']:>9.1%}"
        )
        print(
            f"  {'':38} {pair_data['solo_b']['module']:<23}"
            f" {sb['overall_recall']:>7.1%} {mb['overall_recall']:>7.1%}"
            f" {pair_data['retention_b']:>9.1%}"
        )


if __name__ == "__main__":
    main()
