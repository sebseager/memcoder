"""Diagnostic 7: Routing signal quality.

Tests whether loading the *correct* LoRA (matching the task's knowledge
domain) measurably helps vs. loading the *wrong* LoRA (a different domain).

For each module's probe set:
  correct  — internalize the matching document, run probes
  wrong_X  — internalize a non-matching document, run probes (for each other module)
  baseline — no document internalized, run probes

Metrics:
  routing_benefit  = recall_correct - avg(recall_wrong)
  wrong_vs_baseline = avg(recall_wrong) - recall_baseline

A large routing_benefit means the agent *must* pick the right LoRA.
A wrong_vs_baseline near zero means an irrelevant LoRA is neutral; a
negative value means it actively hurts.

Usage:
  cd experiments/lora-recall
  uv run python diag7_routing.py
"""

import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

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
CHECKPOINT_PATH = str(
    Path(__file__).parents[1]
    / "doc-to-lora"
    / "trained_d2l"
    / "gemma_demo"
    / "checkpoint-80000"
    / "pytorch_model.bin"
)

DOCS_DIR = Path(__file__).parent / "docs"
PROBES_PATH = Path(__file__).parent / "probes.json"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODULES = ["flask_sessions", "click_types", "marshmallow_validate"]
VARIANT = "variant_a"

MAX_NEW_TOKENS = 100


def _read_max_ctx_chunk_len() -> int:
    checkpoint_dir = Path(CHECKPOINT_PATH).parent.parent
    args_path = checkpoint_dir / "args.yaml"
    if args_path.exists():
        with open(args_path) as f:
            args = yaml.unsafe_load(f)
        return int(args.get("max_ctx_chunk_len", -1))
    return -1


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
    return {
        "overall_recall": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DIAGNOSTIC 7: Routing signal quality")
    print("=" * 70)

    with open(PROBES_PATH) as f:
        probes_by_module = json.load(f)

    model, tokenizer = load_model()
    max_chunk_len = _read_max_ctx_chunk_len()
    print(f"max_ctx_chunk_len from checkpoint config: {max_chunk_len}")

    # Pre-load all docs
    docs: dict[str, str] = {}
    for mod in MODULES:
        doc_path = DOCS_DIR / mod / f"{VARIANT}.txt"
        if doc_path.exists():
            docs[mod] = doc_path.read_text()
        else:
            print(f"WARNING: missing {doc_path}")

    all_results: dict = {}

    for target_mod in MODULES:
        if target_mod not in docs:
            continue

        probes = probes_by_module[target_mod]
        mod_results: dict = {}

        # --- baseline (no LoRA) ---
        print(f"\n{'=' * 70}")
        print(f"[{target_mod}] baseline — no document")
        print(f"{'=' * 70}")
        model.reset()
        model._n_ctx_chunks = 1
        results = run_probes(model, tokenizer, probes)
        stats = compute_stats(results)
        mod_results["baseline"] = {"results": results, "stats": stats}
        print(
            f"  baseline recall: {stats['correct']}/{stats['total']}"
            f" = {stats['overall_recall']:.1%}"
        )

        # --- correct LoRA ---
        print(f"\n{'=' * 70}")
        print(f"[{target_mod}] correct — internalize {target_mod}")
        print(f"{'=' * 70}")
        model.reset()
        with pushd(VENDOR_D2L_ROOT):
            n = internalize_chunked(
                model, docs[target_mod], max_chunk_len=max_chunk_len
            )
        print(f"  Internalized: {len(docs[target_mod])} chars, {n} chunk(s)")
        results = run_probes(model, tokenizer, probes)
        stats = compute_stats(results)
        mod_results["correct"] = {"results": results, "stats": stats}
        print(
            f"  correct recall: {stats['correct']}/{stats['total']}"
            f" = {stats['overall_recall']:.1%}"
        )

        # --- wrong LoRAs (one per other module) ---
        wrong_recalls: list[float] = []
        for wrong_mod in MODULES:
            if wrong_mod == target_mod or wrong_mod not in docs:
                continue

            cond_key = f"wrong_{wrong_mod}"
            print(f"\n{'=' * 70}")
            print(f"[{target_mod}] {cond_key} — internalize {wrong_mod}")
            print(f"{'=' * 70}")
            model.reset()
            with pushd(VENDOR_D2L_ROOT):
                n = internalize_chunked(
                    model, docs[wrong_mod], max_chunk_len=max_chunk_len
                )
            print(f"  Internalized: {len(docs[wrong_mod])} chars, {n} chunk(s)")
            results = run_probes(model, tokenizer, probes)
            stats = compute_stats(results)
            mod_results[cond_key] = {"results": results, "stats": stats}
            wrong_recalls.append(stats["overall_recall"])
            print(
                f"  {cond_key} recall: {stats['correct']}/{stats['total']}"
                f" = {stats['overall_recall']:.1%}"
            )

        # --- Compute routing metrics ---
        avg_wrong = sum(wrong_recalls) / len(wrong_recalls) if wrong_recalls else 0
        baseline_recall = mod_results["baseline"]["stats"]["overall_recall"]
        correct_recall = mod_results["correct"]["stats"]["overall_recall"]

        routing_benefit = correct_recall - avg_wrong
        wrong_vs_baseline = avg_wrong - baseline_recall

        mod_results["routing_metrics"] = {
            "correct_recall": correct_recall,
            "avg_wrong_recall": avg_wrong,
            "baseline_recall": baseline_recall,
            "routing_benefit": routing_benefit,
            "wrong_vs_baseline": wrong_vs_baseline,
        }

        print(f"\n  --- Routing metrics for {target_mod} ---")
        print(f"  correct:        {correct_recall:.1%}")
        print(f"  avg(wrong):     {avg_wrong:.1%}")
        print(f"  baseline:       {baseline_recall:.1%}")
        print(f"  routing_benefit (correct - avg_wrong): {routing_benefit:+.1%}")
        print(f"  wrong_vs_baseline:                     {wrong_vs_baseline:+.1%}")

        all_results[target_mod] = mod_results

    # --- Save results ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"diag7_routing_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary CSV
    summary_csv = RESULTS_DIR / f"diag7_routing_{ts}_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "target_module",
                "condition",
                "loaded_doc",
                "recall",
                "correct",
                "total",
            ]
        )
        for target_mod, mod_data in all_results.items():
            for cond_key, cond_data in mod_data.items():
                if cond_key == "routing_metrics":
                    continue
                if cond_key == "baseline":
                    loaded = "(none)"
                elif cond_key == "correct":
                    loaded = target_mod
                else:
                    loaded = cond_key.replace("wrong_", "")

                s = cond_data["stats"]
                writer.writerow(
                    [
                        target_mod,
                        cond_key,
                        loaded,
                        f"{s['overall_recall']:.4f}",
                        s["correct"],
                        s["total"],
                    ]
                )
    print(f"Summary CSV saved to {summary_csv}")

    # Print summary table
    print("\n" + "=" * 70)
    print("ROUTING SIGNAL QUALITY SUMMARY")
    print("=" * 70)
    print(
        f"{'Module':<25} {'Correct':>9} {'Avg Wrong':>11}"
        f" {'Baseline':>10} {'Benefit':>9} {'Wrong-BL':>10}"
    )
    print("-" * 74)
    for target_mod, mod_data in all_results.items():
        m = mod_data["routing_metrics"]
        print(
            f"  {target_mod:<23}"
            f" {m['correct_recall']:>8.1%}"
            f" {m['avg_wrong_recall']:>10.1%}"
            f" {m['baseline_recall']:>9.1%}"
            f" {m['routing_benefit']:>+8.1%}"
            f" {m['wrong_vs_baseline']:>+9.1%}"
        )


if __name__ == "__main__":
    main()
