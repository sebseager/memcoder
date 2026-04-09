"""Diagnostic 5: LoRA recall suite.

For each (module, variant) pair:
  1. Internalize the document via model.internalize()
  2. Ask ~20 cloze-style probe questions
  3. Score exact-match recall
  4. Reset and move to next

Usage:
  cd experiments/lora-recall
    uv run python diag5_recall_suite.py
"""

import csv
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

import torch

# Add vendor src to path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "vendor" / "doc-to-lora" / "src"),
)

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

from internalize_chunked import generate_with_chunks, internalize_chunked

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_D2L_ROOT = PROJECT_ROOT / "vendor" / "doc-to-lora"


@contextmanager
def pushd(path: Path):
    """Temporarily change cwd so doc-to-lora can resolve relative assets."""
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
VARIANTS = ["variant_a", "variant_b"]

MAX_NEW_TOKENS = 100


def _read_max_ctx_chunk_len() -> int:
    """Read max_ctx_chunk_len from the checkpoint's args.yaml."""
    checkpoint_dir = Path(CHECKPOINT_PATH).parent.parent
    args_path = checkpoint_dir / "args.yaml"
    if args_path.exists():
        with open(args_path) as f:
            args = yaml.unsafe_load(f)
        val = args.get("max_ctx_chunk_len", -1)
        return int(val)
    return -1

NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
    "11": "eleven",
    "12": "twelve",
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase and normalize simple formatting/number variants."""
    text = text.lower().strip()
    text = re.sub(r'["\']', "", text)
    text = re.sub(
        r"\b\d+\b",
        lambda match: NUMBER_WORDS.get(match.group(0), match.group(0)),
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text


def exact_match(predicted: str, gold: str) -> bool:
    return normalize(gold) in normalize(predicted)


def extract_answer(response: str, question: str) -> str:
    """Try to extract the answer portion from the model response."""
    # The model response includes the chat, so strip the question prefix
    # Just take the generated part after the question
    return response


def _is_cuda_oom(exc: Exception) -> bool:
    """Best-effort CUDA OOM detection across torch/runtime variants."""
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
    """Retry plan: first preserve quality, then progressively reduce memory."""
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
    """Generate with CUDA-OOM retries so a single probe doesn't abort the run."""
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
                "    [OOM] Retrying with "
                f"max_new_tokens={next_max_new_tokens}, use_cache={next_use_cache}"
            )


# ---------------------------------------------------------------------------
# Main experiment
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


def run_probe(
    model, tokenizer, doc_text: str, probes: list[dict], max_chunk_len: int = -1
) -> list[dict]:
    """Internalize a document and run all probe questions. Returns scored results."""
    model.reset()
    with pushd(VENDOR_D2L_ROOT):
        n_chunks = internalize_chunked(model, doc_text, max_chunk_len=max_chunk_len)
    print(f"  Internalized: {len(doc_text)} chars, {n_chunks} chunk(s)")

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
        # Strip the input prompt from the response to get just the generated part
        input_text = tokenizer.decode(chat_ids[0], skip_special_tokens=True)
        generated = full_response[len(input_text) :].strip()

        del output
        del chat_ids
        _clear_cuda_cache()

        match = exact_match(generated, probe["answer"])

        results.append(
            {
                "id": probe["id"],
                "question": question,
                "gold_answer": probe["answer"],
                "generated": generated,
                "exact_match": match,
                "category": probe["category"],
            }
        )

        status = "✓" if match else "✗"
        print(f"  {status} [{probe['id']}] {question[:60]}...")
        if not match:
            print(f"    Expected: {probe['answer']}")
            print(f"    Got:      {generated[:120]}")

    model.reset()
    return results


def compute_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics from probe results."""
    total = len(results)
    correct = sum(1 for r in results if r["exact_match"])

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["exact_match"]:
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


def run_baseline(model, tokenizer, probes_by_module: dict) -> dict:
    """Run probes WITHOUT any document internalized (baseline)."""
    print("\n" + "=" * 70)
    print("BASELINE (no document)")
    print("=" * 70)

    model.reset()
    model._n_ctx_chunks = 1  # no LoRA active for baseline
    all_results = {}

    for module_name, probes in probes_by_module.items():
        print(f"\n--- {module_name} ---")
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

            del output
            del chat_ids
            _clear_cuda_cache()

            match = exact_match(generated, probe["answer"])
            results.append(
                {
                    "id": probe["id"],
                    "question": question,
                    "gold_answer": probe["answer"],
                    "generated": generated,
                    "exact_match": match,
                    "category": probe["category"],
                }
            )

            status = "✓" if match else "✗"
            print(f"  {status} [{probe['id']}] {question[:60]}...")
            if not match:
                print(f"    Expected: {probe['answer']}")
                print(f"    Got:      {generated[:120]}")

        all_results[module_name] = {
            "results": results,
            "stats": compute_stats(results),
        }

    return all_results


def main():
    print("=" * 70)
    print("DIAGNOSTIC 5: Recall suite")
    print("=" * 70)

    # Load probes
    with open(PROBES_PATH) as f:
        probes_by_module = json.load(f)

    # Load model
    model, tokenizer = load_model()

    # Read chunk config from checkpoint's training args
    max_chunk_len = _read_max_ctx_chunk_len()
    print(f"max_ctx_chunk_len from checkpoint config: {max_chunk_len}")

    # Run baseline first
    baseline = run_baseline(model, tokenizer, probes_by_module)

    # Run all (module, variant) combinations
    all_results = {"baseline": baseline}

    for module_name in MODULES:
        for variant in VARIANTS:
            key = f"{module_name}/{variant}"
            doc_path = DOCS_DIR / module_name / f"{variant}.txt"

            if not doc_path.exists():
                print(f"\nSkipping {key}: {doc_path} not found")
                continue

            print(f"\n{'=' * 70}")
            print(f"Running: {key}")
            print(f"{'=' * 70}")

            doc_text = doc_path.read_text()
            print(f"Document length: {len(doc_text)} chars")

            probes = probes_by_module[module_name]
            t0 = time.time()
            results = run_probe(
                model, tokenizer, doc_text, probes, max_chunk_len=max_chunk_len
            )
            elapsed = time.time() - t0

            stats = compute_stats(results)
            print(
                f"\n  Recall: {stats['correct']}/{stats['total']} = {stats['overall_recall']:.1%}"
            )
            print(f"  Time: {elapsed:.1f}s")
            for cat, cat_stats in stats["per_category"].items():
                print(
                    f"  {cat}: {cat_stats['correct']}/{cat_stats['total']} = {cat_stats['recall']:.1%}"
                )

            all_results[key] = {
                "results": results,
                "stats": stats,
                "elapsed_seconds": elapsed,
                "doc_length_chars": len(doc_text),
            }

    # Save all results with timestamp
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = RESULTS_DIR / f"diag5_recall_{ts}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Write per-probe CSV
    probes_csv_path = RESULTS_DIR / f"diag5_recall_{ts}_probes.csv"
    with open(probes_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "module",
                "variant",
                "probe_id",
                "category",
                "question",
                "gold_answer",
                "generated",
                "exact_match",
            ]
        )
        for cond_key, cond_data in all_results.items():
            if cond_key == "baseline":
                for mod, mod_data in cond_data.items():
                    for r in mod_data["results"]:
                        writer.writerow(
                            [
                                "baseline",
                                mod,
                                "",
                                r["id"],
                                r["category"],
                                r["question"],
                                r["gold_answer"],
                                r["generated"],
                                int(r["exact_match"]),
                            ]
                        )
            else:
                mod, var = cond_key.split("/", 1)
                for r in cond_data["results"]:
                    writer.writerow(
                        [
                            cond_key,
                            mod,
                            var,
                            r["id"],
                            r["category"],
                            r["question"],
                            r["gold_answer"],
                            r["generated"],
                            int(r["exact_match"]),
                        ]
                    )
    print(f"Per-probe CSV saved to {probes_csv_path}")

    # Write summary CSV
    summary_csv_path = RESULTS_DIR / f"diag5_recall_{ts}_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "module",
                "condition",
                "recall",
                "correct",
                "total",
                "baseline_recall",
                "delta_vs_baseline",
            ]
        )
        for module_name in MODULES:
            bl_stats = baseline[module_name]["stats"]
            bl_recall = bl_stats["overall_recall"]
            writer.writerow(
                [
                    module_name,
                    "baseline",
                    f"{bl_recall:.4f}",
                    bl_stats["correct"],
                    bl_stats["total"],
                    f"{bl_recall:.4f}",
                    "0.0000",
                ]
            )
            for variant in VARIANTS:
                key = f"{module_name}/{variant}"
                if key not in all_results:
                    continue
                s = all_results[key]["stats"]
                delta = s["overall_recall"] - bl_recall
                writer.writerow(
                    [
                        module_name,
                        variant,
                        f"{s['overall_recall']:.4f}",
                        s["correct"],
                        s["total"],
                        f"{bl_recall:.4f}",
                        f"{delta:+.4f}",
                    ]
                )
    print(f"Summary CSV saved to {summary_csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Condition':<40} {'Recall':>10} {'Correct':>10} {'Δ baseline':>12}")
    print("-" * 72)

    for module_name in MODULES:
        # Baseline for this module
        bl = baseline[module_name]["stats"]
        print(
            f"  {module_name}/baseline{'':<15} {bl['overall_recall']:>9.1%} {bl['correct']:>5}/{bl['total']}{'':>12}"
        )

        for variant in VARIANTS:
            key = f"{module_name}/{variant}"
            if key in all_results:
                s = all_results[key]["stats"]
                delta = s["overall_recall"] - bl["overall_recall"]
                print(
                    f"  {key:<38} {s['overall_recall']:>9.1%} {s['correct']:>5}/{s['total']}  {delta:>+9.1%}"
                )

    # Variant A vs B comparison
    print("\n" + "=" * 70)
    print("VARIANT A vs B COMPARISON")
    print("=" * 70)
    for module_name in MODULES:
        key_a = f"{module_name}/variant_a"
        key_b = f"{module_name}/variant_b"
        if key_a in all_results and key_b in all_results:
            ra = all_results[key_a]["stats"]["overall_recall"]
            rb = all_results[key_b]["stats"]["overall_recall"]
            delta = rb - ra
            print(f"  {module_name}: A={ra:.1%}  B={rb:.1%}  Δ={delta:+.1%}")

    # Per-category breakdown
    print("\n" + "=" * 70)
    print("PER-CATEGORY BREAKDOWN (all modules combined)")
    print("=" * 70)
    for variant in VARIANTS:
        cat_agg = {}
        for module_name in MODULES:
            key = f"{module_name}/{variant}"
            if key not in all_results:
                continue
            for cat, cs in all_results[key]["stats"]["per_category"].items():
                if cat not in cat_agg:
                    cat_agg[cat] = {"correct": 0, "total": 0}
                cat_agg[cat]["correct"] += cs["correct"]
                cat_agg[cat]["total"] += cs["total"]

        print(f"\n  {variant}:")
        for cat, cs in sorted(cat_agg.items()):
            rate = cs["correct"] / cs["total"] if cs["total"] > 0 else 0
            print(f"    {cat:<20} {cs['correct']:>3}/{cs['total']:<3} = {rate:.1%}")


if __name__ == "__main__":
    main()
