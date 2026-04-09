"""Diagnostic 3: Synthetic canary test.

Internalizes a completely fictitious document about "Blorbinator" — a library
Gemma has zero pretraining signal on — and probes for those made-up facts.

If the LoRA encodes anything, it MUST show up here.
If it confabulates ➜ LoRA is not functioning as a recall mechanism.
If it works     ➜ real documents are the problem (format mismatch).

Usage:
  cd experiments/lora-recall
  uv run python diag_canary.py
"""

import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "vendor" / "doc-to-lora" / "src"),
)

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

from internalize_chunked import generate_with_chunks, internalize_chunked

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

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

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


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ---------------------------------------------------------------------------
# Canary document: 100% fictitious, zero pretraining signal
# ---------------------------------------------------------------------------
CANARY_DOC = """\
The Blorbinator library defines a single validator: CrunchValidator.
CrunchValidator raises FlorpException when the input contains the string "zyx".
The default_message for CrunchValidator is: "Crunch failed: unexpected zyx sequence."
The maximum allowed crunch depth is 7.
The Blorbinator version constant is "0.42.0-alpha".
CrunchValidator accepts an optional parameter called strictness, which defaults to 3.
When strictness is set to 0, CrunchValidator silently ignores all errors.\
"""

CANARY_PROBES = [
    {
        "id": "canary01",
        "question": "What exception does CrunchValidator raise?",
        "answer": "FlorpException",
        "category": "factual",
    },
    {
        "id": "canary02",
        "question": "Complete this exact string: The default_message for CrunchValidator is: _____.",
        "answer": "Crunch failed: unexpected zyx sequence.",
        "category": "factual",
    },
    {
        "id": "canary03",
        "question": "What is the maximum crunch depth?",
        "answer": "7",
        "category": "factual",
    },
    {
        "id": "canary04",
        "question": "What string triggers FlorpException in CrunchValidator?",
        "answer": "zyx",
        "category": "factual",
    },
    {
        "id": "canary05",
        "question": "What is the Blorbinator version constant?",
        "answer": "0.42.0-alpha",
        "category": "factual",
    },
    {
        "id": "canary06",
        "question": "What is the default value of the strictness parameter?",
        "answer": "3",
        "category": "factual",
    },
    {
        "id": "canary07",
        "question": "What happens when strictness is set to 0?",
        "answer": "silently ignores all errors",
        "category": "factual",
    },
]

# Also test with the vendor's known-good document for comparison
SAKANA_PROBES = [
    {
        "id": "sakana01",
        "question": "Where is Sakana AI headquartered?",
        "answer": "Tokyo",
        "category": "vendor_doc",
    },
    {
        "id": "sakana02",
        "question": "Who is the CEO of Sakana AI?",
        "answer": "David Ha",
        "category": "vendor_doc",
    },
]


def normalize(text: str) -> str:
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


def run_probes(model, tokenizer, doc_text, probes, label, max_chunk_len=-1):
    """Internalize a document and probe it."""
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"  Doc: {len(doc_text)} chars")
    print(f"{'─' * 50}")

    model.reset()
    with pushd(VENDOR_D2L_ROOT):
        n_chunks = internalize_chunked(model, doc_text, max_chunk_len=max_chunk_len)
    print(f"  Internalized: {n_chunks} chunk(s)")

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

        with torch.no_grad():
            output = generate_with_chunks(
                model, input_ids=chat_ids, max_new_tokens=MAX_NEW_TOKENS
            )

        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        input_text = tokenizer.decode(chat_ids[0], skip_special_tokens=True)
        generated = full_response[len(input_text) :].strip()

        match = exact_match(generated, probe["answer"])
        status = "✓" if match else "✗"
        print(f"  {status} [{probe['id']}] {question}")
        print(f"      Gold: {probe['answer']}")
        print(f"      Got:  {generated[:200]}")

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

    model.reset()

    correct = sum(1 for r in results if r["exact_match"])
    total = len(results)
    print(f"\n  Recall: {correct}/{total} = {correct / total:.0%}")
    return results


def main():
    print("=" * 70)
    print("DIAGNOSTIC 3: Synthetic canary test")
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

    all_results = {}

    max_chunk_len = _read_max_ctx_chunk_len()
    print(f"max_ctx_chunk_len from checkpoint config: {max_chunk_len}")

    # --- Canary test ---
    canary_results = run_probes(
        model, tokenizer, CANARY_DOC, CANARY_PROBES,
        "CANARY: Blorbinator (fictitious)",
        max_chunk_len=max_chunk_len,
    )
    all_results["canary"] = {
        "document": CANARY_DOC,
        "results": canary_results,
        "correct": sum(1 for r in canary_results if r["exact_match"]),
        "total": len(canary_results),
    }

    # --- Vendor demo doc (positive control) ---
    sakana_path = VENDOR_D2L_ROOT / "data" / "sakana_wiki.txt"
    if sakana_path.exists():
        sakana_doc = sakana_path.read_text()
        sakana_results = run_probes(
            model,
            tokenizer,
            sakana_doc,
            SAKANA_PROBES,
            "POSITIVE CONTROL: Sakana AI wiki (vendor demo doc)",
            max_chunk_len=max_chunk_len,
        )
        all_results["sakana_control"] = {
            "document_path": str(sakana_path),
            "results": sakana_results,
            "correct": sum(1 for r in sakana_results if r["exact_match"]),
            "total": len(sakana_results),
        }

    # --- Baseline: canary probes WITHOUT internalization ---
    print(f"\n{'─' * 50}")
    print("  BASELINE: canary probes without any document")
    print(f"{'─' * 50}")

    model.reset()
    model._n_ctx_chunks = 1  # no LoRA active for baseline
    baseline_results = []
    for probe in CANARY_PROBES:
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
            output = generate_with_chunks(
                model, input_ids=chat_ids, max_new_tokens=MAX_NEW_TOKENS
            )

        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        input_text = tokenizer.decode(chat_ids[0], skip_special_tokens=True)
        generated = full_response[len(input_text) :].strip()

        match = exact_match(generated, probe["answer"])
        status = "✓" if match else "✗"
        print(f"  {status} [{probe['id']}] {question}")
        print(f"      Gold: {probe['answer']}")
        print(f"      Got:  {generated[:200]}")

        baseline_results.append(
            {
                "id": probe["id"],
                "question": question,
                "gold_answer": probe["answer"],
                "generated": generated,
                "exact_match": match,
                "category": probe["category"],
            }
        )

    baseline_correct = sum(1 for r in baseline_results if r["exact_match"])
    print(f"\n  Baseline recall: {baseline_correct}/{len(baseline_results)}")
    all_results["canary_baseline"] = {
        "results": baseline_results,
        "correct": baseline_correct,
        "total": len(baseline_results),
    }

    # --- Summary ---
    print("\n" + "=" * 70)
    print("CANARY TEST SUMMARY")
    print("=" * 70)

    canary_recall = all_results["canary"]["correct"] / all_results["canary"]["total"]
    baseline_recall = (
        all_results["canary_baseline"]["correct"]
        / all_results["canary_baseline"]["total"]
    )

    print(
        f"  Canary (with doc):    {all_results['canary']['correct']}/{all_results['canary']['total']} = {canary_recall:.0%}"
    )
    print(
        f"  Canary (no doc):      {baseline_correct}/{len(baseline_results)} = {baseline_recall:.0%}"
    )
    delta = canary_recall - baseline_recall
    print(f"  Delta:                {delta:+.0%}")

    if canary_recall < 0.3:
        print("\n  ⚠️  CONCLUSION: LoRA is NOT functioning as a recall mechanism.")
        print("     The hypernetwork may not be encoding document content at all.")
    elif canary_recall >= 0.5:
        print("\n  ✓  CONCLUSION: LoRA recall works on synthetic docs.")
        print(
            "     Your real documents may have a format mismatch with training distribution."
        )
    else:
        print("\n  ?  CONCLUSION: Partial recall — investigate further.")

    if "sakana_control" in all_results:
        sr = all_results["sakana_control"]
        print(f"\n  Sakana control:       {sr['correct']}/{sr['total']}")

    # --- Save ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"diag3_canary_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
