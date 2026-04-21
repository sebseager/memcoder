"""
Tiny-scale smoke test for Exp 1 pipeline.

Runs 2 instances through oracle training → generation (conditions B & D)
to verify the adapter lifecycle and basic pipeline correctness.
Does NOT run SWE-bench Docker evaluation — just checks model behavior.

Usage:
    python test_tiny.py
    python test_tiny.py --n 1          # even smaller: single instance
    python test_tiny.py --skip-train   # reuse existing oracle LoRAs
"""

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BUDGET_TOKENS,
    ENABLE_THINKING,
    ORACLE_LORA_DIR,
)
from generate_patches import parse_search_replace_blocks, search_replace_to_diff
from helpers import (
    get_file_content_for_instance,
    load_base_model,
    load_swebench_dataset,
    load_token_counts,
    truncate_to_budget,
)
from prompts import SYSTEM_PROMPT_SEARCH_REPLACE, make_user_prompt
from train_oracle import train_one_oracle

TINY_DIR = Path(__file__).resolve().parent / "results" / "tiny"


def generate_one(model, tokenizer, system_prompt, user_prompt, max_tokens=512):
    """Short generation for smoke testing (capped at 512 tokens)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Tiny-scale smoke test")
    parser.add_argument("--n", type=int, default=2, help="Number of instances")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--keep-loras", action="store_true", help="Don't clean up tiny test LoRAs"
    )
    args = parser.parse_args()

    token_counts = load_token_counts()
    swebench_data = load_swebench_dataset()

    # Use first N instances from pilot_ids.txt (known to work)
    pilot_file = Path(__file__).resolve().parent / "pilot_ids.txt"
    pilot_ids = pilot_file.read_text().strip().split("\n")
    test_ids = pilot_ids[: args.n]

    print(f"=== Tiny-scale test: {len(test_ids)} instance(s) ===")
    for iid in test_ids:
        print(f"  {iid}")

    TINY_DIR.mkdir(parents=True, exist_ok=True)
    base_model = None
    tokenizer = None

    # ------------------------------------------------------------------
    # Step 1: Train oracle LoRAs
    # ------------------------------------------------------------------
    if not args.skip_train:
        print("\n--- Step 1: Train oracle LoRAs ---")
        base_model, tokenizer = load_base_model()

        for i, iid in enumerate(test_ids):
            lora_dir = TINY_DIR / "oracle_loras" / iid
            if lora_dir.exists():
                shutil.rmtree(lora_dir)

            fpath, content = get_file_content_for_instance(
                iid, token_counts, swebench_data
            )
            n_tok = len(tokenizer.encode(content, add_special_tokens=False))
            print(f"\n  [{i + 1}/{len(test_ids)}] {iid}: {fpath} ({n_tok} tokens)")

            # Capture warnings to detect stacking
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                meta, base_model = train_one_oracle(
                    iid, content, base_model, tokenizer, lora_dir
                )

            stacking_warns = [
                w
                for w in caught
                if "second time" in str(w.message)
                or "multiple adapters" in str(w.message)
            ]
            if stacking_warns:
                print(f"  ** FAIL: adapter stacking warnings on instance {i + 1}! **")
                for w in stacking_warns:
                    print(f"     {w.message}")
            else:
                print(f"  OK: no stacking warnings (instance {i + 1})")

            train_loss = meta.get("final_train_loss")
            eval_loss = meta.get("best_eval_loss")
            print(
                f"  train loss={train_loss:.4f}"
                if isinstance(train_loss, (int, float))
                else "  train loss=NA"
            )
            print(
                f"  eval loss={eval_loss:.4f}"
                if isinstance(eval_loss, (int, float))
                else "  eval loss=NA"
            )

        # Reuse the same base weights for generation to avoid a second large load.
        if hasattr(base_model, "unload"):
            base_model = base_model.unload()
        torch.cuda.empty_cache()
    else:
        print("\n--- Step 1: Skipping training (--skip-train) ---")

    # ------------------------------------------------------------------
    # Step 2: Generate for conditions B and D
    # ------------------------------------------------------------------
    print("\n--- Step 2: Generate patches (B and D) ---")
    if base_model is None or tokenizer is None:
        base_model, tokenizer = load_base_model()

    results = {}
    peft_model = None
    for cond in ["B", "D"]:
        results[cond] = []
        print(f"\n  Condition {cond}:")

        for i, iid in enumerate(test_ids):
            row = swebench_data[iid]
            fpath, content = get_file_content_for_instance(
                iid, token_counts, swebench_data
            )
            trunc = truncate_to_budget(content, tokenizer, BUDGET_TOKENS)
            user_prompt = make_user_prompt(row["problem_statement"], trunc, fpath)

            if cond == "D":
                lora_path = TINY_DIR / "oracle_loras" / iid
                if not lora_path.exists():
                    # Fall back to main oracle_loras
                    lora_path = ORACLE_LORA_DIR / iid
                if not lora_path.exists():
                    print(f"    [{i + 1}] {iid}: SKIP (no LoRA)")
                    continue

                from peft import PeftModel

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if peft_model is None:
                        peft_model = PeftModel.from_pretrained(
                            base_model, str(lora_path)
                        )
                    else:
                        peft_model.delete_adapter("default")
                        peft_model.load_adapter(str(lora_path), adapter_name="default")
                        peft_model.set_adapter("default")

                stacking_warns = [
                    w
                    for w in caught
                    if "second time" in str(w.message)
                    or "multiple adapters" in str(w.message)
                ]
                if stacking_warns:
                    print(f"    ** FAIL: adapter stacking on D[{i + 1}]! **")
                    for w in stacking_warns:
                        print(f"       {w.message}")
                else:
                    print(f"    OK: no stacking warnings (D[{i + 1}])")

                model = peft_model
                model.eval()
            else:
                model = base_model
                model.eval()

            t0 = time.time()
            raw = generate_one(
                model, tokenizer, SYSTEM_PROMPT_SEARCH_REPLACE, user_prompt
            )
            elapsed = time.time() - t0

            # Parse SEARCH/REPLACE blocks and convert to diff
            blocks = parse_search_replace_blocks(raw)
            if blocks:
                diff = search_replace_to_diff(blocks, content, fpath)
            else:
                diff = ""

            results[cond].append(
                {
                    "instance_id": iid,
                    "raw_output": raw,
                    "diff": diff,
                    "n_blocks": len(blocks),
                    "generation_time_s": elapsed,
                }
            )

            print(
                f"    [{i + 1}] {iid}: {len(blocks)} block(s), "
                f"diff {len(diff)} chars, {elapsed:.1f}s"
            )
            print(f"        first 200: {repr(raw[:200])}")

    # ------------------------------------------------------------------
    # Step 3: Compare B vs D
    # ------------------------------------------------------------------
    print("\n--- Step 3: B vs D comparison ---")
    b_map = {r["instance_id"]: r for r in results["B"]}
    d_map = {r["instance_id"]: r for r in results["D"]}
    for iid in test_ids:
        b_row = b_map.get(iid, {})
        d_row = d_map.get(iid, {})
        b_out = b_row.get("raw_output", "")
        d_out = d_row.get("raw_output", "")
        same = b_out == d_out
        b_diff = b_row.get("diff", "")
        d_diff = d_row.get("diff", "")
        print(f"  {iid}:")
        print(
            f"    B: {len(b_out)} chars raw, {len(b_diff)} chars diff  |  "
            f"D: {len(d_out)} chars raw, {len(d_diff)} chars diff  |  "
            f"identical_raw={same}"
        )
        if same:
            print(
                "    ** WARNING: D output identical to B — LoRA may not be loading **"
            )

    # Save results (including raw_output for debugging)
    with open(TINY_DIR / "tiny_results.json", "w") as f:
        json.dump(
            {"test_ids": test_ids, "results": results},
            f,
            indent=2,
        )

    print(f"\nResults saved to {TINY_DIR / 'tiny_results.json'}")
    print("\n=== Tiny test complete ===")

    del base_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
