"""
Exp 1 — Oracle Ceiling: Generate patches for all conditions.

For each instance × condition, generates a unified-diff patch using Qwen3-8B.

Conditions:
  A — No context, no LoRA
  B — Truncated context (≤ budget), no LoRA
  C — Full context (no truncation), no LoRA
  D — Truncated context (≤ budget) + oracle LoRA

Usage:
    python generate_patches.py --condition B --ids-file pilot_ids.txt
    python generate_patches.py --condition D --all
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BUDGET_TOKENS,
    ENABLE_THINKING,
    MAX_NEW_TOKENS,
    ORACLE_LORA_DIR,
    PATCHES_DIR,
    TEMPERATURE,
)
from helpers import (
    get_file_content_for_instance,
    load_base_model,
    load_subsets,
    load_swebench_dataset,
    load_token_counts,
    truncate_to_budget,
)
from prompts import SYSTEM_PROMPT, make_user_prompt


def generate_patch(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Generate a patch from the model. Returns raw model output."""
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
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            top_p=None,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def extract_diff_from_output(raw_output: str) -> str:
    """Try to extract a unified diff from model output.

    The model may wrap the diff in markdown code fences or add commentary.
    """
    # Try to find diff blocks
    lines = raw_output.split("\n")
    diff_lines = []
    in_diff = False

    for line in lines:
        if line.startswith("---") or line.startswith("diff --git"):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
        # Stop if we hit a non-diff line after collecting some diff
        if in_diff and not line.strip() and len(diff_lines) > 3:
            # Check if next meaningful content is still diff
            pass

    if diff_lines:
        return "\n".join(diff_lines)

    # Fallback: strip markdown code fences
    if "```" in raw_output:
        parts = raw_output.split("```")
        for part in parts[1::2]:  # odd-indexed parts are inside fences
            stripped = part.strip()
            if stripped.startswith("diff"):
                stripped = stripped[4:].strip()
            if "---" in stripped or "@@" in stripped:
                return stripped

    return raw_output


def run_condition(
    condition: str,
    instance_ids: list[str],
    token_counts: list[dict],
    swebench_data: dict,
    output_path: Path,
):
    """Generate patches for one condition across all given instances."""
    needs_lora = condition == "D"

    print(f"\nLoading model for condition {condition}...")
    if needs_lora:
        # We'll load per-instance LoRA; start with base model
        base_model, tokenizer = load_base_model()
    else:
        base_model, tokenizer = load_base_model()

    predictions = []
    skipped = []

    for i, iid in enumerate(instance_ids):
        print(f"\n  [{i + 1}/{len(instance_ids)}] {iid} (condition {condition})")

        row = swebench_data[iid]
        problem_statement = row["problem_statement"]

        # Determine context
        if condition == "A":
            user_prompt = make_user_prompt(problem_statement)
        else:
            fpath, content = get_file_content_for_instance(
                iid, token_counts, swebench_data
            )

            if condition in ("B", "D"):
                content = truncate_to_budget(content, tokenizer, BUDGET_TOKENS)
            # condition C: full content, no truncation

            user_prompt = make_user_prompt(problem_statement, content, fpath)

        # Load oracle LoRA for condition D
        if needs_lora:
            lora_path = ORACLE_LORA_DIR / iid
            if not lora_path.exists():
                print(f"    SKIP: no oracle LoRA at {lora_path}")
                skipped.append(iid)
                continue
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, str(lora_path))
            model.eval()
        else:
            model = base_model
            model.eval()

        t0 = time.time()
        raw_output = generate_patch(model, tokenizer, SYSTEM_PROMPT, user_prompt)
        gen_time = time.time() - t0

        patch = extract_diff_from_output(raw_output)

        predictions.append(
            {
                "instance_id": iid,
                "model_name_or_path": f"dyprag_exp1_condition_{condition}",
                "model_patch": patch,
                "raw_output": raw_output,
                "generation_time_s": gen_time,
            }
        )

        print(f"    Generated {len(patch)} chars in {gen_time:.1f}s")

        # Remove LoRA adapter to prevent stacking on the next iteration.
        # PeftModel.from_pretrained() attaches adapters to base_model in-place;
        # delete_adapter("default") undoes that so the next instance starts clean.
        if needs_lora:
            model.delete_adapter("default")
            del model
            torch.cuda.empty_cache()

    # Save predictions (JSONL for swebench compatibility)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # Also save a clean version for swebench eval (only required keys)
    eval_path = output_path.with_suffix(".eval.jsonl")
    with open(eval_path, "w") as f:
        for pred in predictions:
            f.write(
                json.dumps(
                    {
                        "instance_id": pred["instance_id"],
                        "model_name_or_path": pred["model_name_or_path"],
                        "model_patch": pred["model_patch"],
                    }
                )
                + "\n"
            )

    print(f"\nSaved {len(predictions)} predictions to {output_path}")
    if skipped:
        print(f"Skipped {len(skipped)} instances: {skipped}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate patches for Exp 1")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C", "D"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance-id", type=str)
    group.add_argument("--ids-file", type=str)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.instance_id:
        instance_ids = [args.instance_id]
    elif args.ids_file:
        instance_ids = Path(args.ids_file).read_text().strip().split("\n")
    else:
        subsets = load_subsets()
        instance_ids = subsets["constrained_instance_ids"]

    token_counts = load_token_counts()
    swebench_data = load_swebench_dataset()

    output_path = PATCHES_DIR / f"condition_{args.condition}.jsonl"

    run_condition(
        condition=args.condition,
        instance_ids=instance_ids,
        token_counts=token_counts,
        swebench_data=swebench_data,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
