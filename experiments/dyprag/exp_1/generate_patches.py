"""
Exp 1 — Oracle Ceiling: Generate patches for all conditions.

For each instance × condition, generates a SEARCH/REPLACE patch using Qwen3-8B,
then converts the result to a unified diff for SWE-bench evaluation.

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
import difflib
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
    unload_peft,
)
from prompts import SYSTEM_PROMPT, make_user_prompt


def generate_raw(model, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Generate a response from the model. Returns raw model output."""
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


# ---------------------------------------------------------------------------
# SEARCH/REPLACE → unified-diff conversion
# ---------------------------------------------------------------------------


def parse_search_replace_blocks(raw_output: str) -> list[tuple[str, str]]:
    """Parse <<<< SEARCH / ==== / >>>> REPLACE blocks from model output.

    Returns a list of (search_text, replace_text) tuples.
    """
    blocks = []
    lines = raw_output.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip() == "<<<< SEARCH":
            search_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() != "====":
                search_lines.append(lines[i])
                i += 1
            i += 1  # skip ====
            replace_lines = []
            while i < len(lines) and lines[i].strip() != ">>>> REPLACE":
                replace_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip >>>> REPLACE
            blocks.append(("\n".join(search_lines), "\n".join(replace_lines)))
        else:
            i += 1
    return blocks


def search_replace_to_diff(
    blocks: list[tuple[str, str]], file_content: str, file_path: str
) -> str:
    """Apply SEARCH/REPLACE blocks to file content and produce a unified diff.

    Each SEARCH block is located in the file by exact substring match.  If the
    SEARCH text is not found, that block is skipped (the model hallucinated).
    """
    modified = file_content
    applied = 0
    for search, replace in blocks:
        if search and search in modified:
            modified = modified.replace(search, replace, 1)
            applied += 1

    if applied == 0:
        return ""

    orig_lines = file_content.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    # Ensure files end with newline for clean diffs
    if orig_lines and not orig_lines[-1].endswith("\n"):
        orig_lines[-1] += "\n"
    if mod_lines and not mod_lines[-1].endswith("\n"):
        mod_lines[-1] += "\n"

    diff = difflib.unified_diff(
        orig_lines,
        mod_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    return "".join(diff)


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
    base_model, tokenizer = load_base_model()

    predictions = []
    skipped = []

    for i, iid in enumerate(instance_ids):
        print(f"\n  [{i + 1}/{len(instance_ids)}] {iid} (condition {condition})")

        row = swebench_data[iid]
        problem_statement = row["problem_statement"]

        # Get file content (needed for prompt and SEARCH/REPLACE → diff conversion)
        file_content = None
        fpath = None
        if condition != "A":
            fpath, file_content = get_file_content_for_instance(
                iid, token_counts, swebench_data
            )

        # Determine prompt context
        if condition == "A":
            user_prompt = make_user_prompt(problem_statement)
        elif condition in ("B", "D"):
            prompt_content = truncate_to_budget(file_content, tokenizer, BUDGET_TOKENS)
            user_prompt = make_user_prompt(problem_statement, prompt_content, fpath)
        else:
            # condition C: full content, no truncation
            user_prompt = make_user_prompt(problem_statement, file_content, fpath)

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
        raw_output = generate_raw(model, tokenizer, SYSTEM_PROMPT, user_prompt)
        gen_time = time.time() - t0

        # Convert SEARCH/REPLACE blocks to unified diff
        blocks = parse_search_replace_blocks(raw_output)
        if blocks and file_content is not None:
            patch = search_replace_to_diff(blocks, file_content, fpath)
        else:
            patch = ""

        predictions.append(
            {
                "instance_id": iid,
                "model_name_or_path": f"dyprag_exp1_condition_{condition}",
                "model_patch": patch,
                "raw_output": raw_output,
                "generation_time_s": gen_time,
            }
        )

        n_blocks = len(blocks)
        print(
            f"    Generated {n_blocks} block(s), "
            f"diff {len(patch)} chars in {gen_time:.1f}s"
        )

        # Fully unload LoRA so base_model is clean for the next instance.
        if needs_lora:
            unload_peft(model)
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
