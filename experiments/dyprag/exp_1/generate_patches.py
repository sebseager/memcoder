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
import collections
import difflib
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BUDGET_TOKENS,
    ENABLE_THINKING,
    GENERATION_MAX_ATTEMPTS,
    GENERATION_RESERVE_TOKENS,
    GENERATION_TOKEN_SCHEDULE_BD,
    GENERATION_TOKEN_SCHEDULE_C,
    MAX_NEW_TOKENS,
    ORACLE_LORA_DIR,
    PATCHES_DIR,
    SEED,
    TEMPERATURE,
    TOP_P,
)
from helpers import (
    get_file_content_for_instance,
    load_base_model,
    load_subsets,
    load_swebench_dataset,
    load_token_counts,
    truncate_to_budget,
)
from peft import PeftModel
from prompts import (
    SYSTEM_PROMPT_SEARCH_REPLACE,
    SYSTEM_PROMPT_UNIFIED_DIFF,
    make_user_prompt,
)


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

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": TEMPERATURE > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if TEMPERATURE > 0:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated part
    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_raw_with_overrides(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate output with per-attempt decoding overrides."""
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

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def extract_unified_diff(raw_output: str) -> str:
    """Extract a unified diff block from model output, if present."""
    text = raw_output.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\\n", "", text)
        text = re.sub(r"\\n```$", "", text)

    match = re.search(r"(?m)^diff --git a/.* b/.*$", text)
    if not match:
        return ""
    diff = text[match.start() :].strip()
    if not diff.endswith("\n"):
        diff += "\n"
    return diff


def is_noop_unified_diff(patch: str) -> bool:
    """Return True when unified diff contains no effective line change."""
    removed = []
    added = []
    for line in patch.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])

    if not removed and not added:
        return True
    return collections.Counter(removed) == collections.Counter(added)


def choose_attempt_max_new_tokens(condition: str, attempt: int) -> int:
    """Pick max_new_tokens per attempt to avoid immediate runaway generations."""
    if condition in ("B", "D"):
        schedule = GENERATION_TOKEN_SCHEDULE_BD
    elif condition == "C":
        schedule = GENERATION_TOKEN_SCHEDULE_C
    else:
        schedule = [MAX_NEW_TOKENS]

    if attempt < len(schedule):
        return min(schedule[attempt], MAX_NEW_TOKENS)
    return MAX_NEW_TOKENS


def count_exact_search_hits(blocks: list[tuple[str, str]], file_content: str) -> int:
    """Count blocks whose SEARCH text appears exactly in source."""
    return sum(1 for search, _ in blocks if search and search in file_content)


def is_degenerate_raw_output(raw_output: str, blocks: list[tuple[str, str]]) -> bool:
    """Detect low-quality outputs that should trigger another attempt."""
    raw_len = len(raw_output)
    block_count = len(blocks)
    if block_count == 0 and raw_len > 6000:
        return True
    if block_count > 20:
        return True

    if block_count > 0:
        noop = sum(1 for s, r in blocks if s.strip() == r.strip())
        if noop >= 5 and noop / block_count >= 0.8:
            return True

    # Repetitive import flooding seen in runaway cases.
    if raw_output.count("dmp_ground_add") >= 10:
        return True
    return False


def get_context_window_limit(model, tokenizer) -> int:
    """Return an effective input context limit for prompt construction."""
    model_limit = getattr(model.config, "max_position_embeddings", None)
    tok_limit = getattr(tokenizer, "model_max_length", None)

    candidates = [
        x for x in (model_limit, tok_limit) if isinstance(x, int) and 0 < x < 1_000_000
    ]
    if not candidates:
        return 32768
    return min(candidates)


def estimate_allowed_context_tokens(
    condition: str,
    model,
    tokenizer,
    problem_statement: str,
    file_path: str,
    system_prompt: str,
) -> tuple[int, int, int]:
    """Estimate max file-content tokens that fit in the prompt."""
    ctx_limit = get_context_window_limit(model, tokenizer)
    empty_prompt = make_user_prompt(problem_statement, "", file_path)
    probe_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": empty_prompt},
    ]
    overhead_tokens = len(
        tokenizer.apply_chat_template(
            probe_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
    )
    budget_by_model = max(256, ctx_limit - GENERATION_RESERVE_TOKENS - overhead_tokens)
    if condition in ("B", "D"):
        return min(BUDGET_TOKENS, budget_by_model), ctx_limit, overhead_tokens
    return budget_by_model, ctx_limit, overhead_tokens


# ---------------------------------------------------------------------------
# SEARCH/REPLACE → unified-diff conversion
# ---------------------------------------------------------------------------


def parse_search_replace_blocks(raw_output: str) -> list[tuple[str, str]]:
    """Parse <<<< SEARCH / ==== / >>>> REPLACE blocks from model output.

    Returns a list of (search_text, replace_text) tuples.
    """
    blocks = []
    lines = raw_output.split("\n")

    def _valid_block(search: str, replace: str) -> bool:
        if not search.strip() or not replace.strip():
            return False
        if search.strip() == replace.strip():
            return False
        bad_markers = ("<<<< SEARCH", ">>>> REPLACE", "====")
        if any(m in search for m in bad_markers):
            return False
        if any(m in replace for m in bad_markers):
            return False
        return True

    i = 0
    while i < len(lines):
        if lines[i].strip() == "<<<< SEARCH":
            search_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() not in ("====", ">>>> REPLACE"):
                search_lines.append(lines[i])
                i += 1

            if i >= len(lines):
                break

            # Canonical form: SEARCH ... ==== ... >>>> REPLACE
            if lines[i].strip() == "====":
                i += 1  # skip ====
                replace_lines = []
                while i < len(lines) and lines[i].strip() not in (
                    ">>>> REPLACE",
                    "<<<< SEARCH",
                ):
                    replace_lines.append(lines[i])
                    i += 1
                # Require a closing REPLACE delimiter for canonical blocks.
                if i < len(lines) and lines[i].strip() == ">>>> REPLACE":
                    i += 1  # skip >>>> REPLACE
                    search = "\n".join(search_lines)
                    replace = "\n".join(replace_lines)
                    if _valid_block(search, replace):
                        blocks.append((search, replace))
                continue

            # Fallback form seen in practice: SEARCH ... >>>> REPLACE ...
            if lines[i].strip() == ">>>> REPLACE":
                i += 1  # skip >>>> REPLACE
                replace_lines = []
                while i < len(lines) and lines[i].strip() != "<<<< SEARCH":
                    replace_lines.append(lines[i])
                    i += 1
                search = "\n".join(search_lines)
                replace = "\n".join(replace_lines)
                if _valid_block(search, replace):
                    blocks.append((search, replace))
                continue
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

    def apply_single_block(text: str, search: str, replace: str) -> tuple[str, bool]:
        # Fast path: exact substring match.
        if search and search in text:
            return text.replace(search, replace, 1), True

        # Fallback: match by stripped line content to recover from whitespace drift.
        mod_lines = text.splitlines()
        search_lines = search.splitlines()
        replace_lines = replace.splitlines()
        if not search_lines:
            return text, False

        n = len(search_lines)
        stripped_mod = [ln.strip() for ln in mod_lines]
        stripped_search = [ln.strip() for ln in search_lines]
        for i in range(0, len(mod_lines) - n + 1):
            if stripped_mod[i : i + n] == stripped_search:
                new_lines = mod_lines[:i] + replace_lines + mod_lines[i + n :]
                return "\n".join(new_lines), True

        return text, False

    modified = file_content
    applied = 0
    for search, replace in blocks:
        if not search.strip() or search.strip() == replace.strip():
            continue
        modified, ok = apply_single_block(modified, search, replace)
        if ok:
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
    peft_model = None
    active_adapter = None

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
        prompt_system = SYSTEM_PROMPT_SEARCH_REPLACE
        context_tokens = 0
        context_budget_used = 0
        context_window_limit = 0
        context_was_truncated = False
        attempts_used = 0
        attempt_max_new_tokens = 0

        if condition == "A":
            # Condition A is a strict floor: no source context, no LoRA.
            prompt_system = SYSTEM_PROMPT_UNIFIED_DIFF
            user_prompt = ""
        elif condition in ("B", "D"):
            full_tokens = len(tokenizer.encode(file_content, add_special_tokens=False))
            allowed_tokens, context_window_limit, _ = estimate_allowed_context_tokens(
                condition=condition,
                model=base_model,
                tokenizer=tokenizer,
                problem_statement=problem_statement,
                file_path=fpath,
                system_prompt=prompt_system,
            )
            prompt_content = truncate_to_budget(file_content, tokenizer, allowed_tokens)
            context_tokens = len(
                tokenizer.encode(prompt_content, add_special_tokens=False)
            )
            context_budget_used = allowed_tokens
            context_was_truncated = context_tokens < full_tokens
            user_prompt = make_user_prompt(problem_statement, prompt_content, fpath)
        else:
            # condition C: full content, no truncation
            full_tokens = len(tokenizer.encode(file_content, add_special_tokens=False))
            allowed_tokens, context_window_limit, _ = estimate_allowed_context_tokens(
                condition=condition,
                model=base_model,
                tokenizer=tokenizer,
                problem_statement=problem_statement,
                file_path=fpath,
                system_prompt=prompt_system,
            )
            prompt_content = truncate_to_budget(file_content, tokenizer, allowed_tokens)
            context_tokens = len(
                tokenizer.encode(prompt_content, add_special_tokens=False)
            )
            context_budget_used = allowed_tokens
            context_was_truncated = context_tokens < full_tokens
            if context_was_truncated:
                print(
                    "    WARN: Condition C context clipped by model limit "
                    f"({full_tokens} -> {context_tokens} tokens)"
                )
            user_prompt = make_user_prompt(problem_statement, prompt_content, fpath)

        # Load oracle LoRA for condition D
        if needs_lora:
            lora_path = ORACLE_LORA_DIR / iid
            if not lora_path.exists():
                print(f"    SKIP: no oracle LoRA at {lora_path}")
                skipped.append(iid)
                continue
            if peft_model is None:
                active_adapter = "default"
                peft_model = PeftModel.from_pretrained(
                    base_model,
                    str(lora_path),
                    adapter_name=active_adapter,
                )
            else:
                new_adapter = f"adapter_{iid}"
                peft_model.load_adapter(str(lora_path), adapter_name=new_adapter)
                peft_model.set_adapter(new_adapter)
                if active_adapter and active_adapter in peft_model.peft_config:
                    peft_model.delete_adapter(active_adapter)
                active_adapter = new_adapter
            model = peft_model
            model.eval()
        else:
            model = base_model
            model.eval()

        raw_output = ""
        patch = ""
        blocks = []
        attempt_trace = []
        t0 = time.time()
        if condition != "A":
            for attempt in range(GENERATION_MAX_ATTEMPTS):
                attempts_used = attempt + 1
                attempt_max_new_tokens = choose_attempt_max_new_tokens(
                    condition, attempt
                )
                # Keep all attempts in sampling mode; avoids hidden greedy fallbacks.
                attempt_temperature = max(0.05, TEMPERATURE - (0.05 * attempt))
                torch.manual_seed(SEED + attempt)

                trial_raw = generate_raw_with_overrides(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=prompt_system,
                    user_prompt=user_prompt,
                    max_new_tokens=attempt_max_new_tokens,
                    temperature=attempt_temperature,
                    top_p=TOP_P,
                )
                trial_blocks = parse_search_replace_blocks(trial_raw)
                exact_hits = (
                    count_exact_search_hits(trial_blocks, file_content)
                    if file_content is not None and trial_blocks
                    else 0
                )
                rejection_reason = ""

                if is_degenerate_raw_output(trial_raw, trial_blocks):
                    rejection_reason = "degenerate_output"
                    attempt_trace.append(
                        {
                            "attempt": attempts_used,
                            "max_new_tokens": attempt_max_new_tokens,
                            "temperature": attempt_temperature,
                            "raw_output_len": len(trial_raw),
                            "n_blocks": len(trial_blocks),
                            "exact_search_hits": exact_hits,
                            "accepted": False,
                            "reason": rejection_reason,
                        }
                    )
                    raw_output = trial_raw
                    blocks = trial_blocks
                    continue

                trial_patch = ""
                if trial_blocks and file_content is not None:
                    if exact_hits == 0:
                        rejection_reason = "no_exact_search_hit"
                        attempt_trace.append(
                            {
                                "attempt": attempts_used,
                                "max_new_tokens": attempt_max_new_tokens,
                                "temperature": attempt_temperature,
                                "raw_output_len": len(trial_raw),
                                "n_blocks": len(trial_blocks),
                                "exact_search_hits": exact_hits,
                                "accepted": False,
                                "reason": rejection_reason,
                            }
                        )
                        raw_output = trial_raw
                        blocks = trial_blocks
                        continue
                    trial_patch = search_replace_to_diff(
                        trial_blocks, file_content, fpath
                    )

                # Last-resort fallback for direct unified-diff output.
                if not trial_patch:
                    trial_patch = extract_unified_diff(trial_raw)
                    if not trial_patch:
                        rejection_reason = "no_patch_extracted"

                if trial_patch and is_noop_unified_diff(trial_patch):
                    trial_patch = ""
                    rejection_reason = "noop_patch"

                # For B/C/D, only accept patches targeting the current file path.
                if (
                    trial_patch
                    and fpath is not None
                    and f"a/{fpath}" not in trial_patch
                ):
                    trial_patch = ""
                    rejection_reason = "wrong_file_path"

                raw_output = trial_raw
                patch = trial_patch
                blocks = trial_blocks
                attempt_trace.append(
                    {
                        "attempt": attempts_used,
                        "max_new_tokens": attempt_max_new_tokens,
                        "temperature": attempt_temperature,
                        "raw_output_len": len(trial_raw),
                        "n_blocks": len(trial_blocks),
                        "exact_search_hits": exact_hits,
                        "accepted": bool(patch),
                        "reason": "accepted"
                        if patch
                        else (rejection_reason or "rejected"),
                    }
                )
                if patch:
                    break
        gen_time = time.time() - t0

        predictions.append(
            {
                "instance_id": iid,
                "model_name_or_path": f"dyprag_exp1_condition_{condition}",
                "model_patch": patch,
                "raw_output": raw_output,
                "generation_time_s": gen_time,
                "generation_attempts": attempts_used,
                "attempt_max_new_tokens": attempt_max_new_tokens,
                "attempt_trace": attempt_trace,
                "context_tokens": context_tokens,
                "context_budget_tokens": context_budget_used,
                "context_window_limit": context_window_limit,
                "context_was_truncated": context_was_truncated,
            }
        )

        n_blocks = len(blocks)
        print(
            f"    Generated {n_blocks} block(s), "
            f"diff {len(patch)} chars in {gen_time:.1f}s"
        )

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
