from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from config import (
    COMPLETIONS_DIR,
    CONDITIONS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    ORACLE_LORA_DIR,
    TEMPERATURE,
    TOP_P,
    TRUNCATION_BUDGET_TOKENS,
)
from helpers import (
    generate_text,
    load_instances,
    load_model_and_tokenizer,
    make_chat_prompt,
    make_file_key,
    normalize_body_prediction,
    select_instances,
    truncate_to_budget,
)
from peft import PeftModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Stage 1 completions for B/C/D")
    p.add_argument("--condition", required=True, choices=CONDITIONS)
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--instance-ids-file", default=None)
    p.add_argument("--max-instances", type=int, default=None)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--top-p", type=float, default=TOP_P)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--trunc-budget", type=int, default=TRUNCATION_BUDGET_TOKENS)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def filter_instances_by_ids(instances: list[dict], ids_file: str | None) -> list[dict]:
    if not ids_file:
        return instances

    wanted_ids = [
        line.strip()
        for line in Path(ids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {row["instance_id"]: row for row in instances}
    filtered = [by_id[iid] for iid in wanted_ids if iid in by_id]
    return filtered


def adapter_base_model_id(adapter_path: Path) -> str | None:
    cfg = adapter_path / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data.get("base_model_name_or_path")


def maybe_load_adapter(
    model,
    adapter_state: dict,
    adapter_path: Path,
    expected_model_id: str,
):
    if not adapter_path.exists():
        return None, "missing_adapter"

    adapter_base = adapter_base_model_id(adapter_path)
    if adapter_base and adapter_base != expected_model_id:
        return None, f"incompatible_adapter_base_model:{adapter_base}"

    active = adapter_state.get("active")
    peft_model = adapter_state.get("peft")
    if peft_model is None:
        peft_model = PeftModel.from_pretrained(
            model, str(adapter_path), adapter_name="default"
        )
        adapter_state["peft"] = peft_model
        adapter_state["active"] = "default"
        return peft_model, "loaded_initial"

    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", adapter_path.name)
    new_name = f"adapter_{safe_name}"
    peft_model.load_adapter(str(adapter_path), adapter_name=new_name)
    peft_model.set_adapter(new_name)
    if active and active in peft_model.peft_config:
        peft_model.delete_adapter(active)
    adapter_state["active"] = new_name
    return peft_model, "swapped"


def main() -> int:
    args = parse_args()
    COMPLETIONS_DIR.mkdir(parents=True, exist_ok=True)

    out_jsonl = COMPLETIONS_DIR / f"condition_{args.condition}.jsonl"
    if out_jsonl.exists() and not args.force:
        print(f"Output exists, use --force to overwrite: {out_jsonl}")
        return 1

    instances = load_instances()
    instances = filter_instances_by_ids(instances, args.instance_ids_file)
    instances = select_instances(instances, args.max_instances)
    print(f"Generating condition {args.condition} for {len(instances)} instances")

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, use_4bit=True)

    adapter_state = {"peft": None, "active": None}
    rows = []

    for i, inst in enumerate(instances):
        iid = inst["instance_id"]
        file_key = make_file_key(inst["repo"], inst["file_path"])
        full_text = inst["full_file"]

        if args.condition in ("B", "D"):
            context = truncate_to_budget(full_text, tokenizer, args.trunc_budget)
        else:
            context = full_text

        model_for_run = model
        adapter_status = "not_used"
        if args.condition == "D":
            adapter_path = ORACLE_LORA_DIR / file_key
            model_for_run, adapter_status = maybe_load_adapter(
                model,
                adapter_state,
                adapter_path,
                expected_model_id=args.model_id,
            )
            if model_for_run is None:
                rows.append(
                    {
                        "instance_id": iid,
                        "condition": args.condition,
                        "file_key": file_key,
                        "status": "skipped",
                        "reason": adapter_status,
                    }
                )
                print(f"[{i + 1}/{len(instances)}] {iid} skip {adapter_status}")
                continue

        sys_prompt, user_prompt = make_chat_prompt(
            masked_function=inst["masked_function"],
            context_text=context,
            function_name=inst["function_name"],
        )

        t0 = time.perf_counter()
        raw = generate_text(
            model=model_for_run,
            tokenizer=tokenizer,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        pred_body = normalize_body_prediction(raw)
        generated_token_count = len(tokenizer.encode(raw, add_special_tokens=False))
        elapsed = time.perf_counter() - t0

        rows.append(
            {
                "instance_id": iid,
                "condition": args.condition,
                "file_key": file_key,
                "status": "ok",
                "repo": inst["repo"],
                "file_path": inst["file_path"],
                "function_name": inst["function_name"],
                "raw_output": raw,
                "predicted_body": pred_body,
                "ground_truth_body": inst["ground_truth_body"],
                "masked_function": inst["masked_function"],
                "context_token_count": len(
                    tokenizer.encode(context, add_special_tokens=False)
                ),
                "generated_token_count": generated_token_count,
                "hit_max_new_tokens": int(generated_token_count >= args.max_new_tokens),
                "adapter_status": adapter_status,
                "generation_time_s": elapsed,
            }
        )
        print(f"[{i + 1}/{len(instances)}] {iid} done in {elapsed:.1f}s")

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "condition": args.condition,
        "model_id": args.model_id,
        "n_instances_requested": len(instances),
        "n_rows_written": len(rows),
        "output_jsonl": str(out_jsonl),
    }
    with (COMPLETIONS_DIR / f"condition_{args.condition}.meta.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"Saved completions: {out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
