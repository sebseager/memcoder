"""
Exp 1 — Oracle Ceiling: Capability interference check.

For each oracle LoRA, run a fixed instruction-following probe set before and
after injection, then measure relative degradation.

README criterion: if mean performance drops >15% consistently, add a capability
regularization term before Phase 1.

Usage:
    python capability_interference.py --ids-file pilot_ids.txt
    python capability_interference.py --instance-id django__django-12284
    python capability_interference.py --all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CAPABILITY_DIR, ENABLE_THINKING, ORACLE_LORA_DIR, SEED
from helpers import load_base_model, load_subsets

PROBES = [
    {"prompt": "Reply with exactly: ALPHA", "expected": "ALPHA"},
    {"prompt": "Reply with exactly: BRAVO", "expected": "BRAVO"},
    {"prompt": "Reply with exactly: CHARLIE", "expected": "CHARLIE"},
    {"prompt": "Reply with exactly: DELTA", "expected": "DELTA"},
    {"prompt": "Reply with exactly: ECHO", "expected": "ECHO"},
    {"prompt": "Reply with exactly: FOXTROT", "expected": "FOXTROT"},
    {"prompt": "Reply with exactly: GOLF", "expected": "GOLF"},
    {"prompt": "Reply with exactly: HOTEL", "expected": "HOTEL"},
    {"prompt": "Reply with exactly: INDIA", "expected": "INDIA"},
    {"prompt": "Reply with exactly: JULIET", "expected": "JULIET"},
    {"prompt": "Reply with exactly: KILO", "expected": "KILO"},
    {"prompt": "Reply with exactly: LIMA", "expected": "LIMA"},
    {"prompt": "Reply with exactly: MIKE", "expected": "MIKE"},
    {"prompt": "Reply with exactly: NOVEMBER", "expected": "NOVEMBER"},
    {"prompt": "Reply with exactly: OSCAR", "expected": "OSCAR"},
    {"prompt": "Reply with exactly: PAPA", "expected": "PAPA"},
    {"prompt": "Reply with exactly: QUEBEC", "expected": "QUEBEC"},
    {"prompt": "Reply with exactly: ROMEO", "expected": "ROMEO"},
    {"prompt": "Reply with exactly: SIERRA", "expected": "SIERRA"},
    {"prompt": "Reply with exactly: TANGO", "expected": "TANGO"},
]


SYSTEM_PROMPT = (
    "You are a precise assistant. Follow the user instruction exactly and "
    "output only the requested text."
)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def generate_text(model, tokenizer, user_prompt: str, max_new_tokens: int = 32) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    torch.manual_seed(SEED)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def score_model_on_probes(
    model, tokenizer, probes: list[dict]
) -> tuple[float, list[dict]]:
    details = []
    correct = 0
    for p in probes:
        pred = generate_text(model, tokenizer, p["prompt"])
        ok = normalize_text(pred) == normalize_text(p["expected"])
        correct += int(ok)
        details.append(
            {
                "prompt": p["prompt"],
                "expected": p["expected"],
                "prediction": pred,
                "correct": ok,
            }
        )
    score = correct / len(probes) if probes else 0.0
    return score, details


def main():
    parser = argparse.ArgumentParser(description="Capability interference check")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance-id", type=str)
    group.add_argument("--ids-file", type=str)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.instance_id:
        instance_ids = [args.instance_id]
    elif args.ids_file:
        instance_ids = [
            x for x in Path(args.ids_file).read_text().splitlines() if x.strip()
        ]
    else:
        subsets = load_subsets()
        instance_ids = subsets["constrained_instance_ids"]

    print(f"Checking capability interference on {len(instance_ids)} LoRA adapter(s)")

    base_model, tokenizer = load_base_model()
    base_model.eval()

    print("Scoring baseline model on 20 probes...")
    baseline_score, baseline_details = score_model_on_probes(
        base_model, tokenizer, PROBES
    )
    print(f"  Baseline score: {baseline_score:.1%}")

    peft_model = None
    active_adapter = None
    adapter_rows = []

    for i, iid in enumerate(instance_ids):
        print(f"\n[{i + 1}/{len(instance_ids)}] {iid}")
        lora_path = ORACLE_LORA_DIR / iid
        if not lora_path.exists():
            print(f"  SKIP: missing LoRA at {lora_path}")
            adapter_rows.append(
                {
                    "instance_id": iid,
                    "status": "missing_lora",
                    "baseline_score": baseline_score,
                }
            )
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

        peft_model.eval()
        score, details = score_model_on_probes(peft_model, tokenizer, PROBES)

        if baseline_score > 0:
            drop_pct = (baseline_score - score) / baseline_score * 100.0
        else:
            drop_pct = 0.0

        print(f"  Adapter score: {score:.1%} (drop {drop_pct:.1f}%)")
        adapter_rows.append(
            {
                "instance_id": iid,
                "status": "ok",
                "baseline_score": baseline_score,
                "adapter_score": score,
                "drop_pct": drop_pct,
                "flag_drop_gt_15pct": drop_pct > 15.0,
                "probe_details": details,
            }
        )

    valid_rows = [r for r in adapter_rows if r.get("status") == "ok"]
    mean_drop = (
        sum(r["drop_pct"] for r in valid_rows) / len(valid_rows) if valid_rows else 0.0
    )
    n_flagged = sum(1 for r in valid_rows if r.get("flag_drop_gt_15pct"))

    summary = {
        "n_adapters_requested": len(instance_ids),
        "n_adapters_scored": len(valid_rows),
        "baseline_score": baseline_score,
        "mean_drop_pct": mean_drop,
        "n_flagged_drop_gt_15pct": n_flagged,
        "flag_rate": n_flagged / len(valid_rows) if valid_rows else 0.0,
        "criterion_consistent_gt_15pct": n_flagged == len(valid_rows)
        and len(valid_rows) > 0,
    }

    CAPABILITY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out = {
        "timestamp_utc": ts,
        "summary": summary,
        "baseline_probe_details": baseline_details,
        "per_adapter": adapter_rows,
    }

    out_path = CAPABILITY_DIR / f"capability_interference_{ts}.json"
    latest_path = CAPABILITY_DIR / "latest.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    with open(latest_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Capability Interference Summary ===")
    print(f"  Adapters scored: {summary['n_adapters_scored']}")
    print(f"  Baseline score: {summary['baseline_score']:.1%}")
    print(f"  Mean drop: {summary['mean_drop_pct']:.1f}%")
    print(f"  Flagged (>15% drop): {summary['n_flagged_drop_gt_15pct']}")
    print(f"  Consistent >15% drop: {summary['criterion_consistent_gt_15pct']}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
