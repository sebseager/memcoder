from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

import pandas as pd
from config import (
    CAPABILITY_DIR,
    CAPABILITY_DROP_THRESHOLD_PCT,
    MODEL_ID,
    ORACLE_LORA_DIR,
)
from helpers import generate_text, load_model_and_tokenizer
from peft import PeftModel

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


SYSTEM_PROMPT = "You are a precise assistant. Follow the user instruction exactly and output only the requested text."


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def score_model(model, tokenizer) -> tuple[float, list[dict]]:
    details = []
    correct = 0
    for probe in PROBES:
        pred = generate_text(
            model=model,
            tokenizer=tokenizer,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=probe["prompt"],
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )
        ok = normalize(pred) == normalize(probe["expected"])
        correct += int(ok)
        details.append(
            {
                "prompt": probe["prompt"],
                "expected": probe["expected"],
                "prediction": pred,
                "correct": ok,
            }
        )
    return correct / len(PROBES), details


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 capability interference check")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--max-adapters", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    CAPABILITY_DIR.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_id=args.model_id, use_4bit=True)
    baseline_score, baseline_details = score_model(model, tokenizer)
    print(f"Baseline score: {baseline_score:.1%}")

    adapter_dirs = sorted(
        [
            p
            for p in ORACLE_LORA_DIR.iterdir()
            if p.is_dir() and (p / "adapter_config.json").exists()
        ],
        key=lambda x: x.name,
    )
    if args.max_adapters is not None:
        adapter_dirs = adapter_dirs[: args.max_adapters]

    rows = []
    peft_model = None
    active = None

    for idx, adir in enumerate(adapter_dirs):
        file_key = adir.name
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                model, str(adir), adapter_name="default"
            )
            active = "default"
        else:
            safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", file_key)
            new_name = f"adapter_{safe_name}"
            peft_model.load_adapter(str(adir), adapter_name=new_name)
            peft_model.set_adapter(new_name)
            if active and active in peft_model.peft_config:
                peft_model.delete_adapter(active)
            active = new_name

        score, probe_details = score_model(peft_model, tokenizer)
        drop_pct = (
            0.0
            if baseline_score <= 0
            else ((baseline_score - score) / baseline_score) * 100.0
        )
        flagged = drop_pct > CAPABILITY_DROP_THRESHOLD_PCT

        rows.append(
            {
                "file_key": file_key,
                "baseline_score": baseline_score,
                "adapter_score": score,
                "drop_pct": drop_pct,
                "flag_drop_gt_threshold": flagged,
                "probe_details": probe_details,
            }
        )
        print(
            f"[{idx + 1}/{len(adapter_dirs)}] {file_key}: score={score:.1%}, drop={drop_pct:.1f}%"
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "timestamp_utc": ts,
        "n_adapters": len(rows),
        "baseline_score": baseline_score,
        "threshold_pct": CAPABILITY_DROP_THRESHOLD_PCT,
        "n_flagged": sum(int(r["flag_drop_gt_threshold"]) for r in rows),
        "baseline_probe_details": baseline_details,
        "per_adapter": rows,
    }

    with (CAPABILITY_DIR / "latest.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with (CAPABILITY_DIR / f"capability_interference_{ts}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(out, f, indent=2)

    flat = [
        {
            "file_key": r["file_key"],
            "baseline_score": r["baseline_score"],
            "adapter_score": r["adapter_score"],
            "drop_pct": r["drop_pct"],
            "flag_drop_gt_threshold": r["flag_drop_gt_threshold"],
        }
        for r in rows
    ]
    pd.DataFrame(flat).to_csv(CAPABILITY_DIR / "latest.csv", index=False)

    print("Saved capability report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
