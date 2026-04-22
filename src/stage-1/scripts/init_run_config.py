from __future__ import annotations

import argparse

from config import (
    MAX_NEW_TOKENS,
    MODEL_ID,
    ORACLE_CHUNK_SIZE,
    SEED,
    TEMPERATURE,
    TOP_P,
    TRUNCATION_BUDGET_TOKENS,
    get_stage1_paths,
)
from helpers import build_run_config, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize Stage 1 run_config.json")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--trunc-budget", type=int, default=TRUNCATION_BUDGET_TOKENS)
    p.add_argument("--chunk-size", type=int, default=ORACLE_CHUNK_SIZE)
    p.add_argument("--behavioral-probes", type=int, default=0)
    p.add_argument("--behavioral-epochs", type=int, default=1)
    p.add_argument("--behavioral-lr-mult", type=float, default=0.5)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--top-p", type=float, default=TOP_P)
    p.add_argument("--mode", default="unknown")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_stage1_paths(args.model_id)
    paths.root.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config(
        seed=args.seed,
        model_id=args.model_id,
        truncation_budget_tokens=args.trunc_budget,
        oracle_chunk_size=args.chunk_size,
        behavioral_probes=args.behavioral_probes,
        behavioral_epochs=args.behavioral_epochs,
        behavioral_lr_mult=args.behavioral_lr_mult,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        notes={
            "mode": args.mode,
            "stage": "stage-1",
            "purpose": "bridge_to_stage_2_preflight",
        },
    )
    write_json(paths.run_config, run_config)
    print(f"Wrote run config: {paths.run_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
