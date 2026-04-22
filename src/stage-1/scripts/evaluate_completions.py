from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from config import (
    CONDITIONS,
    HIGH_GAP_BLEU_THRESHOLD,
    LOW_GAP_BLEU_THRESHOLD,
    MODEL_ID,
    get_stage1_paths,
)

PER_INSTANCE_COLUMNS = [
    "instance_id",
    "condition",
    "repo",
    "file_path",
    "exact_match",
    "pass_at_1_proxy",
    "bleu4",
    "syntax_valid",
    "generation_time_s",
    "context_token_count",
    "generated_token_count",
    "hit_max_new_tokens",
    "bleu_gap_bc",
    "gap_stratum",
]


def normalize_text(s: str) -> str:
    lines = [line.rstrip() for line in s.strip().splitlines()]
    return "\n".join(lines).strip()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu4(prediction: str, reference: str, epsilon: float = 1e-9) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if not pred_tokens:
        return 0.0

    precisions = []
    for n in range(1, 5):
        pred_ngrams = _ngrams(pred_tokens, n)
        ref_ngrams = _ngrams(ref_tokens, n)
        if not pred_ngrams:
            precisions.append(epsilon)
            continue

        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        overlap = sum(min(c, ref_counts[g]) for g, c in pred_counts.items())
        precisions.append(max(epsilon, overlap / len(pred_ngrams)))

    c = len(pred_tokens)
    r = len(ref_tokens)
    brevity_penalty = 1.0 if c > r else math.exp(1.0 - (r / max(c, 1)))
    score = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return float(score)


def make_signature(masked_function: str) -> str:
    first = masked_function.splitlines()[0].rstrip()
    return first


def syntax_is_valid(masked_function: str, body: str) -> bool:
    signature = make_signature(masked_function)
    src = signature + "\n" + body + "\n"
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def evaluate_condition(
    condition: str, completions_dir: Path
) -> tuple[pd.DataFrame, dict]:
    inp = completions_dir / f"condition_{condition}.jsonl"
    if not inp.exists():
        raise FileNotFoundError(f"Missing completions file: {inp}")

    rows = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            pred = rec.get("predicted_body", "")
            gold = rec.get("ground_truth_body", "")
            pred_n = normalize_text(pred)
            gold_n = normalize_text(gold)
            exact = int(pred_n == gold_n)
            bleu = sentence_bleu4(pred_n, gold_n)
            valid = int(
                syntax_is_valid(rec.get("masked_function", "def f():\n    pass"), pred)
            )

            rows.append(
                {
                    "instance_id": rec["instance_id"],
                    "condition": condition,
                    "repo": rec.get("repo", ""),
                    "file_path": rec.get("file_path", ""),
                    "exact_match": exact,
                    "pass_at_1_proxy": exact,
                    "bleu4": bleu,
                    "syntax_valid": valid,
                    "generation_time_s": rec.get("generation_time_s", 0.0),
                    "context_token_count": rec.get("context_token_count", 0),
                    "generated_token_count": rec.get("generated_token_count", 0),
                    "hit_max_new_tokens": rec.get("hit_max_new_tokens", 0),
                    "bleu_gap_bc": float("nan"),
                    "gap_stratum": "unknown",
                }
            )

    df = pd.DataFrame(rows, columns=PER_INSTANCE_COLUMNS)
    if df.empty:
        summary = {
            "condition": condition,
            "n_instances": 0,
            "pass_at_1_proxy_mean": 0.0,
            "bleu4_mean": 0.0,
            "syntax_valid_rate": 0.0,
            "mean_generation_time_s": 0.0,
            "mean_context_token_count": 0.0,
            "mean_generated_token_count": 0.0,
            "max_new_token_hit_rate": 0.0,
        }
        return df, summary

    summary = {
        "condition": condition,
        "n_instances": int(df.shape[0]),
        "pass_at_1_proxy_mean": float(df["pass_at_1_proxy"].mean()),
        "bleu4_mean": float(df["bleu4"].mean()),
        "syntax_valid_rate": float(df["syntax_valid"].mean()),
        "mean_generation_time_s": float(df["generation_time_s"].mean()),
        "mean_context_token_count": float(df["context_token_count"].mean()),
        "mean_generated_token_count": float(df["generated_token_count"].mean()),
        "max_new_token_hit_rate": float(df["hit_max_new_tokens"].mean()),
    }
    return df, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage 1 condition outputs")
    p.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--low-gap-threshold", type=float, default=LOW_GAP_BLEU_THRESHOLD)
    p.add_argument(
        "--high-gap-threshold",
        type=float,
        default=HIGH_GAP_BLEU_THRESHOLD,
    )
    return p.parse_args()


def gap_stratum(
    gap: float,
    low_threshold: float,
    high_threshold: float,
) -> str:
    if math.isnan(gap):
        return "unknown"
    if gap < low_threshold:
        return "low"
    if gap < high_threshold:
        return "medium"
    return "high"


def compute_gap_lookup(
    b_df: pd.DataFrame,
    c_df: pd.DataFrame,
    low_threshold: float,
    high_threshold: float,
) -> pd.DataFrame:
    if b_df.empty or c_df.empty:
        return pd.DataFrame(columns=["instance_id", "bleu_gap_bc", "gap_stratum"])

    gap_df = (
        b_df[["instance_id", "bleu4"]]
        .rename(columns={"bleu4": "bleu_B"})
        .merge(
            c_df[["instance_id", "bleu4"]].rename(columns={"bleu4": "bleu_C"}),
            on="instance_id",
            how="outer",
        )
    )
    gap_df["bleu_gap_bc"] = gap_df["bleu_C"] - gap_df["bleu_B"]
    gap_df["gap_stratum"] = gap_df["bleu_gap_bc"].apply(
        lambda x: (
            gap_stratum(float(x), low_threshold, high_threshold)
            if pd.notna(x)
            else "unknown"
        )
    )
    return gap_df[["instance_id", "bleu_gap_bc", "gap_stratum"]]


def main() -> int:
    args = parse_args()
    paths = get_stage1_paths(args.model_id)
    paths.evaluation.mkdir(parents=True, exist_ok=True)

    conds = CONDITIONS if args.condition == "all" else [args.condition]
    all_summaries = []
    dfs: dict[str, pd.DataFrame] = {}

    for cond in conds:
        df, summary = evaluate_condition(cond, paths.completions)
        dfs[cond] = df
        all_summaries.append(summary)

    gap_lookup = compute_gap_lookup(
        b_df=dfs.get("B", pd.DataFrame()),
        c_df=dfs.get("C", pd.DataFrame()),
        low_threshold=args.low_gap_threshold,
        high_threshold=args.high_gap_threshold,
    )

    for summary in all_summaries:
        cond = summary["condition"]
        df = dfs[cond]
        if not df.empty:
            df = df.merge(
                gap_lookup, on="instance_id", how="left", suffixes=("", "_new")
            )
            if "bleu_gap_bc_new" in df.columns:
                df["bleu_gap_bc"] = df["bleu_gap_bc_new"]
                df = df.drop(columns=["bleu_gap_bc_new"])
            if "gap_stratum_new" in df.columns:
                df["gap_stratum"] = df["gap_stratum_new"]
                df = df.drop(columns=["gap_stratum_new"])
            df["gap_stratum"] = df["gap_stratum"].fillna("unknown")
            df["bleu_gap_bc"] = df["bleu_gap_bc"].astype(float)
            summary["gap_strata_counts"] = (
                df["gap_stratum"].value_counts(dropna=False).to_dict()
            )
        else:
            summary["gap_strata_counts"] = {}

        df_out = paths.evaluation / f"condition_{cond}.per_instance.csv"
        json_out = paths.evaluation / f"condition_{cond}.summary.json"
        df.to_csv(df_out, index=False)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(
            f"{cond}: pass@1_proxy={summary['pass_at_1_proxy_mean']:.3f}, bleu4={summary['bleu4_mean']:.3f}"
        )

    with (paths.evaluation / "all_conditions.summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(all_summaries, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
