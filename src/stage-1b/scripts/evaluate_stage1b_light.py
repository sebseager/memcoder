#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    stage1b_root = repo_root / "src" / "stage-1b"
    p = argparse.ArgumentParser(
        description=(
            "Lightweight Stage 1b evaluation without Docker: exact match, token F1, "
            "BLEU-4, AST validity, and proxy recovery ratio."
        )
    )
    p.add_argument("--predictions-jsonl", type=Path, default=stage1b_root / "outputs" / "stage1b_predictions.jsonl")
    p.add_argument("--output-per-instance-csv", type=Path, default=stage1b_root / "outputs" / "stage1b_light_eval.per_instance.csv")
    p.add_argument("--output-summary-json", type=Path, default=stage1b_root / "outputs" / "stage1b_light_eval.summary.json")
    p.add_argument("--baseline-b", type=float, default=None, help="Condition B score/pass@1. Required for rho.")
    p.add_argument("--ceiling-c", type=float, default=None, help="Condition C score/pass@1. Required for rho.")
    p.add_argument(
        "--rho-metric",
        choices=["exact_match", "token_f1", "bleu4", "syntax_valid"],
        default="token_f1",
        help="Lightweight proxy metric to use as E in rho=(E-B)/(C-B).",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip()


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
        p_grams = _ngrams(pred_tokens, n)
        r_grams = _ngrams(ref_tokens, n)
        if not p_grams:
            precisions.append(epsilon)
            continue
        p_count = Counter(p_grams)
        r_count = Counter(r_grams)
        overlap = sum(min(c, r_count[g]) for g, c in p_count.items())
        precisions.append(max(epsilon, overlap / len(p_grams)))
    c = len(pred_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1.0 - (r / max(c, 1)))
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / 4.0))


def code_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%<>]=?|[()[\]{}.,:]", text)


def token_f1(prediction: str, reference: str) -> float:
    pred = code_tokens(prediction)
    ref = code_tokens(reference)
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0
    p_count = Counter(pred)
    r_count = Counter(ref)
    overlap = sum(min(c, r_count[t]) for t, c in p_count.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def make_signature(masked_function: str) -> str:
    lines = masked_function.splitlines()
    if not lines:
        return "def f():"
    if lines[-1].strip() == "pass":
        return "\n".join(lines[:-1]).rstrip()
    return "\n".join(lines).rstrip()


def syntax_is_valid(masked_function: str, body: str) -> bool:
    signature = make_signature(masked_function)
    src = signature + "\n" + body.rstrip() + "\n"
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        first = signature.splitlines()[0] if signature else ""
        if first.startswith((" ", "\t")):
            try:
                ast.parse("class _Stage1bWrapper:\n" + src + "\n")
                return True
            except SyntaxError:
                return False
        return False


def branch_for_rho(rho: float | None) -> str:
    if rho is None:
        return "not_computed"
    if rho > 0.25:
        return "Path A (strong): proceed to Stage 2 scale-up"
    if rho >= 0.10:
        return "Path B (weak): scale to 1,500 triples and run a second epoch"
    return "Path C (fail): trigger D2L on Gemma-2-2b contingency"


def main() -> int:
    args = parse_args()
    rows = [r for r in read_jsonl(args.predictions_jsonl) if r.get("status", "ok") == "ok"]
    per_instance: list[dict[str, Any]] = []
    for row in rows:
        pred = normalize_text(str(row.get("predicted_body", "")))
        ref = normalize_text(str(row.get("ground_truth_body", "")))
        masked = str(row.get("masked_function", ""))
        per_instance.append(
            {
                "instance_id": row.get("instance_id", ""),
                "repo": row.get("repo", ""),
                "file_path": row.get("file_path", ""),
                "function_name": row.get("function_name", ""),
                "exact_match": int(pred == ref and bool(ref)),
                "token_f1": token_f1(pred, ref),
                "bleu4": sentence_bleu4(pred, ref),
                "syntax_valid": int(syntax_is_valid(masked, pred)),
                "slice_token_count": row.get("slice_token_count", ""),
                "slice_coverage_fraction": row.get("slice_coverage_fraction", ""),
                "generation_time_s": row.get("generation_time_s", ""),
            }
        )

    means: dict[str, float] = {}
    for key in ["exact_match", "token_f1", "bleu4", "syntax_valid"]:
        means[key] = sum(float(r[key]) for r in per_instance) / max(len(per_instance), 1)

    rho = None
    if args.baseline_b is not None and args.ceiling_c is not None and args.ceiling_c != args.baseline_b:
        rho = (means[args.rho_metric] - args.baseline_b) / (args.ceiling_c - args.baseline_b)

    args.output_per_instance_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_per_instance_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "instance_id",
            "repo",
            "file_path",
            "function_name",
            "exact_match",
            "token_f1",
            "bleu4",
            "syntax_valid",
            "slice_token_count",
            "slice_coverage_fraction",
            "generation_time_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_instance)

    summary = {
        "n_instances": len(per_instance),
        "means": means,
        "rho_metric": args.rho_metric,
        "baseline_b": args.baseline_b,
        "ceiling_c": args.ceiling_c,
        "recovery_ratio_rho": rho,
        "gate": branch_for_rho(rho),
        "note": (
            "This is a lightweight proxy evaluation. For the paper metric, rerun the "
            "SWE-rebench Docker harness and compute rho on pass@1."
        ),
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
