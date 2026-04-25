#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
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


def _line_indent(line: str) -> str:
    m = re.match(r"^\s*", line)
    return "" if m is None else m.group(0)


def _target_body_indent(masked_function: str) -> str:
    lines = masked_function.splitlines()
    if not lines:
        return "    "
    last = lines[-1]
    if last.strip() == "pass":
        return _line_indent(last) or "    "
    return "    "


def _find_function_node(
    full_file: str,
    *,
    start_line: int,
    function_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(full_file)
    except SyntaxError:
        return None

    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if function_name and node.name != function_name:
            continue
        decorator_start = min(
            [getattr(dec, "lineno", node.lineno) for dec in node.decorator_list],
            default=node.lineno,
        )
        end_line = int(getattr(node, "end_lineno", node.lineno))
        if node.lineno == start_line or decorator_start <= start_line <= end_line:
            return node
        if best is None:
            best = node
    return best


def _target_body_indent_from_full_file(
    full_file: str,
    *,
    start_line: int,
    function_name: str,
    masked_function: str,
) -> str:
    node = _find_function_node(full_file, start_line=start_line, function_name=function_name)
    lines = full_file.splitlines()
    if node is not None and getattr(node, "body", None):
        body_line_idx = int(node.body[0].lineno) - 1
        if 0 <= body_line_idx < len(lines):
            indent = _line_indent(lines[body_line_idx])
            if indent:
                return indent
    return _target_body_indent(masked_function)


def normalize_body_prediction(
    predicted_body: str,
    masked_function: str,
    *,
    target_indent: str | None = None,
) -> str:
    text = predicted_body.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    lines = text.splitlines()
    while lines and lines[0].strip().lower().startswith(("here's", "the function", "###", "---")):
        lines = lines[1:]

    target_indent = target_indent or _target_body_indent(masked_function)
    indents = [_line_indent(line) for line in lines if line.strip()]
    common_indent = ""
    if indents:
        common_indent = indents[0]
        for indent in indents[1:]:
            common_indent = os.path.commonprefix([common_indent, indent])

    out: list[str] = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        trimmed = line
        if common_indent and line.startswith(common_indent):
            trimmed = line[len(common_indent) :]
        out.append(f"{target_indent}{trimmed.rstrip()}")
    return "\n".join(out).rstrip()


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
        masked = str(row.get("masked_function", ""))
        full_file = str(row.get("full_file", ""))
        target_indent = _target_body_indent_from_full_file(
            full_file,
            start_line=int(row.get("start_line") or 1),
            function_name=str(row.get("function_name", "")),
            masked_function=masked,
        )
        pred = normalize_text(normalize_body_prediction(str(row.get("predicted_body", "")), masked, target_indent=target_indent))
        ref = normalize_text(str(row.get("ground_truth_body", "")))
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
