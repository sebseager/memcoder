#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import difflib
import json
import logging
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm


@dataclass(frozen=True)
class PassResult:
    pass_at_1: int
    source: str
    status: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    stage1a_root = repo_root / "src" / "stage-1a"
    default_predictions = stage1a_root / "outputs" / "stage1a_predictions.jsonl"
    default_per_instance = stage1a_root / "outputs" / "stage1a_score.per_instance.csv"
    default_summary = stage1a_root / "outputs" / "stage1a_score.summary.json"
    default_harness_preds = stage1a_root / "outputs" / "harness_predictions"
    default_harness_reports = stage1a_root / "outputs" / "harness_reports"

    p = argparse.ArgumentParser(
        description=(
            "Score Stage 1a predictions and compute pass@1 via local SWE-rebench "
            "harness (docker)."
        )
    )
    p.add_argument("--predictions-jsonl", type=Path, default=default_predictions)
    p.add_argument("--output-per-instance-csv", type=Path, default=default_per_instance)
    p.add_argument("--output-summary-json", type=Path, default=default_summary)
    p.add_argument("--pass-at-1-mode", choices=["swebench_harness", "exact_only"], default="swebench_harness")
    p.add_argument("--dataset-name", default="nebius/SWE-rebench-leaderboard")
    p.add_argument("--namespace", default="swerebench")
    p.add_argument("--cache-level", default="instance")
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--run-id-prefix", default="stage1a-eval")
    p.add_argument("--docker-config-mode", choices=["safe", "inherit"], default="safe")
    p.add_argument("--harness-predictions-dir", type=Path, default=default_harness_preds)
    p.add_argument("--harness-reports-dir", type=Path, default=default_harness_reports)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip()


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu4(prediction: str, reference: str, epsilon: float = 1e-9) -> float:
    import math

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


def normalize_body_prediction(predicted_body: str, masked_function: str) -> str:
    text = predicted_body.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    lines = text.splitlines()
    # Drop obvious prose wrappers that models often produce.
    while lines and lines[0].strip().lower().startswith(("here's", "the function", "###", "---")):
        lines = lines[1:]

    target_indent = _target_body_indent(masked_function)
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
    src = signature + "\n" + body + "\n"
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        pass

    first_line = signature.splitlines()[0] if signature else ""
    if first_line.startswith((" ", "\t")):
        try:
            ast.parse("class _Wrapper:\n" + src + "\n")
            return True
        except SyntaxError:
            return False
    return False


def patch_full_file_with_prediction(
    full_file: str,
    start_line: int,
    end_line: int,
    masked_function: str,
    predicted_body: str,
) -> str:
    lines = full_file.splitlines()
    signature = make_signature(masked_function)
    replacement = signature + ("\n" + predicted_body.rstrip() if predicted_body.strip() else "")
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    patched = lines[:start_idx] + replacement.splitlines() + lines[end_idx:]
    out = "\n".join(patched)
    if full_file.endswith("\n"):
        out += "\n"
    return out


def build_unified_patch(file_path: str, original_text: str, new_text: str) -> str:
    diff = difflib.unified_diff(
        original_text.splitlines(),
        new_text.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    patch = "\n".join(diff).strip()
    return patch + "\n" if patch else ""


def combine_patches(*patches: str) -> str:
    chunks = [p.strip() for p in patches if p and p.strip()]
    if not chunks:
        return ""
    return "\n\n".join(chunks).strip() + "\n"


def _build_harness_env(mode: str, outputs_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    if mode != "safe":
        return env
    safe_dir = outputs_root / ".docker-safe-config"
    safe_dir.mkdir(parents=True, exist_ok=True)
    cfg = safe_dir / "config.json"
    payload: dict[str, dict] = {"auths": {}}
    if cfg.exists():
        try:
            existing = json.loads(cfg.read_text(encoding="utf-8"))
            auths = existing.get("auths")
            if isinstance(auths, dict):
                payload["auths"] = auths
        except json.JSONDecodeError:
            pass
    cfg.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    env["DOCKER_CONFIG"] = str(safe_dir)
    return env


def _read_harness_report(run_id: str, model_name: str, instance_id: str) -> dict[str, Any] | None:
    report_path = (
        Path("logs")
        / "run_evaluation"
        / run_id
        / model_name
        / instance_id
        / "report.json"
    )
    if not report_path.exists():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload.get(instance_id)


def _move_report(model_name: str, run_id: str, dst_dir: Path) -> str:
    src_name = f"{model_name.replace('/', '__')}.{run_id}.json"
    src = Path(src_name)
    if not src.exists():
        return ""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst.unlink()
    src.replace(dst)
    return str(dst)


def run_harness(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    outputs_root: Path,
) -> dict[str, PassResult]:
    model_name = "stage1a-zero"
    results: dict[str, PassResult] = {}
    submissions: list[dict[str, str]] = []
    instance_ids: list[str] = []
    skipped_syntax = 0
    skipped_missing_file = 0
    skipped_noop_patch = 0

    for rec in tqdm(rows, desc="Build harness submissions", unit="pred"):
        iid = str(rec.get("instance_id", ""))
        if not iid:
            continue

        if not rec.get("syntax_valid", 1):
            results[iid] = PassResult(0, "executed", "syntax_invalid")
            skipped_syntax += 1
            continue

        full_file = str(rec.get("full_file", ""))
        if not full_file.strip():
            results[iid] = PassResult(0, "fallback", "missing_full_file")
            skipped_missing_file += 1
            continue
        file_path = str(rec.get("file_path", ""))
        start_line = int(rec.get("start_line", 1))
        end_line = int(rec.get("end_line", start_line))
        masked_function = str(rec.get("masked_function", ""))
        predicted_body = str(rec.get("predicted_body", ""))
        gold_patch = str(rec.get("gold_patch", ""))

        patched = patch_full_file_with_prediction(
            full_file=full_file,
            start_line=start_line,
            end_line=end_line,
            masked_function=masked_function,
            predicted_body=predicted_body,
        )
        prediction_patch = build_unified_patch(file_path, full_file, patched)
        submission_patch = combine_patches(gold_patch, prediction_patch)
        if not submission_patch.strip():
            results[iid] = PassResult(0, "executed", "noop_patch")
            skipped_noop_patch += 1
            continue

        submissions.append(
            {
                "instance_id": iid,
                "model_name_or_path": model_name,
                "model_patch": submission_patch,
            }
        )
        instance_ids.append(iid)

    if not submissions:
        logging.warning(
            "No harness submissions built (syntax_invalid=%d, missing_full_file=%d, noop_patch=%d).",
            skipped_syntax,
            skipped_missing_file,
            skipped_noop_patch,
        )
        return results
    logging.info(
        "Harness submissions built: %d/%d (syntax_invalid=%d, missing_full_file=%d, noop_patch=%d).",
        len(submissions),
        len(rows),
        skipped_syntax,
        skipped_missing_file,
        skipped_noop_patch,
    )

    run_id = f"{args.run_id_prefix}-{int(time.time())}"
    pred_path = args.harness_predictions_dir / f"{run_id}.jsonl"
    args.harness_predictions_dir.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for row in submissions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cmd = [
        "python",
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        args.dataset_name,
        "--predictions_path",
        str(pred_path),
        "--instance_ids",
        *instance_ids,
        "--cache_level",
        args.cache_level,
        "--run_id",
        run_id,
        "--namespace",
        args.namespace,
        "--timeout",
        str(args.timeout_seconds),
        "--max_workers",
        str(args.max_workers),
    ]
    env = _build_harness_env(args.docker_config_mode, outputs_root)
    logging.info(
        "[harness] run_id=%s n_predictions=%d timeout=%ss workers=%d",
        run_id,
        len(submissions),
        args.timeout_seconds,
        args.max_workers,
    )
    proc = subprocess.run(cmd, check=False, env=env)
    run_exit = proc.returncode
    report_copy = _move_report(model_name, run_id, args.harness_reports_dir)
    if report_copy:
        logging.info("[harness] moved aggregate report to %s", report_copy)

    for iid in tqdm(instance_ids, desc="Read harness reports", unit="instance"):
        report = _read_harness_report(run_id=run_id, model_name=model_name, instance_id=iid)
        if report is None:
            results[iid] = PassResult(0, "fallback", f"missing_report_exit_{run_exit}")
            continue
        if not bool(report.get("patch_successfully_applied")):
            results[iid] = PassResult(0, "executed", "patch_not_applied")
            continue
        resolved = bool(report.get("resolved"))
        results[iid] = PassResult(
            pass_at_1=int(resolved),
            source="executed",
            status="resolved" if resolved else "not_resolved",
        )
    return results


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if not args.predictions_jsonl.exists():
        raise FileNotFoundError(f"Missing predictions JSONL: {args.predictions_jsonl}")
    if args.output_per_instance_csv.exists() and not args.force:
        logging.error("Output exists (use --force): %s", args.output_per_instance_csv)
        return 1

    args.output_per_instance_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    outputs_root = args.output_summary_json.parent

    raw = read_jsonl(args.predictions_jsonl)
    rows = [r for r in raw if r.get("status") == "ok"]
    logging.info("Loaded %d prediction rows (ok status).", len(rows))

    records: list[dict[str, Any]] = []
    normalized_changed = 0
    syntax_valid_before = 0
    syntax_valid_after = 0
    for rec in tqdm(rows, desc="Compute text/syntax metrics", unit="pred"):
        pred_raw = str(rec.get("predicted_body", ""))
        masked_function = str(rec.get("masked_function", ""))
        pred = normalize_body_prediction(pred_raw, masked_function)
        if pred != pred_raw:
            normalized_changed += 1
        gold = str(rec.get("ground_truth_body", ""))
        pred_n = normalize_text(pred)
        gold_n = normalize_text(gold)
        exact = int(pred_n == gold_n)
        bleu = sentence_bleu4(pred_n, gold_n)
        valid_before = int(syntax_is_valid(masked_function, pred_raw))
        valid = int(syntax_is_valid(masked_function, pred))
        syntax_valid_before += valid_before
        syntax_valid_after += valid
        records.append(
            {
                "instance_id": str(rec.get("instance_id", "")),
                "repo": rec.get("repo", ""),
                "file_path": rec.get("file_path", ""),
                "function_name": rec.get("function_name", ""),
                "exact_match": exact,
                "pass_at_1": 0,
                "pass_at_1_source": "pending",
                "pass_at_1_status": "pending",
                "bleu4": bleu,
                "syntax_valid": valid,
                "generation_time_s": float(rec.get("generation_time_s", 0.0)),
                "slice_token_count": int(rec.get("slice_token_count", 0)),
                "slice_coverage_fraction": float(rec.get("slice_coverage_fraction", 0.0)),
                "predicted_body": pred,
                "raw_predicted_body": pred_raw,
                "ground_truth_body": gold,
                "masked_function": masked_function,
                "full_file": rec.get("full_file", ""),
                "start_line": rec.get("start_line"),
                "end_line": rec.get("end_line"),
                "gold_patch": rec.get("gold_patch", ""),
                "syntax_valid_internal": bool(valid),
            }
        )
    logging.info(
        "Normalized %d/%d predictions; syntax-valid before=%d, after=%d.",
        normalized_changed,
        len(rows),
        syntax_valid_before,
        syntax_valid_after,
    )

    if args.pass_at_1_mode == "exact_only":
        for r in records:
            r["pass_at_1"] = int(r["exact_match"])
            r["pass_at_1_source"] = "exact_match_only"
            r["pass_at_1_status"] = "mode_exact_only"
    else:
        harness_results = run_harness(records, args=args, outputs_root=outputs_root)
        for r in records:
            iid = r["instance_id"]
            result = harness_results.get(iid)
            if result is None:
                r["pass_at_1"] = int(r["exact_match"])
                r["pass_at_1_source"] = "fallback_exact_match"
                r["pass_at_1_status"] = "no_harness_result"
            else:
                r["pass_at_1"] = int(result.pass_at_1)
                r["pass_at_1_source"] = result.source
                r["pass_at_1_status"] = result.status

    df = pd.DataFrame(records)
    if df.empty:
        summary = {
            "n_instances": 0,
            "pass_at_1_mean": 0.0,
            "exact_match_mean": 0.0,
            "bleu4_mean": 0.0,
            "syntax_valid_rate": 0.0,
            "slice_coverage_median": 0.0,
            "slice_coverage_target_met": False,
            "pass_at_1_source_counts": {},
            "pass_at_1_status_counts": {},
        }
    else:
        median_cov = float(df["slice_coverage_fraction"].median())
        summary = {
            "n_instances": int(df.shape[0]),
            "pass_at_1_mean": float(df["pass_at_1"].mean()),
            "exact_match_mean": float(df["exact_match"].mean()),
            "bleu4_mean": float(df["bleu4"].mean()),
            "syntax_valid_rate": float(df["syntax_valid"].mean()),
            "slice_coverage_median": median_cov,
            "slice_coverage_target_met": bool(median_cov >= 0.70),
            "pass_at_1_source_counts": df["pass_at_1_source"].value_counts(dropna=False).to_dict(),
            "pass_at_1_status_counts": df["pass_at_1_status"].value_counts(dropna=False).to_dict(),
            "mean_generation_time_s": float(df["generation_time_s"].mean()),
            "mean_slice_token_count": float(df["slice_token_count"].mean()),
        }

    drop_cols = {
        "predicted_body",
        "raw_predicted_body",
        "ground_truth_body",
        "masked_function",
        "full_file",
        "gold_patch",
        "syntax_valid_internal",
    }
    keep_cols = [c for c in df.columns if c not in drop_cols]
    df[keep_cols].to_csv(args.output_per_instance_csv, index=False)
    args.output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    logging.info("Saved per-instance metrics to %s", args.output_per_instance_csv)
    logging.info("Saved summary to %s", args.output_summary_json)
    logging.info(
        "Stage1a pass@1=%.3f (n=%d)",
        summary["pass_at_1_mean"],
        summary["n_instances"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
