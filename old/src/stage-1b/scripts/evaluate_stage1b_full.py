#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import json
import logging
import math
import os
import re
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - convenience for minimal local envs
    def tqdm(iterable, **_: Any):
        return iterable


@dataclass(frozen=True)
class PassResult:
    pass_at_1: int
    source: str
    status: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    stage1b_root = repo_root / "src" / "stage-1b"
    default_outputs = stage1b_root / "outputs"
    p = argparse.ArgumentParser(
        description="Run Stage 1b predictions through the SWE-rebench Docker harness."
    )
    p.add_argument("--predictions-jsonl", type=Path, default=default_outputs / "stage1b_predictions.jsonl")
    p.add_argument("--heldout-jsonl", type=Path, default=default_outputs / "heldout_instances.jsonl")
    p.add_argument("--output-per-instance-csv", type=Path, default=default_outputs / "stage1b_full_eval.per_instance.csv")
    p.add_argument("--output-summary-json", type=Path, default=default_outputs / "stage1b_full_eval.summary.json")
    p.add_argument("--dataset-name", default="nebius/SWE-rebench-leaderboard")
    p.add_argument("--dataset-split", default="train")
    p.add_argument(
        "--no-dataset-metadata",
        action="store_true",
        help="Do not backfill missing gold patches from the SWE-rebench dataset.",
    )
    p.add_argument("--namespace", default="swerebench")
    p.add_argument("--cache-level", default="instance")
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--run-id-prefix", default="stage1b-full-eval")
    p.add_argument("--docker-config-mode", choices=["safe", "inherit"], default="safe")
    p.add_argument("--harness-predictions-dir", type=Path, default=default_outputs / "harness_predictions")
    p.add_argument("--harness-reports-dir", type=Path, default=default_outputs / "harness_reports")
    p.add_argument("--pass-at-1-mode", choices=["swebench_harness", "exact_only"], default="swebench_harness")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip()


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


def patch_full_file_with_prediction(
    full_file: str,
    start_line: int,
    end_line: int,
    masked_function: str,
    predicted_body: str,
    function_name: str = "",
) -> str:
    lines = full_file.splitlines()
    node = _find_function_node(full_file, start_line=start_line, function_name=function_name)
    if node is not None and getattr(node, "body", None):
        body_start_idx = max(0, int(node.body[0].lineno) - 1)
        body_end_idx = min(len(lines), int(getattr(node, "end_lineno", end_line)))
        replacement = predicted_body.rstrip().splitlines()
        patched = lines[:body_start_idx] + replacement + lines[body_end_idx:]
        out = "\n".join(patched)
        if full_file.endswith("\n"):
            out += "\n"
        return out

    signature = make_signature(masked_function)
    replacement = signature + ("\n" + predicted_body.rstrip() if predicted_body.strip() else "")
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    patched = lines[:start_idx] + replacement.splitlines() + lines[end_idx:]
    out = "\n".join(patched)
    if full_file.endswith("\n"):
        out += "\n"
    return out


def patched_file_syntax_is_valid(
    full_file: str,
    start_line: int,
    end_line: int,
    function_name: str,
    masked_function: str,
    body: str,
) -> bool:
    if not full_file.strip():
        src = make_signature(masked_function) + "\n" + body + "\n"
        try:
            ast.parse(src)
            return True
        except SyntaxError:
            return False
    try:
        ast.parse(
            patch_full_file_with_prediction(
                full_file=full_file,
                start_line=start_line,
                end_line=end_line,
                masked_function=masked_function,
                predicted_body=body,
                function_name=function_name,
            )
        )
        return True
    except SyntaxError:
        return False


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


def _dataset_instance_id(row: dict[str, Any], fallback_idx: int) -> str:
    return str(row.get("instance_id") or row.get("id") or f"train-{fallback_idx}")


def load_instance_metadata(args: argparse.Namespace, instance_ids: set[str]) -> dict[str, dict[str, str]]:
    meta: dict[str, dict[str, str]] = {}
    for rec in read_jsonl(args.heldout_jsonl):
        iid = str(rec.get("instance_id", ""))
        if iid:
            meta[iid] = {
                "gold_patch": str(rec.get("gold_patch") or ""),
                "test_patch": str(rec.get("test_patch") or ""),
                "docker_image": str(rec.get("docker_image") or ""),
                "test_cmd": str(rec.get("test_cmd") or ""),
            }

    missing = {
        iid
        for iid in instance_ids
        if not meta.get(iid, {}).get("gold_patch")
    }
    if args.no_dataset_metadata or not missing:
        return meta

    try:
        from datasets import load_dataset
    except ImportError:
        logging.warning("datasets is unavailable; cannot backfill missing gold patches.")
        return meta

    logging.info("Backfilling gold patches for %d instances from %s/%s.", len(missing), args.dataset_name, args.dataset_split)
    try:
        dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not load dataset metadata: %s", exc)
        return meta

    for idx, row in enumerate(dataset, start=1):
        iid = _dataset_instance_id(dict(row), idx)
        if iid not in missing:
            continue
        current = meta.setdefault(iid, {})
        current["gold_patch"] = str(row.get("patch") or row.get("gold_patch") or current.get("gold_patch", ""))
        current["test_patch"] = str(row.get("test_patch") or current.get("test_patch", ""))
        current["docker_image"] = str(row.get("docker_image") or current.get("docker_image", ""))
        current["test_cmd"] = str(row.get("test_cmd") or row.get("test_command") or current.get("test_cmd", ""))
        missing.remove(iid)
        if not missing:
            break
    if missing:
        logging.warning("Missing gold patch metadata for %d instances.", len(missing))
    return meta


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


def group_unique_instance_batches(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    seen_per_batch: list[set[str]] = []
    for rec in records:
        iid = str(rec.get("instance_id", ""))
        for batch, seen in zip(batches, seen_per_batch):
            if iid not in seen:
                batch.append(rec)
                seen.add(iid)
                break
        else:
            batches.append([rec])
            seen_per_batch.append({iid})
    return batches


def run_harness(records: list[dict[str, Any]], *, args: argparse.Namespace, outputs_root: Path) -> dict[str, PassResult]:
    results: dict[str, PassResult] = {}
    args.harness_predictions_dir.mkdir(parents=True, exist_ok=True)
    args.harness_reports_dir.mkdir(parents=True, exist_ok=True)

    batches = group_unique_instance_batches(records)
    logging.info("Running %d harness batch(es) for %d prediction rows.", len(batches), len(records))
    for batch_idx, batch in enumerate(batches, start=1):
        submissions: list[dict[str, str]] = []
        submitted_records: list[dict[str, Any]] = []
        skipped_syntax = skipped_missing_file = skipped_noop_patch = 0

        for rec in batch:
            row_id = str(rec["row_id"])
            if not rec.get("syntax_valid", 0):
                results[row_id] = PassResult(0, "executed", "syntax_invalid")
                skipped_syntax += 1
                continue
            full_file = str(rec.get("full_file", ""))
            if not full_file.strip():
                results[row_id] = PassResult(0, "fallback", "missing_full_file")
                skipped_missing_file += 1
                continue

            patched = patch_full_file_with_prediction(
                full_file=full_file,
                start_line=int(rec.get("start_line", 1)),
                end_line=int(rec.get("end_line", rec.get("start_line", 1))),
                masked_function=str(rec.get("masked_function", "")),
                predicted_body=str(rec.get("predicted_body", "")),
                function_name=str(rec.get("function_name", "")),
            )
            prediction_patch = build_unified_patch(str(rec.get("file_path", "")), full_file, patched)
            submission_patch = combine_patches(str(rec.get("gold_patch", "")), prediction_patch)
            if not submission_patch.strip():
                results[row_id] = PassResult(0, "executed", "noop_patch")
                skipped_noop_patch += 1
                continue

            model_name = f"stage1b-full-batch-{batch_idx}"
            submissions.append(
                {
                    "instance_id": str(rec.get("instance_id", "")),
                    "model_name_or_path": model_name,
                    "model_patch": submission_patch,
                }
            )
            submitted_records.append(rec)

        if not submissions:
            logging.warning(
                "Batch %d produced no submissions (syntax_invalid=%d, missing_full_file=%d, noop_patch=%d).",
                batch_idx,
                skipped_syntax,
                skipped_missing_file,
                skipped_noop_patch,
            )
            continue

        run_id = f"{args.run_id_prefix}-batch-{batch_idx}-{int(time.time())}"
        model_name = f"stage1b-full-batch-{batch_idx}"
        pred_path = args.harness_predictions_dir / f"{run_id}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in submissions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        instance_ids = [str(rec.get("instance_id", "")) for rec in submitted_records]
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
            "[harness] batch=%d/%d run_id=%s n_predictions=%d timeout=%ss workers=%d",
            batch_idx,
            len(batches),
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
        if run_exit != 0:
            logging.warning("[harness] exited with code %d for run_id=%s.", run_exit, run_id)

        for rec in submitted_records:
            row_id = str(rec["row_id"])
            iid = str(rec.get("instance_id", ""))
            report = _read_harness_report(run_id=run_id, model_name=model_name, instance_id=iid)
            if report is None:
                results[row_id] = PassResult(0, "fallback", f"missing_report_exit_{run_exit}")
                continue
            if not bool(report.get("patch_successfully_applied")):
                results[row_id] = PassResult(0, "executed", "patch_not_applied")
                continue
            resolved = bool(report.get("resolved"))
            results[row_id] = PassResult(
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
    if (args.output_per_instance_csv.exists() or args.output_summary_json.exists()) and not args.force:
        logging.error("Output exists; pass --force to overwrite.")
        return 1

    args.output_per_instance_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    outputs_root = args.output_summary_json.parent

    raw_rows = [r for r in read_jsonl(args.predictions_jsonl) if r.get("status", "ok") == "ok"]
    instance_ids = {str(r.get("instance_id", "")) for r in raw_rows if r.get("instance_id")}
    instance_meta = load_instance_metadata(args, instance_ids)

    records: list[dict[str, Any]] = []
    normalized_changed = 0
    syntax_valid_before = 0
    syntax_valid_after = 0
    missing_gold_patch = 0
    for idx, rec in enumerate(tqdm(raw_rows, desc="Prepare Stage 1b full eval", unit="pred"), start=1):
        iid = str(rec.get("instance_id", ""))
        meta = instance_meta.get(iid, {})
        masked_function = str(rec.get("masked_function", ""))
        full_file = str(rec.get("full_file", ""))
        start_line = int(rec.get("start_line") or 1)
        end_line = int(rec.get("end_line") or start_line)
        function_name = str(rec.get("function_name", ""))
        pred_raw = str(rec.get("predicted_body", ""))
        target_indent = _target_body_indent_from_full_file(
            full_file,
            start_line=start_line,
            function_name=function_name,
            masked_function=masked_function,
        )
        pred = normalize_body_prediction(pred_raw, masked_function, target_indent=target_indent)
        if pred != pred_raw:
            normalized_changed += 1
        valid_before = int(
            patched_file_syntax_is_valid(
                full_file=full_file,
                start_line=start_line,
                end_line=end_line,
                function_name=function_name,
                masked_function=masked_function,
                body=pred_raw,
            )
        )
        valid = int(
            patched_file_syntax_is_valid(
                full_file=full_file,
                start_line=start_line,
                end_line=end_line,
                function_name=function_name,
                masked_function=masked_function,
                body=pred,
            )
        )
        syntax_valid_before += valid_before
        syntax_valid_after += valid
        gold_patch = str(rec.get("gold_patch") or meta.get("gold_patch") or "")
        if not gold_patch.strip():
            missing_gold_patch += 1
        gold = str(rec.get("ground_truth_body", ""))
        pred_n = normalize_text(pred)
        gold_n = normalize_text(gold)
        records.append(
            {
                "row_id": f"row-{idx}",
                "instance_id": iid,
                "repo": rec.get("repo", ""),
                "file_path": rec.get("file_path", ""),
                "function_name": function_name,
                "exact_match": int(pred_n == gold_n and bool(gold_n)),
                "token_f1": token_f1(pred_n, gold_n),
                "bleu4": sentence_bleu4(pred_n, gold_n),
                "syntax_valid": valid,
                "pass_at_1": 0,
                "pass_at_1_source": "pending",
                "pass_at_1_status": "pending",
                "slice_token_count": int(rec.get("slice_token_count", 0)),
                "slice_coverage_fraction": float(rec.get("slice_coverage_fraction", 0.0)),
                "generation_time_s": float(rec.get("generation_time_s", 0.0)),
                "gold_patch_present": int(bool(gold_patch.strip())),
                "predicted_body": pred,
                "raw_predicted_body": pred_raw,
                "ground_truth_body": gold,
                "masked_function": masked_function,
                "full_file": full_file,
                "start_line": start_line,
                "end_line": end_line,
                "gold_patch": gold_patch,
            }
        )
    logging.info(
        "Prepared %d rows; normalized=%d, syntax-valid before=%d, after=%d, missing gold_patch=%d.",
        len(records),
        normalized_changed,
        syntax_valid_before,
        syntax_valid_after,
        missing_gold_patch,
    )

    if args.pass_at_1_mode == "exact_only":
        for r in records:
            r["pass_at_1"] = int(r["exact_match"])
            r["pass_at_1_source"] = "exact_match_only"
            r["pass_at_1_status"] = "mode_exact_only"
    else:
        harness_results = run_harness(records, args=args, outputs_root=outputs_root)
        for r in records:
            result = harness_results.get(str(r["row_id"]))
            if result is None:
                r["pass_at_1"] = int(r["exact_match"])
                r["pass_at_1_source"] = "fallback_exact_match"
                r["pass_at_1_status"] = "no_harness_result"
            else:
                r["pass_at_1"] = int(result.pass_at_1)
                r["pass_at_1_source"] = result.source
                r["pass_at_1_status"] = result.status

    fieldnames = [
        "row_id",
        "instance_id",
        "repo",
        "file_path",
        "function_name",
        "exact_match",
        "token_f1",
        "bleu4",
        "syntax_valid",
        "pass_at_1",
        "pass_at_1_source",
        "pass_at_1_status",
        "slice_token_count",
        "slice_coverage_fraction",
        "generation_time_s",
        "gold_patch_present",
    ]
    with args.output_per_instance_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in records) / max(len(records), 1)

    summary = {
        "n_instances": len(records),
        "n_unique_swe_instances": len({r["instance_id"] for r in records}),
        "n_harness_batches": len(group_unique_instance_batches(records)),
        "pass_at_1_mean": mean("pass_at_1"),
        "exact_match_mean": mean("exact_match"),
        "token_f1_mean": mean("token_f1"),
        "bleu4_mean": mean("bleu4"),
        "syntax_valid_rate": mean("syntax_valid"),
        "gold_patch_present_rate": mean("gold_patch_present"),
        "normalized_changed": normalized_changed,
        "syntax_valid_before_normalization": syntax_valid_before,
        "syntax_valid_after_normalization": syntax_valid_after,
        "pass_at_1_source_counts": Counter(str(r["pass_at_1_source"]) for r in records),
        "pass_at_1_status_counts": Counter(str(r["pass_at_1_status"]) for r in records),
        "config": {
            "predictions_jsonl": str(args.predictions_jsonl),
            "dataset_name": args.dataset_name,
            "dataset_split": args.dataset_split,
            "cache_level": args.cache_level,
            "timeout_seconds": args.timeout_seconds,
            "max_workers": args.max_workers,
            "pass_at_1_mode": args.pass_at_1_mode,
        },
    }
    args.output_summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
