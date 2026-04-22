from __future__ import annotations

import argparse
import ast
import difflib
import json
import math
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from config import (
    CONDITIONS,
    HIGH_GAP_BLEU_THRESHOLD,
    INSTANCES_JSONL,
    LOW_GAP_BLEU_THRESHOLD,
    MODEL_ID,
    STAGE0_DIR,
    get_stage1_paths,
)

# Defaults tuned to match Stage 0's verify_wipe configuration, which has already
# proven stable on this dataset.
DEFAULT_HARNESS_DATASET = "nebius/SWE-rebench-leaderboard"
DEFAULT_HARNESS_NAMESPACE = "swerebench"
DEFAULT_HARNESS_CACHE_LEVEL = "instance"
DEFAULT_HARNESS_TIMEOUT_SECONDS = 1800
DEFAULT_HARNESS_MAX_WORKERS = 2

# Gold patches are not inlined in stage1_instances.jsonl (the Stage-0 final
# artifact) but are required at eval time so we can submit `gold_patch +
# prediction_patch` to the harness -- mirroring Stage-0's verify_wipe which
# submits `gold_patch + mask_patch`. Stage-0 keeps them in
# `05_gold_passed_verified_instances.jsonl` keyed by instance_id.
DEFAULT_GOLD_PATCHES_JSONL = (
    STAGE0_DIR / "outputs" / "05_gold_passed_verified_instances.jsonl"
)


@dataclass(frozen=True)
class InstanceMeta:
    instance_id: str
    repo: str
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    masked_function: str
    full_file: str
    docker_image: str
    test_cmd: str
    gold_patch: str


@dataclass(frozen=True)
class PassAtOneResult:
    pass_at_1: int
    source: str
    status: str


PER_INSTANCE_COLUMNS = [
    "instance_id",
    "condition",
    "repo",
    "file_path",
    "exact_match",
    "pass_at_1",
    "pass_at_1_source",
    "pass_at_1_status",
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
    lines = masked_function.splitlines()
    if not lines:
        return "def f():"
    if lines[-1].strip() == "pass":
        return "\n".join(lines[:-1]).rstrip()
    return "\n".join(lines).rstrip()


def syntax_is_valid(masked_function: str, body: str) -> bool:
    signature = make_signature(masked_function)
    src = signature + "\n" + body + "\n"
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


def load_instance_meta(instances_jsonl: Path) -> dict[str, InstanceMeta]:
    if not instances_jsonl.exists():
        print(
            "Warning: instances file not found, "
            f"harness pass@1 disabled: {instances_jsonl}"
        )
        return {}

    # Hydrate artifact-backed rows by delegating to the shared helper, which loads
    # full_file / masked_file / masked_function from disk.
    from helpers import hydrate_instance_row  # local import to avoid torch at import

    out: dict[str, InstanceMeta] = {}
    with instances_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            hydrate_instance_row(rec)

            instance_id = rec.get("instance_id")
            start_line = rec.get("start_line")
            end_line = rec.get("end_line")
            if not isinstance(instance_id, str):
                continue
            if not isinstance(start_line, int) or not isinstance(end_line, int):
                continue
            if start_line < 1 or end_line < start_line:
                continue

            out[instance_id] = InstanceMeta(
                instance_id=instance_id,
                repo=str(rec.get("repo", "")),
                file_path=str(rec.get("file_path", "")),
                function_name=str(rec.get("function_name", "")),
                start_line=start_line,
                end_line=end_line,
                masked_function=str(rec.get("masked_function", "")),
                full_file=str(rec.get("full_file", "")),
                docker_image=str(rec.get("docker_image", "")),
                test_cmd=str(rec.get("test_cmd", "")),
            )
    return out


def build_signature_plus_body(masked_function: str, predicted_body: str) -> str:
    signature = make_signature(masked_function)
    body = predicted_body.rstrip()
    if not body:
        return signature
    return signature + "\n" + body


def patch_full_file_with_prediction(
    full_file: str,
    start_line: int,
    end_line: int,
    masked_function: str,
    predicted_body: str,
) -> str:
    lines = full_file.splitlines()
    replacement = build_signature_plus_body(masked_function, predicted_body)
    replacement_lines = replacement.splitlines()
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    patched_lines = lines[:start_idx] + replacement_lines + lines[end_idx:]
    patched = "\n".join(patched_lines)
    if full_file.endswith("\n"):
        patched += "\n"
    return patched


def build_unified_patch(file_path: str, original_text: str, new_text: str) -> str:
    original_lines = original_text.splitlines()
    new_lines = new_text.splitlines()
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    patch = "\n".join(diff).strip()
    return patch + "\n" if patch else ""


def _build_harness_env(mode: str, stage1_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    if mode != "safe":
        return env

    safe_dir = stage1_root / ".docker-safe-config"
    safe_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = safe_dir / "config.json"
    payload: dict[str, dict] = {"auths": {}}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        auths = existing.get("auths")
        if isinstance(auths, dict):
            payload["auths"] = auths
    cfg_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    env["DOCKER_CONFIG"] = str(safe_dir)
    return env


def _move_report_to_outputs(
    model_name: str, run_id: str, reports_dir: Path
) -> str:
    src_name = f"{model_name.replace('/', '__')}.{run_id}.json"
    src_path = Path(src_name)
    if not src_path.exists():
        return ""
    reports_dir.mkdir(parents=True, exist_ok=True)
    dst_path = reports_dir / src_path.name
    if dst_path.exists():
        dst_path.unlink()
    src_path.replace(dst_path)
    return str(dst_path)


def _read_harness_report(run_id: str, model_name: str, instance_id: str) -> dict | None:
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


class HarnessExecutor:
    """Run SWE-rebench harness evaluations for a batch of predictions.

    Each call submits one predictions JSONL covering all instances in the batch
    and parses per-instance reports. This matches Stage 0's verify_wipe wiring
    so we get the same pass/fail signal that made Stage 0 instances pre-verified.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        stage1_root: Path,
        predictions_dir: Path,
        reports_dir: Path,
        dataset_name: str,
        namespace: str,
        cache_level: str,
        timeout_seconds: int,
        max_workers: int,
        run_id_prefix: str,
        docker_config_mode: str,
    ) -> None:
        self.enabled = enabled
        self.stage1_root = stage1_root
        self.predictions_dir = predictions_dir
        self.reports_dir = reports_dir
        self.dataset_name = dataset_name
        self.namespace = namespace
        self.cache_level = cache_level
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.max_workers = max(1, int(max_workers))
        self.run_id_prefix = run_id_prefix
        self.docker_config_mode = docker_config_mode

    def evaluate_condition(
        self,
        condition: str,
        records: list[dict],
        instances: dict[str, InstanceMeta],
    ) -> dict[str, PassAtOneResult]:
        if not self.enabled or not records:
            return {}

        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        predictions: list[dict] = []
        results: dict[str, PassAtOneResult] = {}
        instance_ids: list[str] = []
        patch_texts: dict[str, str] = {}

        for rec in records:
            iid = str(rec.get("instance_id", ""))
            if not iid:
                continue
            meta = instances.get(iid)
            if meta is None:
                results[iid] = PassAtOneResult(
                    pass_at_1=0,
                    source="fallback",
                    status="missing_instance_meta",
                )
                continue
            if not meta.full_file:
                results[iid] = PassAtOneResult(
                    pass_at_1=0,
                    source="fallback",
                    status="missing_full_file",
                )
                continue

            predicted = str(rec.get("predicted_body", ""))
            if not rec.get("_syntax_valid_", True):
                results[iid] = PassAtOneResult(
                    pass_at_1=0,
                    source="executed",
                    status="syntax_invalid",
                )
                continue

            patched = patch_full_file_with_prediction(
                full_file=meta.full_file,
                start_line=meta.start_line,
                end_line=meta.end_line,
                masked_function=meta.masked_function,
                predicted_body=predicted,
            )
            patch_text = build_unified_patch(meta.file_path, meta.full_file, patched)
            if not patch_text.strip():
                # No-op patch: prediction exactly matches base file (unexpected, but
                # treat as "resolved by base_commit", i.e. the test suite is already
                # green without any change).
                results[iid] = PassAtOneResult(
                    pass_at_1=0,
                    source="executed",
                    status="noop_patch",
                )
                continue

            predictions.append(
                {
                    "instance_id": iid,
                    "model_name_or_path": f"stage1-{condition.lower()}",
                    "model_patch": patch_text,
                }
            )
            instance_ids.append(iid)
            patch_texts[iid] = patch_text

        if not predictions:
            return results

        run_id = f"{self.run_id_prefix}-{condition.lower()}-{int(time.time())}"
        model_name = f"stage1-{condition.lower()}"
        pred_path = self.predictions_dir / f"{run_id}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.dataset_name,
            "--predictions_path",
            str(pred_path),
            "--instance_ids",
            *instance_ids,
            "--cache_level",
            self.cache_level,
            "--run_id",
            run_id,
            "--namespace",
            self.namespace,
            "--timeout",
            str(self.timeout_seconds),
            "--max_workers",
            str(self.max_workers),
        ]

        env = _build_harness_env(self.docker_config_mode, self.stage1_root)
        print(
            f"[harness] condition={condition} run_id={run_id} "
            f"n_predictions={len(predictions)} timeout={self.timeout_seconds}s"
        )
        proc = subprocess.run(cmd, check=False, env=env)
        run_exit = proc.returncode
        _move_report_to_outputs(
            model_name=model_name, run_id=run_id, reports_dir=self.reports_dir
        )

        for iid in instance_ids:
            report = _read_harness_report(
                run_id=run_id, model_name=model_name, instance_id=iid
            )
            if report is None:
                results[iid] = PassAtOneResult(
                    pass_at_1=0,
                    source="fallback",
                    status=f"missing_report_exit_{run_exit}",
                )
                continue
            patch_applied = bool(report.get("patch_successfully_applied"))
            resolved = bool(report.get("resolved"))
            if not patch_applied:
                results[iid] = PassAtOneResult(
                    pass_at_1=0,
                    source="executed",
                    status="patch_not_applied",
                )
                continue
            results[iid] = PassAtOneResult(
                pass_at_1=int(resolved),
                source="executed",
                status="resolved" if resolved else "not_resolved",
            )

        return results


def evaluate_condition(
    condition: str,
    completions_dir: Path,
    instance_meta: dict[str, InstanceMeta],
    executor: HarnessExecutor,
    pass_at_1_mode: str,
) -> tuple[pd.DataFrame, dict]:
    inp = completions_dir / f"condition_{condition}.jsonl"
    if not inp.exists():
        raise FileNotFoundError(f"Missing completions file: {inp}")

    rows: list[dict] = []
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
                syntax_is_valid(
                    rec.get("masked_function", "def f():\n    pass"), pred
                )
            )
            rows.append(
                {
                    "instance_id": rec["instance_id"],
                    "condition": condition,
                    "repo": rec.get("repo", ""),
                    "file_path": rec.get("file_path", ""),
                    "exact_match": exact,
                    "pass_at_1": 0,
                    "pass_at_1_source": "pending",
                    "pass_at_1_status": "pending",
                    "bleu4": bleu,
                    "syntax_valid": valid,
                    "generation_time_s": rec.get("generation_time_s", 0.0),
                    "context_token_count": rec.get("context_token_count", 0),
                    "generated_token_count": rec.get("generated_token_count", 0),
                    "hit_max_new_tokens": rec.get("hit_max_new_tokens", 0),
                    "bleu_gap_bc": float("nan"),
                    "gap_stratum": "unknown",
                    "_predicted_body_": pred,
                    "_masked_function_": rec.get("masked_function", ""),
                    "_syntax_valid_": bool(valid),
                }
            )

    if pass_at_1_mode == "exact_only":
        for r in rows:
            r["pass_at_1"] = int(r["exact_match"])
            r["pass_at_1_source"] = "exact_match_only"
            r["pass_at_1_status"] = "mode_exact_only"
    else:
        # Build records consumed by the harness executor.
        harness_records: list[dict] = []
        for r in rows:
            harness_records.append(
                {
                    "instance_id": r["instance_id"],
                    "predicted_body": r["_predicted_body_"],
                    "_syntax_valid_": r["_syntax_valid_"],
                }
            )
        result_map = executor.evaluate_condition(
            condition=condition,
            records=harness_records,
            instances=instance_meta,
        )
        for r in rows:
            iid = r["instance_id"]
            result = result_map.get(iid)
            if result is None:
                # executor disabled or no result (e.g. skipped); fall through to
                # exact-match fallback so downstream analysis has a numeric value.
                r["pass_at_1"] = int(r["exact_match"])
                r["pass_at_1_source"] = "fallback_exact_match"
                r["pass_at_1_status"] = "no_harness_result"
            else:
                r["pass_at_1"] = int(result.pass_at_1)
                r["pass_at_1_source"] = result.source
                r["pass_at_1_status"] = result.status

    for r in rows:
        r.pop("_predicted_body_", None)
        r.pop("_masked_function_", None)
        r.pop("_syntax_valid_", None)

    df = pd.DataFrame(rows, columns=PER_INSTANCE_COLUMNS)
    if df.empty:
        summary = {
            "condition": condition,
            "n_instances": 0,
            "exact_match_mean": 0.0,
            "pass_at_1_mean": 0.0,
            "bleu4_mean": 0.0,
            "syntax_valid_rate": 0.0,
            "mean_generation_time_s": 0.0,
            "mean_context_token_count": 0.0,
            "mean_generated_token_count": 0.0,
            "max_new_token_hit_rate": 0.0,
            "pass_at_1_source_counts": {},
            "pass_at_1_status_counts": {},
        }
        return df, summary

    summary = {
        "condition": condition,
        "n_instances": int(df.shape[0]),
        "exact_match_mean": float(df["exact_match"].mean()),
        "pass_at_1_mean": float(df["pass_at_1"].mean()),
        "bleu4_mean": float(df["bleu4"].mean()),
        "syntax_valid_rate": float(df["syntax_valid"].mean()),
        "mean_generation_time_s": float(df["generation_time_s"].mean()),
        "mean_context_token_count": float(df["context_token_count"].mean()),
        "mean_generated_token_count": float(df["generated_token_count"].mean()),
        "max_new_token_hit_rate": float(df["hit_max_new_tokens"].mean()),
        "pass_at_1_source_counts": df["pass_at_1_source"]
        .value_counts(dropna=False)
        .to_dict(),
        "pass_at_1_status_counts": df["pass_at_1_status"]
        .value_counts(dropna=False)
        .to_dict(),
    }
    return df, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage 1 condition outputs")
    p.add_argument("--condition", choices=CONDITIONS + ["all"], default="all")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument(
        "--pass-at-1-mode",
        choices=["swebench_harness", "exact_only"],
        default="swebench_harness",
        help=(
            "How to compute pass@1: SWE-rebench harness (docker-based execution) "
            "or exact-string-match fallback."
        ),
    )
    p.add_argument(
        "--instances-jsonl",
        type=Path,
        default=INSTANCES_JSONL,
        help="Path to Stage 0 stage1_instances.jsonl with span metadata.",
    )
    # Harness tunables (mirror stage-0/scripts/verify_wipe.py).
    p.add_argument("--harness-dataset-name", default=DEFAULT_HARNESS_DATASET)
    p.add_argument("--harness-namespace", default=DEFAULT_HARNESS_NAMESPACE)
    p.add_argument("--harness-cache-level", default=DEFAULT_HARNESS_CACHE_LEVEL)
    p.add_argument(
        "--harness-timeout-seconds",
        type=int,
        default=DEFAULT_HARNESS_TIMEOUT_SECONDS,
    )
    p.add_argument(
        "--harness-max-workers", type=int, default=DEFAULT_HARNESS_MAX_WORKERS
    )
    p.add_argument("--harness-run-id-prefix", default="stage1-eval")
    p.add_argument(
        "--harness-docker-config-mode",
        choices=["safe", "inherit"],
        default="safe",
    )
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

    instance_meta = load_instance_meta(args.instances_jsonl)

    harness_predictions_dir = paths.evaluation / "harness_predictions"
    harness_reports_dir = paths.evaluation / "harness_reports"
    executor = HarnessExecutor(
        enabled=(args.pass_at_1_mode == "swebench_harness"),
        stage1_root=paths.root,
        predictions_dir=harness_predictions_dir,
        reports_dir=harness_reports_dir,
        dataset_name=args.harness_dataset_name,
        namespace=args.harness_namespace,
        cache_level=args.harness_cache_level,
        timeout_seconds=args.harness_timeout_seconds,
        max_workers=args.harness_max_workers,
        run_id_prefix=args.harness_run_id_prefix,
        docker_config_mode=args.harness_docker_config_mode,
    )

    conds = CONDITIONS if args.condition == "all" else [args.condition]
    all_summaries: list[dict] = []
    dfs: dict[str, pd.DataFrame] = {}

    for cond in conds:
        df, summary = evaluate_condition(
            condition=cond,
            completions_dir=paths.completions,
            instance_meta=instance_meta,
            executor=executor,
            pass_at_1_mode=args.pass_at_1_mode,
        )
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
            f"{cond}: pass@1={summary['pass_at_1_mean']:.3f}, "
            f"exact={summary['exact_match_mean']:.3f}, bleu4={summary['bleu4_mean']:.3f}"
        )

    with (paths.evaluation / "all_conditions.summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(all_summaries, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
