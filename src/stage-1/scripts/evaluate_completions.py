from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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

SKIP_PATH_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


@dataclass(frozen=True)
class InstanceMeta:
    instance_id: str
    repo: str
    file_path: str
    function_name: str
    start_line: int
    end_line: int
    masked_function: str


@dataclass(frozen=True)
class PassAtOneResult:
    pass_at_1: int
    source: str
    status: str
    selected_test_count: int


PER_INSTANCE_COLUMNS = [
    "instance_id",
    "condition",
    "repo",
    "file_path",
    "exact_match",
    "pass_at_1",
    "pass_at_1_source",
    "pass_at_1_status",
    "selected_test_count",
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
            f"Warning: instances file not found, execution pass@1 disabled: {instances_jsonl}"
        )
        return {}

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
            )
    return out


def repo_root_for_name(repos_root: Path, repo_name: str) -> Path:
    return repos_root / repo_name.replace("/", "__")


def is_test_file(relative_path: PurePosixPath) -> bool:
    name = relative_path.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return "tests" in relative_path.parts


def module_candidates_for_file(file_path: str) -> list[str]:
    path = PurePosixPath(file_path)
    parts = list(path.with_suffix("").parts)
    if not parts:
        return []
    if parts[-1] == "__init__":
        parts = parts[:-1]

    candidates = set()
    if parts:
        candidates.add(".".join(parts))
        candidates.add(parts[-1])
    if len(parts) > 1 and parts[0] in {"src", "lib", "python"}:
        candidates.add(".".join(parts[1:]))

    return sorted(x for x in candidates if x)


class PassAtOneExecutor:
    def __init__(
        self,
        *,
        mode: str,
        repos_root: Path,
        instances: dict[str, InstanceMeta],
        test_timeout_seconds: int,
        max_relevant_tests: int,
    ) -> None:
        self.mode = mode
        self.repos_root = repos_root
        self.instances = instances
        self.test_timeout_seconds = max(1, int(test_timeout_seconds))
        self.max_relevant_tests = max(1, int(max_relevant_tests))
        self.pytest_available = importlib.util.find_spec("pytest") is not None
        self._warned_missing_pytest = False

        self._repo_tests_cache: dict[str, list[str]] = {}
        self._test_text_cache: dict[tuple[str, str], str] = {}
        self._selection_cache: dict[tuple[str, str, str], list[str]] = {}
        self._baseline_result_cache: dict[tuple[str, tuple[str, ...]], int] = {}

    def _fallback(
        self,
        exact_match: int,
        status: str,
        selected_test_count: int = 0,
    ) -> PassAtOneResult:
        return PassAtOneResult(
            pass_at_1=int(exact_match),
            source="fallback_exact_match",
            status=status,
            selected_test_count=selected_test_count,
        )

    def _run_pytest(
        self,
        *,
        repo_root: Path,
        selected_tests: list[str],
    ) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-x", *selected_tests],
                cwd=repo_root,
                text=True,
                capture_output=True,
                timeout=self.test_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return -1, ""

        combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode, combined_output

    def _list_repo_tests(self, repo_name: str) -> list[str]:
        cached = self._repo_tests_cache.get(repo_name)
        if cached is not None:
            return cached

        repo_root = repo_root_for_name(self.repos_root, repo_name)
        if not repo_root.exists():
            self._repo_tests_cache[repo_name] = []
            return []

        tests: list[str] = []
        for path in repo_root.rglob("*.py"):
            if any(part in SKIP_PATH_PARTS for part in path.parts):
                continue
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root).as_posix()
            if is_test_file(PurePosixPath(rel)):
                tests.append(rel)

        tests.sort()
        self._repo_tests_cache[repo_name] = tests
        return tests

    def _read_test_text(self, repo_name: str, rel_path: str) -> str:
        key = (repo_name, rel_path)
        cached = self._test_text_cache.get(key)
        if cached is not None:
            return cached

        repo_root = repo_root_for_name(self.repos_root, repo_name)
        test_path = repo_root / rel_path
        try:
            text = test_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            text = ""
        self._test_text_cache[key] = text
        return text

    def _select_relevant_tests(
        self,
        repo_name: str,
        file_path: str,
        function_name: str,
    ) -> list[str]:
        cache_key = (repo_name, file_path, function_name)
        cached = self._selection_cache.get(cache_key)
        if cached is not None:
            return cached

        tests = self._list_repo_tests(repo_name)
        if not tests:
            self._selection_cache[cache_key] = []
            return []

        module_patterns = [
            re.compile(
                rf"(^|\n)\s*(from\s+{re.escape(module)}\s+import\b|import\s+{re.escape(module)}(\s|$|,|\.))",
                flags=re.MULTILINE,
            )
            for module in module_candidates_for_file(file_path)
        ]
        function_pattern = (
            re.compile(rf"\b{re.escape(function_name)}\b") if function_name else None
        )
        source_stem = PurePosixPath(file_path).stem
        stem_pattern = (
            re.compile(rf"\b{re.escape(source_stem)}\b") if source_stem else None
        )

        scored: list[tuple[int, str]] = []
        for rel_path in tests:
            text = self._read_test_text(repo_name, rel_path)
            if not text:
                continue

            score = 0
            if function_pattern is not None and function_pattern.search(text):
                score += 2
            if any(pattern.search(text) for pattern in module_patterns):
                score += 3
            if stem_pattern is not None and stem_pattern.search(text):
                score += 1

            if score > 0:
                scored.append((score, rel_path))

        scored.sort(key=lambda x: (-x[0], x[1]))

        selected = [rel for _, rel in scored[: self.max_relevant_tests]]
        if not selected:
            source_path = PurePosixPath(file_path)
            source_rel = source_path.as_posix()
            if source_rel in tests and is_test_file(source_path):
                selected = [source_rel]
            else:
                same_dir = [
                    rel
                    for rel in tests
                    if PurePosixPath(rel).parent == source_path.parent
                ]
                if same_dir:
                    selected = sorted(same_dir)[: self.max_relevant_tests]
                elif source_stem:
                    stem_matches = [
                        rel
                        for rel in tests
                        if source_stem.lower() in PurePosixPath(rel).stem.lower()
                    ]
                    selected = sorted(stem_matches)[: self.max_relevant_tests]

        self._selection_cache[cache_key] = selected
        return selected

    def _build_patched_function(
        self,
        masked_function: str,
        predicted_body: str,
    ) -> str:
        signature = make_signature(masked_function)
        body = predicted_body.rstrip()
        if not body:
            return signature
        return signature + "\n" + body

    def _patch_file_text(
        self,
        original_text: str,
        start_line: int,
        end_line: int,
        masked_function: str,
        predicted_body: str,
    ) -> str:
        lines = original_text.splitlines()
        replacement = self._build_patched_function(masked_function, predicted_body)
        replacement_lines = replacement.splitlines()

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        patched_lines = lines[:start_idx] + replacement_lines + lines[end_idx:]
        patched = "\n".join(patched_lines)
        if original_text.endswith("\n"):
            patched += "\n"
        return patched

    def evaluate_record(
        self,
        rec: dict,
        *,
        exact_match: int,
        syntax_valid: bool,
    ) -> PassAtOneResult:
        if self.mode == "exact_only":
            return PassAtOneResult(
                pass_at_1=int(exact_match),
                source="exact_match_only",
                status="mode_exact_only",
                selected_test_count=0,
            )

        if not syntax_valid:
            return PassAtOneResult(
                pass_at_1=0,
                source="executed",
                status="syntax_invalid",
                selected_test_count=0,
            )

        if not self.pytest_available:
            if not self._warned_missing_pytest:
                print(
                    "Warning: pytest not installed in current interpreter; "
                    "pass@1 falls back to exact match."
                )
                self._warned_missing_pytest = True
            return self._fallback(exact_match, "pytest_not_installed")

        instance_id = str(rec.get("instance_id", ""))
        meta = self.instances.get(instance_id)
        if meta is None:
            return self._fallback(exact_match, "missing_instance_meta")

        repo_name = str(rec.get("repo") or meta.repo)
        repo_root = repo_root_for_name(self.repos_root, repo_name)
        if not repo_root.exists():
            return self._fallback(exact_match, "missing_repo_clone")

        target_file = repo_root / meta.file_path
        if not target_file.exists():
            return self._fallback(exact_match, "missing_target_file")

        selected_tests = self._select_relevant_tests(
            repo_name=repo_name,
            file_path=meta.file_path,
            function_name=meta.function_name,
        )
        if not selected_tests:
            return self._fallback(exact_match, "no_relevant_tests")

        cache_key = (repo_name, tuple(selected_tests))
        baseline_rc = self._baseline_result_cache.get(cache_key)
        if baseline_rc is None:
            baseline_rc, _ = self._run_pytest(
                repo_root=repo_root,
                selected_tests=selected_tests,
            )
            self._baseline_result_cache[cache_key] = baseline_rc

        if baseline_rc == -1:
            return self._fallback(
                exact_match,
                "baseline_timeout",
                selected_test_count=len(selected_tests),
            )
        if baseline_rc == 5:
            return self._fallback(
                exact_match,
                "baseline_no_tests_collected",
                selected_test_count=len(selected_tests),
            )
        if baseline_rc != 0:
            return self._fallback(
                exact_match,
                f"baseline_pytest_error_{baseline_rc}",
                selected_test_count=len(selected_tests),
            )

        try:
            original_text = target_file.read_text(encoding="utf-8", errors="ignore")
            patched_text = self._patch_file_text(
                original_text=original_text,
                start_line=meta.start_line,
                end_line=meta.end_line,
                masked_function=meta.masked_function,
                predicted_body=str(rec.get("predicted_body", "")),
            )
        except OSError:
            return self._fallback(exact_match, "io_error")

        target_file.write_text(patched_text, encoding="utf-8")
        try:
            patched_rc, patched_output = self._run_pytest(
                repo_root=repo_root,
                selected_tests=selected_tests,
            )
            if patched_rc == -1:
                return PassAtOneResult(
                    pass_at_1=0,
                    source="executed",
                    status="timeout",
                    selected_test_count=len(selected_tests),
                )
        finally:
            target_file.write_text(original_text, encoding="utf-8")

        if patched_rc == 0:
            return PassAtOneResult(
                pass_at_1=1,
                source="executed",
                status="passed",
                selected_test_count=len(selected_tests),
            )
        if patched_rc == 1:
            if "No module named pytest" in patched_output:
                return self._fallback(
                    exact_match,
                    "pytest_not_installed",
                    selected_test_count=len(selected_tests),
                )
            return PassAtOneResult(
                pass_at_1=0,
                source="executed",
                status="failed",
                selected_test_count=len(selected_tests),
            )
        if patched_rc == 5:
            return self._fallback(
                exact_match,
                "no_tests_collected",
                selected_test_count=len(selected_tests),
            )
        return self._fallback(
            exact_match,
            f"pytest_error_{patched_rc}",
            selected_test_count=len(selected_tests),
        )


def evaluate_condition(
    condition: str,
    completions_dir: Path,
    pass_executor: PassAtOneExecutor,
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
            pass_result = pass_executor.evaluate_record(
                rec,
                exact_match=exact,
                syntax_valid=bool(valid),
            )

            rows.append(
                {
                    "instance_id": rec["instance_id"],
                    "condition": condition,
                    "repo": rec.get("repo", ""),
                    "file_path": rec.get("file_path", ""),
                    "exact_match": exact,
                    "pass_at_1": pass_result.pass_at_1,
                    "pass_at_1_source": pass_result.source,
                    "pass_at_1_status": pass_result.status,
                    "selected_test_count": pass_result.selected_test_count,
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
            "exact_match_mean": 0.0,
            "pass_at_1_mean": 0.0,
            "bleu4_mean": 0.0,
            "syntax_valid_rate": 0.0,
            "mean_generation_time_s": 0.0,
            "mean_context_token_count": 0.0,
            "mean_generated_token_count": 0.0,
            "max_new_token_hit_rate": 0.0,
            "mean_selected_test_count": 0.0,
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
        "mean_selected_test_count": float(df["selected_test_count"].mean()),
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
        choices=["pytest_heuristic", "exact_only"],
        default="pytest_heuristic",
        help="How to compute pass@1: targeted pytest execution or exact-match-only fallback.",
    )
    p.add_argument(
        "--test-timeout-seconds",
        type=int,
        default=30,
        help="Per-instance timeout for targeted pytest execution.",
    )
    p.add_argument(
        "--max-relevant-tests",
        type=int,
        default=8,
        help="Cap on selected relevant test files per instance.",
    )
    p.add_argument(
        "--instances-jsonl",
        type=Path,
        default=INSTANCES_JSONL,
        help="Path to Stage 0 instances.jsonl with span metadata.",
    )
    p.add_argument(
        "--stage0-repos-dir",
        type=Path,
        default=STAGE0_DIR / "data" / "repos",
        help="Root directory containing cloned Stage 0 repos.",
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
    pass_executor = PassAtOneExecutor(
        mode=args.pass_at_1_mode,
        repos_root=args.stage0_repos_dir,
        instances=instance_meta,
        test_timeout_seconds=args.test_timeout_seconds,
        max_relevant_tests=args.max_relevant_tests,
    )

    conds = CONDITIONS if args.condition == "all" else [args.condition]
    all_summaries = []
    dfs: dict[str, pd.DataFrame] = {}

    for cond in conds:
        df, summary = evaluate_condition(cond, paths.completions, pass_executor)
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
