#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import statistics
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from common import read_jsonl, write_csv, write_json, write_jsonl
from config import (
    MAX_BODY_LINES,
    MIN_BODY_LINES,
    MIN_FILE_TOKENS,
    OUTPUTS_DIR,
    REPOS_CACHE_DIR,
    TOKENIZER_ID,
    TOP_CANDIDATES_PER_INSTANCE,
    ensure_stage0_dirs,
)
from patch_utils import parse_touched_files
from repo_utils import ensure_repo_checkout
from transformers import AutoTokenizer
from tree_sitter_languages import get_parser

SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "__pycache__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate Python function candidates for Stage 0."
    )
    parser.add_argument(
        "--instances-jsonl", default=str(OUTPUTS_DIR / "01_filtered_instances.jsonl")
    )
    parser.add_argument("--tokenizer-id", default=TOKENIZER_ID)
    parser.add_argument("--min-file-tokens", type=int, default=MIN_FILE_TOKENS)
    parser.add_argument("--min-body-lines", type=int, default=MIN_BODY_LINES)
    parser.add_argument("--max-body-lines", type=int, default=MAX_BODY_LINES)
    parser.add_argument(
        "--top-k-per-instance", type=int, default=TOP_CANDIDATES_PER_INSTANCE
    )
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--debug-every", type=int, default=10)
    return parser.parse_args()


def tree_sitter_function_positions(
    source_text: str, parser
) -> list[tuple[int, int, int]]:
    tree = parser.parse(source_text.encode("utf-8"))
    out: list[tuple[int, int, int]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            out.append(
                (node.start_point[0] + 1, node.start_point[1], node.end_point[0] + 1)
            )
        stack.extend(node.children)
    return out


def iter_python_files(repo_dir: Path):
    for path in repo_dir.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.is_file():
            yield path


def collect_defined_names(module: ast.Module) -> set[str]:
    names: set[str] = set()

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def collect_local_defs(fn_node: ast.AST) -> set[str]:
    local: set[str] = set()
    if isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for arg in (
            list(fn_node.args.args)
            + list(fn_node.args.kwonlyargs)
            + list(fn_node.args.posonlyargs)
        ):
            local.add(arg.arg)
        if fn_node.args.vararg:
            local.add(fn_node.args.vararg.arg)
        if fn_node.args.kwarg:
            local.add(fn_node.args.kwarg.arg)

    for node in ast.walk(fn_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local.add(node.id)
    return local


def collect_load_names(fn_node: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.add(node.id)
    return out


def file_token_count(tokenizer, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    return len(encoded["input_ids"])


def main() -> None:
    args = parse_args()
    ensure_stage0_dirs()

    instances = read_jsonl(Path(args.instances_jsonl))
    if args.max_instances > 0:
        instances = instances[: args.max_instances]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Language\(path, name\) is deprecated.*",
            category=FutureWarning,
        )
        parser = get_parser("python")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
    tokenizer.model_max_length = int(1e12)

    all_candidates: list[dict] = []
    ranked_rows: list[dict] = []
    stats_rows: list[dict] = []

    for idx, inst in enumerate(instances, start=1):
        if args.debug_every > 0 and idx % args.debug_every == 0:
            print(f"[{idx}/{len(instances)}] processing {inst['instance_id']}")

        repo = inst["repo"]
        base_commit = inst["base_commit"]
        touched_files = parse_touched_files(inst.get("patch") or "")

        try:
            repo_dir = ensure_repo_checkout(repo, base_commit, REPOS_CACHE_DIR)
        except Exception as exc:  # noqa: BLE001
            stats_rows.append(
                {
                    "instance_id": inst["instance_id"],
                    "repo": repo,
                    "candidate_count": 0,
                    "best_external_reference_count": 0,
                    "error": f"repo_checkout_failed: {type(exc).__name__}",
                }
            )
            continue

        instance_candidates: list[dict] = []

        for py_path in iter_python_files(repo_dir):
            rel_path = py_path.relative_to(repo_dir).as_posix()
            if rel_path in touched_files:
                continue

            try:
                text = py_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue

            tok_count = file_token_count(tokenizer, text)
            if tok_count <= args.min_file_tokens:
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=SyntaxWarning)
                    module = ast.parse(text)
            except SyntaxError:
                continue

            ts_positions = tree_sitter_function_positions(text, parser)
            if not ts_positions:
                continue

            ast_map: dict[tuple[int, int], ast.AST] = {}
            for node in ast.walk(module):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ast_map[(node.lineno, node.col_offset)] = node

            file_defined_names = collect_defined_names(module)

            for start_line, start_col, end_line in ts_positions:
                node = ast_map.get((start_line, start_col))
                if node is None or not getattr(node, "body", None):
                    continue
                if not node.body:
                    continue

                body_start = node.body[0].lineno
                body_end = node.end_lineno or end_line
                body_line_count = body_end - body_start + 1
                if (
                    body_line_count < args.min_body_lines
                    or body_line_count > args.max_body_lines
                ):
                    continue

                local_defs = collect_local_defs(node)
                load_names = collect_load_names(node)
                external_refs = sorted(
                    (load_names - local_defs - {node.name}) & file_defined_names
                )
                if not external_refs:
                    continue

                candidate = {
                    "instance_id": inst["instance_id"],
                    "repo": repo,
                    "created_at": inst.get("created_at"),
                    "base_commit": base_commit,
                    "environment_setup_commit": inst.get("environment_setup_commit"),
                    "file_path": rel_path,
                    "function_name": node.name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "col_offset": start_col,
                    "body_start_line": body_start,
                    "body_end_line": body_end,
                    "body_line_count": body_line_count,
                    "file_token_count": tok_count,
                    "external_reference_count": len(external_refs),
                    "external_references": external_refs,
                }
                instance_candidates.append(candidate)

        instance_candidates.sort(
            key=lambda row: (
                -row["external_reference_count"],
                -row["file_token_count"],
                -row["body_line_count"],
                row["file_path"],
                row["start_line"],
            )
        )

        all_candidates.extend(instance_candidates)
        top_candidates = instance_candidates[: args.top_k_per_instance]

        ranked_rows.append(
            {
                "instance_id": inst["instance_id"],
                "repo": repo,
                "created_at": inst.get("created_at"),
                "base_commit": base_commit,
                "environment_setup_commit": inst.get("environment_setup_commit"),
                "docker_image": inst.get("docker_image"),
                "image_name": inst.get("image_name"),
                "install_config": inst.get("install_config") or {},
                "test_cmd": inst.get("test_cmd", ""),
                "FAIL_TO_PASS": inst.get("FAIL_TO_PASS") or [],
                "PASS_TO_PASS": inst.get("PASS_TO_PASS") or [],
                "gold_patch": inst.get("patch") or "",
                "candidate_count": len(instance_candidates),
                "top_candidates": top_candidates,
            }
        )

        stats_rows.append(
            {
                "instance_id": inst["instance_id"],
                "repo": repo,
                "candidate_count": len(instance_candidates),
                "best_external_reference_count": (
                    top_candidates[0]["external_reference_count"]
                    if top_candidates
                    else 0
                ),
                "error": "",
            }
        )

    output_all = OUTPUTS_DIR / "02_function_candidates.jsonl"
    output_ranked = OUTPUTS_DIR / "02_ranked_instance_candidates.jsonl"
    output_stats_csv = OUTPUTS_DIR / "02_candidate_stats.csv"
    output_hist_png = OUTPUTS_DIR / "02_candidate_count_hist.png"
    output_summary = OUTPUTS_DIR / "02_candidate_summary.json"

    write_jsonl(output_all, all_candidates)
    write_jsonl(output_ranked, ranked_rows)
    write_csv(
        output_stats_csv,
        stats_rows,
        fieldnames=[
            "instance_id",
            "repo",
            "candidate_count",
            "best_external_reference_count",
            "error",
        ],
    )

    counts = [row["candidate_count"] for row in stats_rows]
    plt.figure(figsize=(7, 4))
    if counts:
        plt.hist(counts, bins=min(25, max(5, len(set(counts)))))
    plt.xlabel("Candidates per instance")
    plt.ylabel("Frequency")
    plt.title("Stage 0 function candidate distribution")
    plt.tight_layout()
    plt.savefig(output_hist_png, dpi=160)
    plt.close()

    nonzero = [x for x in counts if x > 0]
    summary = {
        "instances_processed": len(instances),
        "instances_with_candidates": sum(1 for x in counts if x > 0),
        "total_candidates": len(all_candidates),
        "candidate_count_mean": statistics.mean(counts) if counts else 0,
        "candidate_count_median": statistics.median(counts) if counts else 0,
        "candidate_count_nonzero_mean": statistics.mean(nonzero) if nonzero else 0,
        "outputs": {
            "all_candidates_jsonl": str(output_all),
            "ranked_instances_jsonl": str(output_ranked),
            "candidate_stats_csv": str(output_stats_csv),
            "candidate_hist_plot": str(output_hist_png),
        },
    }
    write_json(output_summary, summary)

    print(f"Instances processed: {len(instances)}")
    print(f"Instances with candidates: {summary['instances_with_candidates']}")
    print(f"Total candidates: {len(all_candidates)}")


if __name__ == "__main__":
    main()
