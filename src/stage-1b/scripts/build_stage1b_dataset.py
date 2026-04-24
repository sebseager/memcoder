#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


SKIP_PARTS = {".git", ".venv", "venv", "node_modules", "build", "dist", "__pycache__"}


@dataclass(frozen=True)
class SliceArtifact:
    context_text: str
    covered_refs: list[str]
    missing_refs: list[str]
    coverage_fraction: float
    token_count: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    stage1b_root = repo_root / "src" / "stage-1b"
    stage0_instances = repo_root / "src" / "stage-0" / "outputs" / "stage1_instances.jsonl"
    stage1a_model = repo_root / "src" / "stage-1a" / "models" / "Qwen3-8B"
    p = argparse.ArgumentParser(
        description="Build SHINE ift-c1qa triples from SWE-rebench training rows."
    )
    p.add_argument("--dataset-name", default="nebius/SWE-rebench-leaderboard")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-instances-jsonl", type=Path, default=stage0_instances)
    p.add_argument("--tokenizer-path", default=str(stage1a_model))
    p.add_argument("--repo-cache-dir", type=Path, default=stage1b_root / "cache" / "repos")
    p.add_argument("--output-shine-json", type=Path, default=stage1b_root / "data" / "ift_c1qa_code_train.json")
    p.add_argument("--output-records-jsonl", type=Path, default=stage1b_root / "outputs" / "train_triples.records.jsonl")
    p.add_argument("--output-heldout-jsonl", type=Path, default=stage1b_root / "outputs" / "heldout_instances.jsonl")
    p.add_argument("--output-summary-json", type=Path, default=stage1b_root / "outputs" / "build_dataset.summary.json")
    p.add_argument("--max-triples", type=int, default=500)
    p.add_argument("--heldout", type=int, default=20)
    p.add_argument("--max-source-instances", type=int, default=0)
    p.add_argument("--max-functions-per-instance", type=int, default=3)
    p.add_argument("--context-max-tokens", type=int, default=1024)
    p.add_argument("--conversation-max-tokens", type=int, default=4096)
    p.add_argument("--min-file-tokens", type=int, default=2048)
    p.add_argument("--min-body-lines", type=int, default=10)
    p.add_argument("--max-body-lines", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug-every", type=int, default=25)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def repo_key(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "__", repo)


def ensure_repo_checkout(repo: str, commit: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = cache_dir / repo_key(repo)
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--no-tags", "--filter=blob:none", f"https://github.com/{repo}.git", str(repo_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    subprocess.run(
        ["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", commit],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--force", commit],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return repo_dir


def parse_touched_files(patch: str) -> set[str]:
    out: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
            if path != "/dev/null":
                out.add(path)
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[2].startswith("a/"):
                out.add(parts[2][2:])
    return out


def iter_python_files(repo_dir: Path):
    for path in repo_dir.rglob("*.py"):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.is_file():
            yield path


def tree_sitter_positions(source_text: str, parser: Any | None) -> list[tuple[int, int, int]]:
    if parser is None:
        return []
    tree = parser.parse(source_text.encode("utf-8"))
    out: list[tuple[int, int, int]] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "function_definition":
            out.append((node.start_point[0] + 1, node.start_point[1], node.end_point[0] + 1))
        stack.extend(node.children)
    return out


def ast_positions(module: ast.Module) -> list[tuple[int, int, int]]:
    return [
        (int(node.lineno), int(node.col_offset), int(getattr(node, "end_lineno", node.lineno)))
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


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
        for arg in list(fn_node.args.args) + list(fn_node.args.kwonlyargs) + list(fn_node.args.posonlyargs):
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
    return {node.id for node in ast.walk(fn_node) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}


def collect_called_names(fn_node: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def collect_attribute_names(fn_node: ast.AST) -> set[str]:
    return {node.attr for node in ast.walk(fn_node) if isinstance(node, ast.Attribute)}


def module_level_defs(module: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = node
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                out[alias.asname or alias.name] = node
    return out


def class_attr_defs(module: ast.Module) -> dict[str, dict[str, ast.AST]]:
    out: dict[str, dict[str, ast.AST]] = {}
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        attrs: dict[str, ast.AST] = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        attrs[target.id] = stmt
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                attrs[stmt.target.id] = stmt
        out[node.name] = attrs
    return out


def attach_parents(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            setattr(child, "_parent", node)


def node_source_segment(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node)
    if segment is not None:
        return segment
    lines = text.splitlines()
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(lines[start - 1 : end])


def function_signature_plus_doc(function_source: str, function_name: str) -> str:
    if not function_source.strip():
        return f"def {function_name}(...):"
    try:
        tree = ast.parse(textwrap.dedent(function_source))
    except SyntaxError:
        return function_source.strip()
    target = next(
        (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name),
        None,
    )
    if target is None:
        target = next((node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if target is None or not getattr(target, "body", None):
        return function_source.strip()
    lines = function_source.splitlines()
    first_body = int(target.body[0].lineno)
    sig = "\n".join(lines[int(target.lineno) - 1 : first_body - 1]).rstrip()
    doc = ast.get_docstring(target, clean=False) or ""
    if not doc:
        return sig
    return f'{sig}\n    """{doc.strip(chr(10))}"""'


def mask_function(source_text: str, node: ast.AST) -> tuple[str, str, str]:
    lines = source_text.splitlines(keepends=True)
    body_start = int(node.body[0].lineno)
    body_end = int(getattr(node, "end_lineno", body_start))
    first_stmt = node.body[0]
    keep_until = body_start - 1
    if isinstance(first_stmt, ast.Expr) and isinstance(getattr(first_stmt, "value", None), ast.Constant):
        if isinstance(first_stmt.value.value, str):
            keep_until = int(getattr(first_stmt, "end_lineno", keep_until))
    replace_start = keep_until + 1
    sample_line = lines[replace_start - 1] if replace_start - 1 < len(lines) else lines[body_start - 1]
    indent = re.match(r"^\s*", sample_line).group(0)
    if not indent:
        indent = re.match(r"^\s*", lines[int(node.lineno) - 1]).group(0) + "    "
    function_source = "".join(lines[int(node.lineno) - 1 : body_end])
    ground_truth_body = "".join(lines[body_start - 1 : body_end]).rstrip()
    masked_function = "".join(lines[int(node.lineno) - 1 : replace_start - 1] + [f"{indent}pass\n"]).rstrip()
    return function_source, ground_truth_body, masked_function


def file_token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_slice(row: dict[str, Any], tokenizer: Any, max_tokens: int) -> SliceArtifact:
    full_file = str(row.get("full_file", ""))
    refs = [str(x) for x in (row.get("external_references") or [])]
    fn_name = str(row.get("function_name", ""))
    start_line = int(row.get("start_line", 1))
    fn_source = str(row.get("function_source", ""))
    try:
        tree = ast.parse(full_file)
    except SyntaxError:
        text = fn_source[:4000]
        return SliceArtifact(text, [], refs, 0.0 if refs else 1.0, file_token_count(tokenizer, text))
    attach_parents(tree)
    target = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == fn_name
            and int(getattr(node, "lineno", -1)) == start_line
        ),
        None,
    )
    if target is None:
        target = next(
            (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name),
            None,
        )
    if target is None:
        text = fn_source[:4000]
        return SliceArtifact(text, [], refs, 0.0 if refs else 1.0, file_token_count(tokenizer, text))

    defs = module_level_defs(tree)
    class_defs = class_attr_defs(tree)
    sections: list[tuple[str, str]] = [("TARGET_SIGNATURE", function_signature_plus_doc(fn_source, fn_name).strip())]
    if refs:
        sections.append(("REFERENCE_NAME_INDEX", "Known in-file references:\n" + ", ".join(sorted(refs))))
    for name in sorted(collect_called_names(target)):
        node = defs.get(name)
        if node is not None and not isinstance(node, ast.ClassDef):
            sections.append(("ONE_HOP_CALLEE", node_source_segment(full_file, node)))
    for name in refs:
        node = defs.get(name)
        if node is not None:
            sections.append(("REFERENCE_SYMBOL", node_source_segment(full_file, node)))
    parent = getattr(target, "_parent", None)
    if isinstance(parent, ast.ClassDef):
        for attr in sorted(collect_attribute_names(target)):
            node = class_defs.get(parent.name, {}).get(attr)
            if node is not None:
                sections.append(("CLASS_ATTRIBUTE", node_source_segment(full_file, node)))

    selected = ""
    for idx, (title, body) in enumerate(sections):
        if not body.strip():
            continue
        block = f"# [{title}]\n{body.strip()}"
        candidate = block if idx == 0 else f"{selected}\n\n{block}"
        if file_token_count(tokenizer, candidate) <= max_tokens:
            selected = candidate
        elif not selected:
            ids = tokenizer.encode(block, add_special_tokens=False)[:max_tokens]
            selected = tokenizer.decode(ids, skip_special_tokens=True)
            break
        else:
            break
    covered = [r for r in refs if re.search(rf"\b{re.escape(r)}\b", selected)]
    missing = [r for r in refs if r not in covered]
    frac = 1.0 if not refs else len(covered) / len(refs)
    return SliceArtifact(selected, sorted(covered), missing, frac, file_token_count(tokenizer, selected))


def load_train_rows(dataset_name: str, split_name: str) -> list[dict[str, Any]]:
    try:
        ds = load_dataset(dataset_name, split=split_name)
        return [dict(row) for row in ds]
    except Exception:
        loaded = load_dataset(dataset_name)
        if isinstance(loaded, DatasetDict) and split_name in loaded:
            return [dict(row) for row in loaded[split_name]]
        if isinstance(loaded, DatasetDict):
            rows: list[dict[str, Any]] = []
            for name, split in loaded.items():
                if name.lower() not in {"test", "validation", "eval"}:
                    rows.extend(dict(row) for row in split)
            return rows
        return [dict(row) for row in loaded]


def make_question(sig_doc: str, masked_function: str) -> str:
    return (
        "Complete the missing Python function body.\n"
        "Return ONLY body lines with correct indentation. No markdown, no explanation.\n\n"
        f"Function signature + docstring:\n{sig_doc}\n\n"
        f"Masked function:\n{masked_function}"
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("build_stage1b_dataset")

    for path in (args.output_shine_json, args.output_records_jsonl, args.output_summary_json, args.output_heldout_jsonl):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")

    random.seed(args.seed)
    eval_rows = read_jsonl(args.eval_instances_jsonl)
    eval_repos = {str(row.get("repo", "")) for row in eval_rows if row.get("repo")}
    eval_ids = {str(row.get("instance_id", "")) for row in eval_rows if row.get("instance_id")}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, use_fast=True)

    try:
        from tree_sitter_languages import get_parser

        parser = get_parser("python")
    except Exception as exc:  # noqa: BLE001
        log.warning("tree-sitter unavailable; falling back to AST positions: %s", exc)
        parser = None

    raw_rows = load_train_rows(args.dataset_name, args.train_split)
    random.shuffle(raw_rows)
    if args.max_source_instances > 0:
        raw_rows = raw_rows[: args.max_source_instances]

    records: list[dict[str, Any]] = []
    shine_items: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    progress = tqdm(raw_rows, desc="Building Stage 1b triples", unit="instance")
    for idx, inst in enumerate(progress, start=1):
        if len(records) >= args.max_triples and len(heldout) >= args.heldout:
            break
        repo = str(inst.get("repo") or inst.get("repository") or "")
        base_commit = str(inst.get("base_commit") or inst.get("commit") or "")
        instance_id = str(inst.get("instance_id") or inst.get("id") or f"train-{idx}")
        if not repo or not base_commit or repo in eval_repos or instance_id in eval_ids:
            continue
        if inst.get("has_test_patch") is False or int(inst.get("num_modified_files") or 1) != 1:
            continue

        try:
            repo_dir = ensure_repo_checkout(repo, base_commit, args.repo_cache_dir)
        except Exception as exc:  # noqa: BLE001
            errors.append({"instance_id": instance_id, "repo": repo, "error": f"checkout_failed: {type(exc).__name__}"})
            continue

        candidates: list[dict[str, Any]] = []
        touched = parse_touched_files(str(inst.get("patch") or ""))
        for py_path in iter_python_files(repo_dir):
            rel_path = py_path.relative_to(repo_dir).as_posix()
            if rel_path in touched:
                continue
            try:
                source = py_path.read_text(encoding="utf-8")
                tok_count = file_token_count(tokenizer, source)
                if tok_count <= args.min_file_tokens:
                    continue
                module = ast.parse(source)
            except (UnicodeDecodeError, SyntaxError):
                continue
            ast_map = {
                (int(node.lineno), int(node.col_offset)): node
                for node in ast.walk(module)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            positions = tree_sitter_positions(source, parser) or ast_positions(module)
            defined = collect_defined_names(module)
            for start_line, start_col, end_line in positions:
                node = ast_map.get((start_line, start_col))
                if node is None or not getattr(node, "body", None):
                    continue
                body_start = int(node.body[0].lineno)
                body_end = int(getattr(node, "end_lineno", end_line))
                body_lines = body_end - body_start + 1
                if body_lines < args.min_body_lines or body_lines > args.max_body_lines:
                    continue
                external_refs = sorted((collect_load_names(node) - collect_local_defs(node) - {node.name}) & defined)
                if not external_refs:
                    continue
                function_source, ground_truth_body, masked_function = mask_function(source, node)
                row = {
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "file_path": rel_path,
                    "function_name": node.name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "body_line_count": body_lines,
                    "file_token_count": tok_count,
                    "external_reference_count": len(external_refs),
                    "external_references": external_refs,
                    "full_file": source,
                    "function_source": function_source,
                    "masked_function": masked_function,
                    "ground_truth_body": ground_truth_body,
                }
                candidates.append(row)

        candidates.sort(
            key=lambda r: (
                -int(r["external_reference_count"]),
                -int(r["file_token_count"]),
                -int(r["body_line_count"]),
                str(r["file_path"]),
                int(r["start_line"]),
            )
        )
        for cand in candidates[: args.max_functions_per_instance]:
            slice_art = build_slice(cand, tokenizer, args.context_max_tokens)
            sig_doc = function_signature_plus_doc(cand["function_source"], cand["function_name"])
            masked_function = cand["masked_function"]
            question = make_question(sig_doc, masked_function)
            conversations = [{"role": "user", "content": question}, {"role": "assistant", "content": cand["ground_truth_body"]}]
            try:
                conversation_ids = tokenizer.apply_chat_template(
                    conversations,
                    add_generation_prompt=False,
                    tokenize=True,
                    enable_thinking=False,
                )
                conversation_len = len(conversation_ids)
            except TypeError:
                conversation_len = file_token_count(tokenizer, question) + file_token_count(tokenizer, cand["ground_truth_body"])
            item = {
                "context": slice_art.context_text,
                "conversations": conversations,
                "contextlen": slice_art.token_count,
                "conversationlen": conversation_len,
            }
            rec = {
                **{k: v for k, v in cand.items() if k not in {"full_file"}},
                "masked_function": masked_function,
                "question": question,
                "slice_context": slice_art.context_text,
                "slice_token_count": slice_art.token_count,
                "slice_coverage_fraction": slice_art.coverage_fraction,
                "covered_external_references": slice_art.covered_refs,
                "missing_external_references": slice_art.missing_refs,
                "conversation_token_count": conversation_len,
            }
            if len(heldout) < args.heldout:
                heldout.append({**rec, "full_file": cand["full_file"]})
            else:
                records.append(rec)
                shine_items.append(item)
            if len(records) >= args.max_triples:
                break
        if args.debug_every > 0 and idx % args.debug_every == 0:
            log.info("[%d/%d] triples=%d heldout=%d errors=%d", idx, len(raw_rows), len(records), len(heldout), len(errors))

    write_json(args.output_shine_json, shine_items)
    write_jsonl(args.output_records_jsonl, records)
    write_jsonl(args.output_heldout_jsonl, heldout)
    summary = {
        "n_train_triples": len(records),
        "n_heldout": len(heldout),
        "n_source_rows_seen": len(raw_rows),
        "n_errors": len(errors),
        "repo_disjoint_eval_repos": len(eval_repos),
        "output_shine_json": str(args.output_shine_json),
        "output_records_jsonl": str(args.output_records_jsonl),
        "output_heldout_jsonl": str(args.output_heldout_jsonl),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        | {"eval_repos_excluded": sorted(eval_repos)[:50]},
        "errors_sample": errors[:20],
    }
    write_json(args.output_summary_json, summary)
    log.info("Wrote %d SHINE training items to %s", len(shine_items), args.output_shine_json)
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
