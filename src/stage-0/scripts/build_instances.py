from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tiktoken
from config import (
    DEFAULT_CONFIG,
    INSTANCES_DIR,
    OUTPUTS_DIR,
    REPOS_DIR,
    config_as_json_dict,
    ensure_stage_dirs,
)
from tree_sitter import Node
from tree_sitter_languages import get_parser

ENCODING = tiktoken.get_encoding("cl100k_base")
PARSER = get_parser("python")
SKIP_PATH_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


class AssignedNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)
        self.generic_visit(node)


class LoadedNameCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)
        self.generic_visit(node)


class ModuleAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.module_level_names: set[str] = set()
        self.class_attribute_names: set[str] = set()
        self.imported_modules: set[str] = set()
        self.functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.module_level_names.add(stmt.name)
            elif isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.module_level_names.add(target.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self.module_level_names.add(stmt.target.id)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    self.imported_modules.add(alias.name)
            elif isinstance(stmt, ast.ImportFrom) and stmt.module:
                self.imported_modules.add(stmt.module)

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        self.class_attribute_names.add(target.id)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self.class_attribute_names.add(stmt.target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.functions.append(node)
        self.generic_visit(node)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage 0 function completion instances"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=OUTPUTS_DIR / "repo_candidates.json",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=OUTPUTS_DIR / "instances.jsonl",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=OUTPUTS_DIR / "instances_summary.json",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=0,
        help="Optional cap to process fewer cloned repos",
    )
    return parser.parse_args()


def token_count(text: str) -> int:
    return len(ENCODING.encode(text))


def iter_python_files(repo_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in repo_path.rglob("*.py"):
        if not path.is_file():
            continue
        if any(part in SKIP_PATH_PARTS for part in path.parts):
            continue
        files.append(path)
    return files


def walk_nodes(node: Node) -> list[Node]:
    out: list[Node] = []
    stack = [node]
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(reversed(current.children))
    return out


def ast_function_index(
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> dict[tuple[int, int, str], ast.FunctionDef | ast.AsyncFunctionDef]:
    index: dict[tuple[int, int, str], ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for fn in functions:
        if getattr(fn, "end_lineno", None) is None:
            continue
        index[(fn.lineno, fn.end_lineno, fn.name)] = fn
    return index


def function_local_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    local_names: set[str] = set()

    for arg in (
        list(fn.args.posonlyargs)
        + list(fn.args.args)
        + list(fn.args.kwonlyargs)
        + ([fn.args.vararg] if fn.args.vararg else [])
        + ([fn.args.kwarg] if fn.args.kwarg else [])
    ):
        if arg is not None:
            local_names.add(arg.arg)

    assigned_collector = AssignedNameCollector()
    assigned_collector.visit(fn)
    local_names |= assigned_collector.names
    return local_names


def references_file_internal_symbol(
    fn: ast.FunctionDef | ast.AsyncFunctionDef,
    file_symbols: set[str],
) -> tuple[bool, list[str]]:
    loaded_collector = LoadedNameCollector()
    loaded_collector.visit(fn)

    local_names = function_local_names(fn)
    disallowed = local_names | {fn.name}
    refs = sorted(
        name
        for name in loaded_collector.names
        if name in file_symbols and name not in disallowed
    )
    return (len(refs) > 0, refs)


def signature_and_body_from_node(
    file_text: str, node: Node
) -> tuple[str, str, int] | None:
    body_node = node.child_by_field_name("body")
    if body_node is None:
        return None

    lines = file_text.splitlines()
    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    body_start_line = body_node.start_point[0] + 1

    if start_line < 1 or end_line > len(lines) or body_start_line > len(lines):
        return None

    signature_lines = lines[start_line - 1 : body_start_line - 1]
    body_lines = lines[body_start_line - 1 : end_line]
    if not signature_lines or not body_lines:
        return None

    indent = ""
    stripped = body_lines[0].lstrip(" \t")
    if len(body_lines[0]) > len(stripped):
        indent = body_lines[0][: len(body_lines[0]) - len(stripped)]
    else:
        indent = "    "

    signature = "\n".join(signature_lines).rstrip()
    body = "\n".join(body_lines).rstrip()
    masked = f"{signature}\n{indent}pass"
    body_line_count = len(body_lines)
    return masked, body, body_line_count


def resolve_module_to_files(repo_root: Path, module: str) -> list[str]:
    parts = module.split(".")
    base = repo_root.joinpath(*parts)
    candidates = [base.with_suffix(".py"), base / "__init__.py"]

    resolved: list[str] = []
    for path in candidates:
        if path.exists() and path.is_file():
            resolved.append(str(path.relative_to(repo_root)))
    return resolved


def analyze_file(
    repo_name: str, repo_root: Path, file_path: Path, file_index: int
) -> list[dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if token_count(text) <= DEFAULT_CONFIG.truncation_token_budget:
        return []

    try:
        parsed_ast = ast.parse(text)
    except SyntaxError:
        return []

    analyzer = ModuleAnalyzer()
    analyzer.visit(parsed_ast)

    file_symbols = analyzer.module_level_names | analyzer.class_attribute_names
    fn_index = ast_function_index(analyzer.functions)

    tree = PARSER.parse(text.encode("utf-8"))
    nodes = [n for n in walk_nodes(tree.root_node) if n.type == "function_definition"]

    imports_context: list[dict[str, Any]] = []
    for module in sorted(analyzer.imported_modules):
        resolved = resolve_module_to_files(repo_root, module)
        imports_context.append({"module": module, "resolved_files": resolved})

    results: list[dict[str, Any]] = []
    relative_path = str(file_path.relative_to(repo_root))

    for fn_counter, node in enumerate(nodes, start=1):
        name_node = node.child_by_field_name("name")
        if name_node is None:
            continue

        fn_name = text[name_node.start_byte : name_node.end_byte]
        fn_start = node.start_point[0] + 1
        fn_end = node.end_point[0] + 1

        ast_fn = fn_index.get((fn_start, fn_end, fn_name))
        if ast_fn is None:
            continue

        extracted = signature_and_body_from_node(text, node)
        if extracted is None:
            continue
        masked, body, body_lines = extracted

        if (
            body_lines < DEFAULT_CONFIG.min_body_lines
            or body_lines > DEFAULT_CONFIG.max_body_lines
        ):
            continue

        has_external_refs, ref_names = references_file_internal_symbol(
            ast_fn, file_symbols
        )
        if not has_external_refs:
            continue

        instance_id = f"{repo_name}__{file_index:04d}__{fn_counter:03d}"
        results.append(
            {
                "instance_id": instance_id,
                "repo": repo_name,
                "file_path": relative_path,
                "function_name": fn_name,
                "start_line": fn_start,
                "end_line": fn_end,
                "body_line_count": body_lines,
                "file_token_count": token_count(text),
                "referenced_file_symbols": ref_names,
                "full_file": text,
                "masked_function": masked,
                "ground_truth_body": body,
                "cross_file_context": imports_context,
            }
        )

    return results


def round_robin_select(
    candidates: list[dict[str, Any]], max_instances: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[item["repo"]].append(item)

    for repo in grouped:
        grouped[repo].sort(
            key=lambda x: (x["body_line_count"], x["file_token_count"]), reverse=True
        )

    selected: list[dict[str, Any]] = []
    repos = sorted(grouped.keys())

    while len(selected) < max_instances:
        made_progress = False
        for repo in repos:
            if grouped[repo]:
                selected.append(grouped[repo].pop(0))
                made_progress = True
                if len(selected) >= max_instances:
                    break
        if not made_progress:
            break

    return selected


def write_instance_artifacts(instances: list[dict[str, Any]]) -> None:
    for item in instances:
        instance_dir = INSTANCES_DIR / item["instance_id"]
        instance_dir.mkdir(parents=True, exist_ok=True)
        (instance_dir / "full_file.py").write_text(item["full_file"], encoding="utf-8")
        (instance_dir / "masked_function.py").write_text(
            item["masked_function"] + "\n", encoding="utf-8"
        )
        (instance_dir / "ground_truth_body.py").write_text(
            item["ground_truth_body"] + "\n", encoding="utf-8"
        )
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"full_file", "masked_function", "ground_truth_body"}
        }
        (instance_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )


def main() -> None:
    args = parse_args()
    ensure_stage_dirs()

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    repos = payload.get("selected_repos", [])
    if args.max_repos > 0:
        repos = repos[: args.max_repos]

    all_candidates: list[dict[str, Any]] = []

    for repo_meta in repos:
        repo_name = repo_meta["full_name"]
        local_name = repo_name.replace("/", "__")
        repo_root = REPOS_DIR / local_name
        if not repo_root.exists():
            print(f"Skipping {repo_name}: repo not cloned at {repo_root}")
            continue

        py_files = iter_python_files(repo_root)
        print(f"Analyzing {repo_name}: {len(py_files)} python files")
        for idx, py_file in enumerate(py_files, start=1):
            try:
                file_instances = analyze_file(repo_name, repo_root, py_file, idx)
            except Exception as exc:  # noqa: BLE001
                print(f"Error analyzing {py_file}: {exc}")
                continue
            all_candidates.extend(file_instances)

    selected_instances = round_robin_select(
        all_candidates, DEFAULT_CONFIG.target_max_instances
    )
    write_instance_artifacts(selected_instances)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for item in selected_instances:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_repo: dict[str, int] = defaultdict(int)
    for item in selected_instances:
        by_repo[item["repo"]] += 1

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config_as_json_dict(DEFAULT_CONFIG),
        "candidate_instance_count": len(all_candidates),
        "selected_instance_count": len(selected_instances),
        "meets_target_min_instances": len(selected_instances)
        >= DEFAULT_CONFIG.target_min_instances,
        "selected_instances_by_repo": dict(sorted(by_repo.items())),
        "output_jsonl": str(args.output_jsonl),
        "instance_artifacts_dir": str(INSTANCES_DIR),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Built instances: "
        f"candidates={summary['candidate_instance_count']} "
        f"selected={summary['selected_instance_count']} "
        f"target_min={DEFAULT_CONFIG.target_min_instances}"
    )


if __name__ == "__main__":
    main()
