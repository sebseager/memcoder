from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import UTC, datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import tiktoken
from config import (
    DEFAULT_CONFIG,
    INSTANCES_DIR,
    OUTPUTS_DIR,
    config_as_json_dict,
    ensure_stage_dirs,
)
from tqdm import tqdm
from tree_sitter import Node
from tree_sitter_languages import get_parser

ENCODING = tiktoken.get_encoding("cl100k_base")
PARSER = get_parser("python")
SKIP_PATH_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__"}
SUMMARY_COUNT_PATTERNS = {
    "passed": re.compile(r"(?P<count>\d+)\s+passed"),
    "failed": re.compile(r"(?P<count>\d+)\s+failed"),
    "errors": re.compile(r"(?P<count>\d+)\s+error"),
    "skipped": re.compile(r"(?P<count>\d+)\s+skipped"),
}
COMMON_SOURCE_ROOTS = {"src", "lib", "python", "package"}
IGNORED_TEST_PREFIXES = ("tests/integration/", "tests/e2e/")


def diagnostic_print(enabled: bool, message: str) -> None:
    if not enabled:
        return
    now = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{now} pid={os.getpid()}] {message}", flush=True)


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
        "--env-setup",
        type=Path,
        default=OUTPUTS_DIR / "env_setup.json",
    )
    parser.add_argument(
        "--test-coverage",
        type=Path,
        default=OUTPUTS_DIR / "test_coverage.json",
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
    parser.add_argument(
        "--max-candidates-per-repo",
        type=int,
        default=DEFAULT_CONFIG.max_candidates_per_repo,
        help="Cap expensive per-function testability checks per repo (0 means no cap)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(cpu_count(), 4)),
        help="Number of repo-level workers for expensive testability checks",
    )
    parser.add_argument(
        "--pytest-timeout-seconds",
        type=int,
        default=240,
        help="Hard timeout per pytest invocation during candidate scoring",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    parser.add_argument(
        "--no-debug-prints",
        action="store_true",
        help="Disable diagnostic prints from repo/candidate execution",
    )
    parser.add_argument(
        "--debug-every",
        type=int,
        default=25,
        help="Emit candidate-level heartbeat every N candidates per repo",
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


def normalize_rel_path(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root)).replace("\\", "/")


def is_test_file(path: Path, repo_root: Path) -> bool:
    rel_path = normalize_rel_path(path, repo_root)
    name = path.name.lower()
    return (
        rel_path.startswith("tests/")
        or "/tests/" in f"/{rel_path}"
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def is_ignored_test_selector(test_selector: str) -> bool:
    normalized = test_selector.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in IGNORED_TEST_PREFIXES)


def module_candidates_for_file(relative_path: str) -> set[str]:
    module_parts = list(Path(relative_path).with_suffix("").parts)
    if module_parts and module_parts[-1] == "__init__":
        module_parts = module_parts[:-1]

    candidates: set[str] = set()
    if module_parts:
        candidates.add(".".join(module_parts))
        candidates.add(module_parts[-1])

    for idx, part in enumerate(module_parts):
        if part in COMMON_SOURCE_ROOTS and idx + 1 < len(module_parts):
            candidates.add(".".join(module_parts[idx + 1 :]))

    return {item for item in candidates if item}


def import_signals_from_test_text(test_text: str) -> set[str]:
    try:
        module_ast = ast.parse(test_text)
    except SyntaxError:
        return set()

    signals: set[str] = set()
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = alias.name
                signals.add(imported)
                signals.add(imported.split(".")[-1])
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            module_name = node.module
            signals.add(module_name)
            for alias in node.names:
                if alias.name == "*":
                    continue
                signals.add(alias.name)
                signals.add(f"{module_name}.{alias.name}")

    return signals


def build_file_to_test_mapping(
    repo_root: Path,
    py_files: list[Path],
) -> tuple[dict[str, list[str]], dict[str, str]]:
    test_files = [
        path
        for path in py_files
        if is_test_file(path, repo_root)
        and not is_ignored_test_selector(normalize_rel_path(path, repo_root))
    ]
    source_files = [path for path in py_files if not is_test_file(path, repo_root)]

    source_candidates: dict[str, set[str]] = {}
    for source_file in source_files:
        rel_source = normalize_rel_path(source_file, repo_root)
        source_candidates[rel_source] = module_candidates_for_file(rel_source)

    mapping: dict[str, set[str]] = defaultdict(set)
    test_file_text_cache: dict[str, str] = {}

    for test_file in test_files:
        try:
            test_text = test_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue

        rel_test = normalize_rel_path(test_file, repo_root)
        test_file_text_cache[rel_test] = test_text
        import_signals = import_signals_from_test_text(test_text)
        if not import_signals:
            continue

        for rel_source, module_names in source_candidates.items():
            if module_names & import_signals:
                mapping[rel_source].add(rel_test)

    return (
        {rel_source: sorted(test_paths) for rel_source, test_paths in mapping.items()},
        test_file_text_cache,
    )


def keyword_filter_for_candidate(
    function_name: str,
    relevant_tests: list[str],
    test_file_text_cache: dict[str, str],
) -> str | None:
    candidate_name = function_name.strip()
    if len(candidate_name) < 3:
        return None

    boundary = re.compile(rf"\b{re.escape(candidate_name)}\b")
    for rel_test in relevant_tests:
        if boundary.search(test_file_text_cache.get(rel_test, "")):
            return candidate_name
    return None


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
) -> tuple[str, str, int, int, int, str] | None:
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
    return masked, body, body_line_count, body_start_line, end_line, indent


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
    repo_name: str,
    repo_local_name: str,
    repo_root: Path,
    file_path: Path,
    file_index: int,
) -> list[dict[str, Any]]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

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
        masked, body, body_lines, body_start_line, body_end_line, body_indent = (
            extracted
        )

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

        instance_id = f"{repo_local_name}__{file_index:04d}__{fn_counter:03d}"
        results.append(
            {
                "instance_id": instance_id,
                "repo": repo_name,
                "repo_local_name": repo_local_name,
                "file_path": relative_path,
                "function_name": fn_name,
                "start_line": fn_start,
                "end_line": fn_end,
                "body_start_line": body_start_line,
                "body_end_line": body_end_line,
                "body_indent": body_indent,
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


def parse_summary_counts(stdout: str, stderr: str) -> dict[str, int]:
    blob = f"{stdout}\n{stderr}"
    counts: dict[str, int] = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    for key, pattern in SUMMARY_COUNT_PATTERNS.items():
        match = pattern.search(blob)
        if match:
            counts[key] = int(match.group("count"))
    return counts


def testcase_node_id(test_case: ET.Element) -> str:
    name = test_case.attrib.get("name", "unknown")
    file_path = test_case.attrib.get("file")
    class_name = test_case.attrib.get("classname")
    if file_path:
        return f"{file_path}::{name}"
    if class_name:
        return f"{class_name}::{name}"
    return name


def parse_junit_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {
            "passed_tests": [],
            "failed_tests": [],
            "error_tests": [],
            "skipped_tests": [],
            "all_tests": [],
            "outcome_by_test": {},
            "had_report": False,
        }

    tree = ET.parse(report_path)
    root = tree.getroot()

    passed: set[str] = set()
    failed: set[str] = set()
    errored: set[str] = set()
    skipped: set[str] = set()
    outcome_by_test: dict[str, str] = {}

    for test_case in root.iter("testcase"):
        node_id = testcase_node_id(test_case)
        has_failure = any(child.tag.endswith("failure") for child in test_case)
        has_error = any(child.tag.endswith("error") for child in test_case)
        has_skipped = any(child.tag.endswith("skipped") for child in test_case)

        if has_error:
            errored.add(node_id)
            outcome_by_test[node_id] = "error"
        elif has_failure:
            failed.add(node_id)
            outcome_by_test[node_id] = "failed"
        elif has_skipped:
            skipped.add(node_id)
            outcome_by_test[node_id] = "skipped"
        else:
            passed.add(node_id)
            outcome_by_test[node_id] = "passed"

    all_tests = sorted(passed | failed | errored | skipped)
    return {
        "passed_tests": sorted(passed),
        "failed_tests": sorted(failed),
        "error_tests": sorted(errored),
        "skipped_tests": sorted(skipped),
        "all_tests": all_tests,
        "outcome_by_test": outcome_by_test,
        "had_report": True,
    }


def run_repo_pytest(
    repo_root: Path,
    pytest_bin: Path,
    junit_path: Path,
    timeout_seconds: int,
    *,
    test_selectors: list[str] | None = None,
    keyword_expression: str | None = None,
) -> dict[str, Any]:
    if junit_path.exists():
        junit_path.unlink()

    command = [
        str(pytest_bin),
        "--tb=no",
        "-q",
        "--timeout=30",
        "--import-mode=importlib",
        "--ignore=tests/integration",
        "--ignore=tests/e2e",
        f"--junitxml={junit_path}",
    ]

    if keyword_expression:
        command.extend(["-k", keyword_expression])

    if test_selectors:
        command.extend(
            selector
            for selector in test_selectors
            if not is_ignored_test_selector(selector)
        )

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True

    runtime_seconds = time.monotonic() - started

    parsed = parse_junit_report(junit_path)
    passed_tests = parsed["passed_tests"]
    failed_tests = parsed["failed_tests"]
    error_tests = parsed["error_tests"]
    skipped_tests = parsed["skipped_tests"]
    all_tests = parsed["all_tests"]
    outcome_by_test = parsed["outcome_by_test"]
    used_summary_fallback = False

    if not parsed["had_report"]:
        summary_counts = parse_summary_counts(stdout, stderr)
        used_summary_fallback = True
        pass_count = summary_counts["passed"]
        fail_count = summary_counts["failed"]
        error_count = summary_counts["errors"]
        skipped_count = summary_counts["skipped"]
    else:
        pass_count = len(passed_tests)
        fail_count = len(failed_tests)
        error_count = len(error_tests)
        skipped_count = len(skipped_tests)

    return {
        "exit_code": exit_code,
        "runtime_seconds": round(runtime_seconds, 3),
        "timed_out": timed_out,
        "command": command,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "error_tests": error_tests,
        "skipped_tests": skipped_tests,
        "all_tests": all_tests,
        "outcome_by_test": outcome_by_test,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "error_count": error_count,
        "skipped_count": skipped_count,
        "used_summary_fallback": used_summary_fallback,
        "stdout": stdout[-2000:],
        "stderr": stderr[-2000:],
    }


def replace_body_lines(
    full_text: str,
    body_start_line: int,
    body_end_line: int,
    replacement_lines: list[str],
) -> str:
    lines = full_text.splitlines()
    prefix = lines[: body_start_line - 1]
    suffix = lines[body_end_line:]
    rebuilt = prefix + replacement_lines + suffix
    rebuilt_text = "\n".join(rebuilt)
    if full_text.endswith("\n"):
        rebuilt_text += "\n"
    return rebuilt_text


def evaluate_candidate_testability(
    repo_task: dict[str, Any],
    candidate: dict[str, Any],
    coverage_map: dict[str, list[str]],
    test_file_text_cache: dict[str, str],
) -> dict[str, Any]:
    debug_enabled = bool(repo_task.get("debug_enabled", False))
    repo_root = Path(repo_task["repo_path"])
    pytest_bin = Path(repo_task["pytest_bin"])
    target_file = repo_root / candidate["file_path"]
    relevant_tests = coverage_map.get(candidate["file_path"], [])

    if not relevant_tests:
        skipped = dict(candidate)
        skipped.update(
            {
                "testable": False,
                "skip_reason": "no_relevant_tests_for_target_file",
                "scoped_test_targets": [],
                "scoped_test_count": 0,
            }
        )
        return skipped

    keyword_expression = keyword_filter_for_candidate(
        candidate.get("function_name", ""),
        relevant_tests,
        test_file_text_cache,
    )

    report_dir = repo_root / ".stage0_candidate_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    original_text = candidate["full_file"]
    masked_body_line = f"{candidate['body_indent']}pass"
    masked_text = replace_body_lines(
        original_text,
        candidate["body_start_line"],
        candidate["body_end_line"],
        [masked_body_line],
    )
    ground_truth_text = replace_body_lines(
        original_text,
        candidate["body_start_line"],
        candidate["body_end_line"],
        candidate["ground_truth_body"].splitlines(),
    )

    masked_report = report_dir / f"{candidate['instance_id']}_masked.xml"
    restored_report = report_dir / f"{candidate['instance_id']}_gt.xml"

    masked_result: dict[str, Any] = {}
    restored_result: dict[str, Any] = {}
    try:
        target_file.write_text(masked_text, encoding="utf-8")
        masked_result = run_repo_pytest(
            repo_root,
            pytest_bin,
            masked_report,
            timeout_seconds=int(repo_task["pytest_timeout_seconds"]),
            test_selectors=relevant_tests,
            keyword_expression=keyword_expression,
        )

        target_file.write_text(ground_truth_text, encoding="utf-8")
        restored_result = run_repo_pytest(
            repo_root,
            pytest_bin,
            restored_report,
            timeout_seconds=int(repo_task["pytest_timeout_seconds"]),
            test_selectors=relevant_tests,
            keyword_expression=keyword_expression,
        )
    finally:
        target_file.write_text(original_text, encoding="utf-8")

    excluded_tests = set(repo_task["baseline_excluded_tests"])
    before_pass = set(masked_result.get("passed_tests", [])) - excluded_tests
    after_pass = set(restored_result.get("passed_tests", [])) - excluded_tests
    gt_patch_test_delta = len(after_pass - before_pass)

    all_tests = (
        set(masked_result.get("all_tests", []))
        | set(restored_result.get("all_tests", []))
    ) - excluded_tests
    masked_outcomes = masked_result.get("outcome_by_test", {})
    restored_outcomes = restored_result.get("outcome_by_test", {})
    affected_tests = sorted(
        test_id
        for test_id in all_tests
        if masked_outcomes.get(test_id, "not_run")
        != restored_outcomes.get(test_id, "not_run")
    )

    testable = gt_patch_test_delta > 0
    if masked_result.get("exit_code") == 2 or restored_result.get("exit_code") == 2:
        testable = False

    masked_exit = int(masked_result.get("exit_code", -1))
    restored_exit = int(restored_result.get("exit_code", -1))
    masked_timed_out = bool(masked_result.get("timed_out", False))
    restored_timed_out = bool(restored_result.get("timed_out", False))

    if masked_timed_out or restored_timed_out:
        diagnostic_print(
            debug_enabled,
            (
                f"timeout repo={repo_task['local_name']} instance={candidate['instance_id']} "
                f"masked_exit={masked_exit} restored_exit={restored_exit} "
                f"tests={len(relevant_tests)}"
            ),
        )

    if masked_exit not in {0, 1, 5} or restored_exit not in {0, 1, 5}:
        diagnostic_print(
            debug_enabled,
            (
                f"nonstandard-exit repo={repo_task['local_name']} instance={candidate['instance_id']} "
                f"masked_exit={masked_exit} restored_exit={restored_exit} "
                f"tests={len(relevant_tests)}"
            ),
        )

    candidate = dict(candidate)
    candidate.update(
        {
            "affected_tests": affected_tests,
            "gt_patch_test_delta": gt_patch_test_delta,
            "testable": testable,
            "masked_test_exit_code": masked_result.get("exit_code"),
            "masked_test_runtime_seconds": masked_result.get("runtime_seconds"),
            "restored_test_exit_code": restored_result.get("exit_code"),
            "restored_test_runtime_seconds": restored_result.get("runtime_seconds"),
            "baseline_pass_count": repo_task["baseline_pass_count"],
            "baseline_fail_count": repo_task["baseline_fail_count"],
            "baseline_runtime_seconds": repo_task["baseline_runtime_seconds"],
            "env_python": f"{DEFAULT_CONFIG.env_python_version}.x",
            "env_install_method": repo_task["env_install_method"],
            "repo_weight": repo_task.get("repo_weight", 1.0),
            "selection_tier": repo_task.get("selection_tier", "strict"),
            "scoped_test_targets": relevant_tests,
            "scoped_test_count": len(relevant_tests),
            "function_name_filter": keyword_expression,
            "masked_test_timed_out": masked_timed_out,
            "restored_test_timed_out": restored_timed_out,
            "masked_test_command": " ".join(
                str(part) for part in masked_result.get("command", [])
            ),
            "restored_test_command": " ".join(
                str(part) for part in restored_result.get("command", [])
            ),
        }
    )
    return candidate


def process_repo(repo_task: dict[str, Any]) -> dict[str, Any]:
    repo_name = repo_task["full_name"]
    repo_local_name = repo_task["local_name"]
    repo_root = Path(repo_task["repo_path"])
    debug_enabled = bool(repo_task.get("debug_enabled", False))
    debug_every = max(1, int(repo_task.get("debug_every", 25)))
    process_started = time.monotonic()

    diagnostic_print(debug_enabled, f"repo-start {repo_local_name}")

    if not repo_root.exists():
        return {
            "repo": repo_name,
            "repo_local_name": repo_local_name,
            "error": f"Repo not found: {repo_root}",
            "candidate_count": 0,
            "evaluated_count": 0,
            "testable_count": 0,
            "mapped_source_file_count": 0,
            "mapped_test_file_count": 0,
            "instances": [],
        }

    pytest_bin = Path(repo_task["pytest_bin"])
    if not pytest_bin.exists():
        return {
            "repo": repo_name,
            "repo_local_name": repo_local_name,
            "error": f"Missing pytest executable at {pytest_bin}",
            "candidate_count": 0,
            "evaluated_count": 0,
            "testable_count": 0,
            "mapped_source_file_count": 0,
            "mapped_test_file_count": 0,
            "instances": [],
        }

    py_files = iter_python_files(repo_root)
    coverage_map, test_file_text_cache = build_file_to_test_mapping(repo_root, py_files)
    diagnostic_print(
        debug_enabled,
        (
            f"repo-indexed {repo_local_name} py_files={len(py_files)} "
            f"mapped_sources={len(coverage_map)} mapped_tests={len(test_file_text_cache)}"
        ),
    )

    candidates: list[dict[str, Any]] = []
    for idx, py_file in enumerate(py_files, start=1):
        try:
            file_instances = analyze_file(
                repo_name,
                repo_local_name,
                repo_root,
                py_file,
                idx,
            )
        except Exception:  # noqa: BLE001
            continue
        candidates.extend(file_instances)

    candidates.sort(
        key=lambda item: (item["body_line_count"], item["file_token_count"]),
        reverse=True,
    )

    max_candidates = int(repo_task.get("max_candidates_per_repo", 0))
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    diagnostic_print(
        debug_enabled,
        f"repo-candidates {repo_local_name} total={len(candidates)}",
    )

    testable_instances: list[dict[str, Any]] = []
    skipped_no_relevant = 0
    candidate_exception_count = 0
    timeout_count = 0
    nonstandard_exit_count = 0

    candidate_iterable = candidates
    if repo_task.get("show_candidate_progress", False):
        candidate_iterable = tqdm(
            candidates,
            total=len(candidates),
            desc=f"{repo_local_name}: candidates",
            leave=False,
            dynamic_ncols=True,
        )

    for idx, candidate in enumerate(candidate_iterable, start=1):
        try:
            evaluated = evaluate_candidate_testability(
                repo_task,
                candidate,
                coverage_map,
                test_file_text_cache,
            )
        except Exception as exc:  # noqa: BLE001
            candidate_with_error = dict(candidate)
            candidate_with_error["testable"] = False
            candidate_with_error["candidate_error"] = str(exc)
            candidate_with_error["candidate_traceback"] = traceback.format_exc()
            evaluated = candidate_with_error
            candidate_exception_count += 1
            diagnostic_print(
                debug_enabled,
                (
                    f"candidate-exception repo={repo_local_name} "
                    f"instance={candidate.get('instance_id', 'unknown')} error={exc}"
                ),
            )
            diagnostic_print(
                debug_enabled,
                candidate_with_error["candidate_traceback"].rstrip(),
            )

        if evaluated.get("skip_reason") == "no_relevant_tests_for_target_file":
            skipped_no_relevant += 1

        if evaluated.get("masked_test_timed_out") or evaluated.get(
            "restored_test_timed_out"
        ):
            timeout_count += 1

        masked_exit = evaluated.get("masked_test_exit_code")
        restored_exit = evaluated.get("restored_test_exit_code")
        if masked_exit is not None and restored_exit is not None:
            if masked_exit not in {0, 1, 5} or restored_exit not in {0, 1, 5}:
                nonstandard_exit_count += 1

        if evaluated.get("testable", False):
            testable_instances.append(evaluated)

        if idx == 1 or idx == len(candidates) or idx % debug_every == 0:
            diagnostic_print(
                debug_enabled,
                (
                    f"candidate-progress repo={repo_local_name} "
                    f"{idx}/{len(candidates)} testable={len(testable_instances)} "
                    f"skipped_no_tests={skipped_no_relevant} "
                    f"exceptions={candidate_exception_count} timeouts={timeout_count}"
                ),
            )

    runtime_seconds = round(time.monotonic() - process_started, 3)
    diagnostic_print(
        debug_enabled,
        (
            f"repo-done {repo_local_name} candidates={len(candidates)} "
            f"testable={len(testable_instances)} skipped_no_tests={skipped_no_relevant} "
            f"exceptions={candidate_exception_count} timeouts={timeout_count} "
            f"nonstandard_exits={nonstandard_exit_count} runtime_s={runtime_seconds}"
        ),
    )

    return {
        "repo": repo_name,
        "repo_local_name": repo_local_name,
        "error": None,
        "candidate_count": len(candidates),
        "evaluated_count": len(candidates),
        "testable_count": len(testable_instances),
        "mapped_source_file_count": len(coverage_map),
        "mapped_test_file_count": len(test_file_text_cache),
        "skipped_no_relevant_tests_count": skipped_no_relevant,
        "candidate_exception_count": candidate_exception_count,
        "candidate_timeout_count": timeout_count,
        "candidate_nonstandard_exit_count": nonstandard_exit_count,
        "runtime_seconds": runtime_seconds,
        "instances": testable_instances,
    }


def round_robin_select(
    candidates: list[dict[str, Any]],
    max_instances: int,
    repo_weights: dict[str, float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[item["repo"]].append(item)

    for repo in grouped:
        grouped[repo].sort(
            key=lambda x: (
                x.get("gt_patch_test_delta", 0),
                x["body_line_count"],
                x["file_token_count"],
            ),
            reverse=True,
        )

    selected: list[dict[str, Any]] = []
    repos = sorted(
        grouped.keys(), key=lambda repo: repo_weights.get(repo, 1.0), reverse=True
    )

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


def reset_instance_artifacts_dir() -> None:
    if INSTANCES_DIR.exists():
        shutil.rmtree(INSTANCES_DIR)
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)


def snapshot_repo_commit(repo_root: Path) -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return None

    eval_commit_path = repo_root / ".eval-commit"
    eval_commit_path.write_text(f"{commit}\n", encoding="utf-8")
    return commit


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
    progress_enabled = not args.no_progress
    debug_enabled = not args.no_debug_prints

    candidates_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    selected_repo_meta = {
        item["full_name"]: item for item in candidates_payload.get("selected_repos", [])
    }

    env_payload = json.loads(args.env_setup.read_text(encoding="utf-8"))
    env_map = {
        item["full_name"]: item for item in env_payload.get("qualified_repos", [])
    }

    coverage_payload = json.loads(args.test_coverage.read_text(encoding="utf-8"))
    coverage_repos = coverage_payload.get("qualified_repos", [])

    repo_tasks: list[dict[str, Any]] = []
    for coverage_meta in coverage_repos:
        full_name = coverage_meta["full_name"]
        env_meta = env_map.get(full_name)
        if env_meta is None:
            continue

        repo_path = Path(env_meta["repo_path"])
        venv_path = Path(env_meta["venv_path"])
        pytest_bin = venv_path / "bin" / "pytest"

        baseline_failed = coverage_meta.get("baseline_failed_tests", [])
        baseline_error = coverage_meta.get("baseline_error_tests", [])
        baseline_excluded = sorted(set(baseline_failed) | set(baseline_error))

        local_name = full_name.replace("/", "__")
        repo_tasks.append(
            {
                "full_name": full_name,
                "local_name": local_name,
                "repo_path": str(repo_path),
                "pytest_bin": str(pytest_bin),
                "env_install_method": env_meta.get("env_install_method"),
                "baseline_pass_count": int(coverage_meta.get("baseline_pass_count", 0)),
                "baseline_fail_count": int(coverage_meta.get("baseline_fail_count", 0)),
                "baseline_runtime_seconds": float(
                    coverage_meta.get("baseline_runtime_seconds", 0.0)
                ),
                "baseline_excluded_tests": baseline_excluded,
                "repo_weight": float(coverage_meta.get("repo_weight", 1.0)),
                "selection_tier": coverage_meta.get("selection_tier", "strict"),
                "max_candidates_per_repo": args.max_candidates_per_repo,
                "pytest_timeout_seconds": args.pytest_timeout_seconds,
                "show_candidate_progress": progress_enabled and args.workers <= 1,
                "debug_enabled": debug_enabled,
                "debug_every": args.debug_every,
                "service_dependency_flags": selected_repo_meta.get(full_name, {}).get(
                    "service_dependency_flags", []
                ),
            }
        )

    if args.max_repos > 0:
        repo_tasks = repo_tasks[: args.max_repos]

    if not repo_tasks:
        raise RuntimeError(
            "No repos available for instance building. Run discover, clone, "
            "setup_repo_env, and score_test_coverage first."
        )

    print(
        f"Running repo-level instance extraction on {len(repo_tasks)} repos "
        f"with workers={args.workers}"
    )

    if args.workers <= 1:
        repo_iterable = repo_tasks
        if progress_enabled:
            repo_iterable = tqdm(
                repo_tasks,
                total=len(repo_tasks),
                desc="Repos",
                dynamic_ncols=True,
            )
        repo_results: list[dict[str, Any]] = []
        for task in repo_iterable:
            result = process_repo(task)
            repo_results.append(result)
            diagnostic_print(
                debug_enabled,
                (
                    f"repo-result {result['repo_local_name']} "
                    f"testable={result['testable_count']} "
                    f"exceptions={result.get('candidate_exception_count', 0)} "
                    f"timeouts={result.get('candidate_timeout_count', 0)} "
                    f"runtime_s={result.get('runtime_seconds', 0.0)}"
                ),
            )
    else:
        with Pool(processes=args.workers) as pool:
            repo_iterable = pool.imap_unordered(process_repo, repo_tasks)
            if progress_enabled:
                repo_iterable = tqdm(
                    repo_iterable,
                    total=len(repo_tasks),
                    desc="Repos",
                    dynamic_ncols=True,
                )
            repo_results = []
            for result in repo_iterable:
                repo_results.append(result)
                diagnostic_print(
                    debug_enabled,
                    (
                        f"repo-result {result['repo_local_name']} "
                        f"testable={result['testable_count']} "
                        f"exceptions={result.get('candidate_exception_count', 0)} "
                        f"timeouts={result.get('candidate_timeout_count', 0)} "
                        f"runtime_s={result.get('runtime_seconds', 0.0)}"
                    ),
                )

    all_testable_candidates: list[dict[str, Any]] = []
    repo_candidate_counts: dict[str, int] = {}
    repo_testable_counts: dict[str, int] = {}
    repo_mapped_source_file_counts: dict[str, int] = {}
    repo_mapped_test_file_counts: dict[str, int] = {}
    repo_errors: dict[str, str] = {}

    for repo_result in repo_results:
        repo_name = repo_result["repo"]
        repo_candidate_counts[repo_name] = int(repo_result["candidate_count"])
        repo_testable_counts[repo_name] = int(repo_result["testable_count"])
        repo_mapped_source_file_counts[repo_name] = int(
            repo_result.get("mapped_source_file_count", 0)
        )
        repo_mapped_test_file_counts[repo_name] = int(
            repo_result.get("mapped_test_file_count", 0)
        )
        if repo_result.get("error"):
            repo_errors[repo_name] = str(repo_result["error"])
            continue
        all_testable_candidates.extend(repo_result["instances"])

    repo_weight_map = {
        task["full_name"]: float(task.get("repo_weight", 1.0)) for task in repo_tasks
    }
    selected_instances = round_robin_select(
        all_testable_candidates,
        DEFAULT_CONFIG.target_max_instances,
        repo_weight_map,
    )

    selected_repo_names = {item["repo"] for item in selected_instances}
    repo_commits: dict[str, str] = {}
    for task in repo_tasks:
        repo_name = task["full_name"]
        if repo_name not in selected_repo_names:
            continue
        commit = snapshot_repo_commit(Path(task["repo_path"]))
        if commit:
            repo_commits[repo_name] = commit

    for item in selected_instances:
        item["repo_commit"] = repo_commits.get(item["repo"])

    reset_instance_artifacts_dir()
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
        "candidate_instance_count": int(sum(repo_candidate_counts.values())),
        "testable_candidate_count": len(all_testable_candidates),
        "selected_instance_count": len(selected_instances),
        "meets_target_min_instances": len(selected_instances)
        >= DEFAULT_CONFIG.target_min_instances,
        "repo_candidate_counts": dict(sorted(repo_candidate_counts.items())),
        "repo_testable_counts": dict(sorted(repo_testable_counts.items())),
        "repo_mapped_source_file_counts": dict(
            sorted(repo_mapped_source_file_counts.items())
        ),
        "repo_mapped_test_file_counts": dict(
            sorted(repo_mapped_test_file_counts.items())
        ),
        "repo_errors": dict(sorted(repo_errors.items())),
        "repo_commits": dict(sorted(repo_commits.items())),
        "selected_instances_by_repo": dict(sorted(by_repo.items())),
        "output_jsonl": str(args.output_jsonl),
        "instance_artifacts_dir": str(INSTANCES_DIR),
        "inputs": {
            "candidates": str(args.candidates),
            "env_setup": str(args.env_setup),
            "test_coverage": str(args.test_coverage),
        },
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        "Built instances: "
        f"candidates={summary['candidate_instance_count']} "
        f"testable={summary['testable_candidate_count']} "
        f"selected={summary['selected_instance_count']} "
        f"target_min={DEFAULT_CONFIG.target_min_instances}"
    )


if __name__ == "__main__":
    main()
