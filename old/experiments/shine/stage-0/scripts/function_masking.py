from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class FunctionMaskResult:
    masked_file_text: str
    ground_truth_body: str
    function_source: str


def _leading_ws(line: str) -> str:
    stripped = line.lstrip(" \t")
    return line[: len(line) - len(stripped)]


def _find_function_by_position(
    module: ast.AST, lineno: int, col_offset: int
) -> ast.AST | None:
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                getattr(node, "lineno", None) == lineno
                and getattr(node, "col_offset", None) == col_offset
            ):
                return node
    return None


def mask_function_by_position(
    source_text: str, lineno: int, col_offset: int
) -> FunctionMaskResult:
    module = ast.parse(source_text)
    node = _find_function_by_position(module, lineno=lineno, col_offset=col_offset)
    if node is None:
        raise ValueError(
            f"Function not found at lineno={lineno}, col_offset={col_offset}"
        )

    if not getattr(node, "body", None):
        raise ValueError("Function node has no body")
    if not node.body:
        raise ValueError("Function body is empty")

    lines = source_text.splitlines(keepends=True)
    body_start = node.body[0].lineno
    body_end = node.end_lineno
    if body_end is None:
        raise ValueError("Function end_lineno missing")

    first_stmt = node.body[0]
    keep_until = body_start - 1
    if isinstance(first_stmt, ast.Expr) and isinstance(
        getattr(first_stmt, "value", None), ast.Constant
    ):
        if isinstance(first_stmt.value.value, str):
            keep_until = first_stmt.end_lineno or keep_until

    replace_start = keep_until + 1

    if replace_start <= body_end:
        sample_line = lines[replace_start - 1] if replace_start - 1 < len(lines) else ""
    else:
        sample_line = lines[body_start - 1] if body_start - 1 < len(lines) else ""

    indent = _leading_ws(sample_line)
    if not indent:
        def_line = lines[node.lineno - 1]
        indent = _leading_ws(def_line) + "    "

    replacement = [f"{indent}pass\n"]

    masked_lines = lines[: replace_start - 1] + replacement + lines[body_end:]
    masked_text = "".join(masked_lines)

    function_source = "".join(lines[node.lineno - 1 : body_end])
    ground_truth_body = "".join(lines[body_start - 1 : body_end])

    return FunctionMaskResult(
        masked_file_text=masked_text,
        ground_truth_body=ground_truth_body,
        function_source=function_source,
    )
