from __future__ import annotations

import difflib
import re
from pathlib import Path

PATCH_FILE_RE = re.compile(r"^\+\+\+\s+b/(.+)$", re.MULTILINE)


def parse_touched_files(patch_text: str) -> set[str]:
    touched: set[str] = set()
    for match in PATCH_FILE_RE.finditer(patch_text or ""):
        touched.add(match.group(1).strip())
    return touched


def build_unified_patch(file_path: str, original_text: str, new_text: str) -> str:
    original_lines = original_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "\n".join(diff).strip() + "\n"


def combine_patches(*patches: str) -> str:
    chunks = [p.strip() for p in patches if p and p.strip()]
    if not chunks:
        return ""
    return "\n\n".join(chunks).strip() + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
