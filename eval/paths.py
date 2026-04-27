"""Repo-root path resolution shared between eval modules and legacy scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: Path | str | None) -> Path | None:
    """Resolve a path against the repo root.

    - ``None`` stays ``None``.
    - Absolute paths are returned unchanged (even if they do not yet exist).
    - Relative paths are interpreted as repo-root relative.
    """
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p
