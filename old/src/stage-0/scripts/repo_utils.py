from __future__ import annotations

import subprocess
from pathlib import Path

from common import slugify_repo


def run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def git_has_commit(repo_dir: Path, commit: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def ensure_repo_checkout(repo: str, commit: str, repos_cache_dir: Path) -> Path:
    repo_dir = repos_cache_dir / slugify_repo(repo)
    remote_url = f"https://github.com/{repo}.git"

    if not repo_dir.exists():
        run_git(["clone", "--filter=blob:none", remote_url, str(repo_dir)])

    run_git(["remote", "set-url", "origin", remote_url], cwd=repo_dir)
    run_git(["fetch", "origin", "--prune"], cwd=repo_dir)

    if not git_has_commit(repo_dir, commit):
        run_git(["fetch", "origin", commit, "--depth", "1"], cwd=repo_dir)

    run_git(["checkout", "-f", commit], cwd=repo_dir)
    run_git(["clean", "-fd"], cwd=repo_dir)
    return repo_dir
