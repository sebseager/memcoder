from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from config import OUTPUTS_DIR, REPOS_DIR, ensure_stage_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone selected Stage 0 repositories")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=OUTPUTS_DIR / "repo_candidates.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of selected repos to clone",
    )
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    ensure_stage_dirs()

    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    repos = payload.get("selected_repos", [])
    if args.limit > 0:
        repos = repos[: args.limit]

    cloned = 0
    skipped = 0

    for repo in repos:
        full_name = repo["full_name"]
        local_name = full_name.replace("/", "__")
        local_path = REPOS_DIR / local_name
        if local_path.exists():
            skipped += 1
            continue

        clone_url = repo["clone_url"]
        print(f"Cloning {full_name} -> {local_path}")
        run(["git", "clone", "--depth", "1", clone_url, str(local_path)])
        cloned += 1

    print(f"Clone complete. cloned={cloned} skipped={skipped} total={len(repos)}")


if __name__ == "__main__":
    main()
