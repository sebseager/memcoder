from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import (
    DEFAULT_CONFIG,
    OUTPUTS_DIR,
    REPOS_DIR,
    config_as_json_dict,
    ensure_stage_dirs,
)

INSTALL_FALLBACK_REQUIREMENTS = (
    "requirements-dev.txt",
    "requirements_test.txt",
    "requirements.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create isolated per-repo uv environments and install dependencies"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=OUTPUTS_DIR / "repo_candidates.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "env_setup.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of repos to process",
    )
    parser.add_argument(
        "--python-version",
        default=DEFAULT_CONFIG.env_python_version,
        help="Python version used for per-repo venv creation",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    return {
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "returncode": completed.returncode,
        "runtime_seconds": round(elapsed, 3),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def command_succeeded(result: dict[str, Any]) -> bool:
    return int(result.get("returncode", 1)) == 0


def try_install_with_optional_no_build_isolation(
    base_command: list[str], cwd: Path
) -> tuple[bool, bool, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []

    first = run_command(base_command, cwd=cwd)
    attempts.append(first)
    if command_succeeded(first):
        return True, False, attempts

    fallback_command = [*base_command, "--no-build-isolation"]
    second = run_command(fallback_command, cwd=cwd)
    attempts.append(second)
    if command_succeeded(second):
        return True, True, attempts

    return False, False, attempts


def repo_record_template(repo_meta: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    return {
        "full_name": repo_meta["full_name"],
        "clone_url": repo_meta.get("clone_url"),
        "local_name": repo_meta["full_name"].replace("/", "__"),
        "repo_path": str(repo_root),
        "env_python": DEFAULT_CONFIG.env_python_version,
        "env_install_method": None,
        "requirements_file": None,
        "used_no_build_isolation": False,
        "install_success": False,
        "disqualified": True,
        "disqualify_reason": "not_processed",
        "attempts": [],
    }


def main() -> None:
    args = parse_args()
    ensure_stage_dirs()

    if shutil.which("uv") is None:
        raise RuntimeError("`uv` is required but was not found on PATH")

    candidates_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    selected_repos = candidates_payload.get("selected_repos", [])
    if args.limit > 0:
        selected_repos = selected_repos[: args.limit]

    repo_results: list[dict[str, Any]] = []
    qualified_repos: list[dict[str, Any]] = []

    for repo_meta in selected_repos:
        full_name = repo_meta["full_name"]
        local_name = full_name.replace("/", "__")
        repo_root = REPOS_DIR / local_name
        venv_path = repo_root / ".venv"

        result = repo_record_template(repo_meta, repo_root)
        result["env_python"] = args.python_version
        result["venv_path"] = str(venv_path)

        if not repo_root.exists():
            result["disqualify_reason"] = "repo_not_cloned"
            repo_results.append(result)
            print(f"Disqualified {full_name}: repo not found at {repo_root}")
            continue

        if venv_path.exists():
            result["attempts"].append(
                {
                    "command": ["uv", "venv", str(venv_path), "--python", args.python_version],
                    "cwd": str(repo_root),
                    "returncode": 0,
                    "runtime_seconds": 0.0,
                    "stdout": "",
                    "stderr": "Reused existing virtual environment",
                }
            )
        else:
            create_venv = run_command(
                ["uv", "venv", str(venv_path), "--python", args.python_version],
                cwd=repo_root,
            )
            result["attempts"].append(create_venv)
            if not command_succeeded(create_venv):
                result["disqualify_reason"] = "venv_creation_failed"
                repo_results.append(result)
                print(f"Disqualified {full_name}: failed to create venv")
                continue

        editable_ok, editable_nobi, editable_attempts = (
            try_install_with_optional_no_build_isolation(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_path),
                    "-e",
                    ".[dev,test]",
                ],
                cwd=repo_root,
            )
        )
        result["attempts"].extend(editable_attempts)

        install_ok = False
        install_method = None
        requirements_file = None
        used_no_build_isolation = editable_nobi

        if editable_ok:
            install_ok = True
            install_method = "editable"
        else:
            for req_name in INSTALL_FALLBACK_REQUIREMENTS:
                req_path = repo_root / req_name
                if not req_path.exists():
                    continue
                req_ok, req_nobi, req_attempts = (
                    try_install_with_optional_no_build_isolation(
                        [
                            "uv",
                            "pip",
                            "install",
                            "--python",
                            str(venv_path),
                            "-r",
                            req_name,
                        ],
                        cwd=repo_root,
                    )
                )
                result["attempts"].extend(req_attempts)
                if req_ok:
                    install_ok = True
                    install_method = "requirements"
                    requirements_file = req_name
                    used_no_build_isolation = used_no_build_isolation or req_nobi
                    break

        if install_ok:
            tooling = run_command(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(venv_path),
                    "pytest",
                    "pytest-timeout",
                ],
                cwd=repo_root,
            )
            result["attempts"].append(tooling)
            if not command_succeeded(tooling):
                install_ok = False
                result["disqualify_reason"] = "pytest_tooling_install_failed"

        result["install_success"] = install_ok
        result["env_install_method"] = install_method
        result["requirements_file"] = requirements_file
        result["used_no_build_isolation"] = used_no_build_isolation

        if install_ok:
            result["disqualified"] = False
            result["disqualify_reason"] = None
            qualified_repos.append(
                {
                    "full_name": full_name,
                    "local_name": local_name,
                    "repo_path": str(repo_root),
                    "venv_path": str(venv_path),
                    "env_install_method": install_method,
                    "used_no_build_isolation": used_no_build_isolation,
                    "requirements_file": requirements_file,
                }
            )
            print(f"Qualified {full_name}: install_method={install_method}")
        else:
            if result["disqualify_reason"] == "not_processed":
                result["disqualify_reason"] = "dependency_install_failed"
            print(f"Disqualified {full_name}: {result['disqualify_reason']}")

        repo_results.append(result)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config_as_json_dict(DEFAULT_CONFIG),
        "stats": {
            "input_repo_count": len(selected_repos),
            "qualified_repo_count": len(qualified_repos),
            "disqualified_repo_count": len(selected_repos) - len(qualified_repos),
            "meets_min_repo_target": len(qualified_repos)
            >= DEFAULT_CONFIG.min_repo_count,
        },
        "qualified_repos": qualified_repos,
        "repos": repo_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote env setup results to {args.output}")


if __name__ == "__main__":
    main()
