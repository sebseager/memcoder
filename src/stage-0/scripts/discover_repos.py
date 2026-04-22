from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests
from config import (
    DEFAULT_CONFIG,
    OUTPUTS_DIR,
    REPOS_DIR,
    config_as_json_dict,
    ensure_stage_dirs,
)

GITHUB_API = "https://api.github.com"
LINK_RE = re.compile(r"<([^>]+)>; rel=\"([^\"]+)\"")
TEST_FILE_RE = re.compile(r"(^|/)test_[^/]+\.py$")

INSTALLABLE_MARKERS = {"pyproject.toml", "setup.py"}
SERVICE_DEPENDENCY_PACKAGES = (
    "boto3",
    "redis",
    "psycopg2",
    "pymongo",
    "stripe",
    "twilio",
    "sendgrid",
)
CI_RELAXED_STATES = {"pending", "no_status", "unknown"}
CHECK_RUN_SUCCESS_CONCLUSIONS = {"success", "neutral", "skipped"}
CHECK_RUN_FAILURE_CONCLUSIONS = {
    "failure",
    "timed_out",
    "cancelled",
    "action_required",
    "startup_failure",
}
SKIP_PATH_PARTS = {".git", ".venv", "venv", "node_modules", "__pycache__"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover Stage 0 repo candidates")
    parser.add_argument("--max-search-pages", type=int, default=5)
    parser.add_argument("--per-page", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "repo_candidates.json",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Delay between GitHub API calls to stay below rate limits",
    )
    parser.add_argument(
        "--require-cutoff-pass",
        action="store_true",
        help="Keep only repos where first commit is after the cutoff",
    )
    parser.add_argument(
        "--disable-relaxed-fallback",
        action="store_true",
        help="Disable relaxed CI fallback when strict filtering yields too few repos",
    )
    parser.add_argument(
        "--max-dependency-files",
        type=int,
        default=8,
        help="Maximum dependency files to inspect per repo for service dependency flags",
    )
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=OUTPUTS_DIR / "test_coverage.json",
        help="Optional prior test coverage artifact used to penalize historically slow repos",
    )
    parser.add_argument(
        "--slow-runtime-seconds",
        type=float,
        default=30.0,
        help="Baseline runtime above which repos receive a heavy discovery penalty",
    )
    parser.add_argument(
        "--hard-exclude-runtime-seconds",
        type=float,
        default=55.0,
        help="Baseline runtime above which repos are excluded when runtime signal is reliable",
    )
    parser.add_argument(
        "--disable-local-fallback",
        action="store_true",
        help="Disable fallback candidate discovery from locally cloned repos",
    )
    parser.add_argument(
        "--local-repos-dir",
        type=Path,
        default=REPOS_DIR,
        help="Directory scanned for local fallback candidates",
    )
    return parser.parse_args()


def github_session() -> requests.Session:
    session = requests.Session()
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    session.headers["Accept"] = "application/vnd.github+json"
    session.headers["X-GitHub-Api-Version"] = "2022-11-28"
    return session


def request_json(
    session: requests.Session, url: str, *, params: dict[str, Any] | None = None
) -> tuple[Any, requests.Response]:
    response = session.get(url, params=params, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(
            f"GitHub API error {response.status_code} for {url}: {response.text[:300]}"
        )
    return response.json(), response


def parse_link_header(link_header: str | None) -> dict[str, str]:
    if not link_header:
        return {}
    links: dict[str, str] = {}
    for chunk in link_header.split(","):
        match = LINK_RE.search(chunk.strip())
        if match:
            links[match.group(2)] = match.group(1)
    return links


def fetch_repo_tree(
    session: requests.Session, owner: str, repo: str, ref: str
) -> list[dict[str, Any]]:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{ref}"
    payload, _ = request_json(session, url, params={"recursive": "1"})
    return payload.get("tree", [])


def analyze_tree(entries: list[dict[str, Any]]) -> dict[str, Any]:
    python_file_count = 0
    has_test_suite = False
    has_installable_marker = False
    dependency_files: list[dict[str, Any]] = []

    for entry in entries:
        if entry.get("type") != "blob":
            continue

        path = entry.get("path", "")
        path_lower = path.lower()
        basename = path_lower.rsplit("/", 1)[-1]

        if path_lower.endswith(".py"):
            python_file_count += 1
        if path_lower.startswith("tests/") or "/tests/" in path_lower:
            has_test_suite = True
        elif TEST_FILE_RE.search(path_lower):
            has_test_suite = True

        if path_lower in INSTALLABLE_MARKERS:
            has_installable_marker = True

        if path_lower == "pyproject.toml":
            dependency_files.append(entry)
        elif basename == "setup.py":
            dependency_files.append(entry)
        elif basename.startswith("requirements") and (
            basename.endswith(".txt") or basename.endswith(".in")
        ):
            dependency_files.append(entry)

    return {
        "python_file_count": python_file_count,
        "has_test_suite": has_test_suite,
        "has_installable_marker": has_installable_marker,
        "dependency_files": dependency_files,
    }


def decode_blob_content(blob_payload: dict[str, Any]) -> str:
    content = blob_payload.get("content", "")
    encoding = blob_payload.get("encoding", "")
    if encoding == "base64":
        try:
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return ""
    return str(content)


def scan_service_dependencies(text: str) -> list[str]:
    lowered = text.lower()
    flagged: set[str] = set()

    for package in SERVICE_DEPENDENCY_PACKAGES:
        if re.search(rf"(^|[^a-z0-9_-]){re.escape(package)}([^a-z0-9_-]|$)", lowered):
            flagged.add(package)

    if "celery" in lowered:
        has_broker_signal = any(
            token in lowered for token in ("redis", "amqp", "rabbitmq", "broker", "sqs")
        )
        if has_broker_signal:
            flagged.add("celery")

    return sorted(flagged)


def dependency_flags_from_tree(
    session: requests.Session,
    owner: str,
    repo: str,
    dependency_files: list[dict[str, Any]],
    *,
    max_files: int,
) -> list[str]:
    if not dependency_files:
        return []

    def sort_key(entry: dict[str, Any]) -> tuple[int, str]:
        path = entry.get("path", "")
        if path == "pyproject.toml":
            return (0, path)
        if path.endswith("requirements.txt"):
            return (1, path)
        if path.endswith("requirements-dev.txt"):
            return (2, path)
        return (3, path)

    selected_files = sorted(dependency_files, key=sort_key)[:max_files]
    flags: set[str] = set()
    for dep_file in selected_files:
        sha = dep_file.get("sha")
        if not sha:
            continue
        blob_url = f"{GITHUB_API}/repos/{owner}/{repo}/git/blobs/{sha}"
        try:
            blob_payload, _ = request_json(session, blob_url)
        except Exception:  # noqa: BLE001
            continue

        text = decode_blob_content(blob_payload)
        flags.update(scan_service_dependencies(text))

    return sorted(flags)


def load_runtime_profile(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}

    profile: dict[str, dict[str, Any]] = {}
    for item in payload.get("repos", []):
        full_name = item.get("full_name")
        if not full_name:
            continue

        total_tests = int(
            item.get(
                "total_test_count",
                len(item.get("baseline_passed_tests", []))
                + len(item.get("baseline_failed_tests", []))
                + len(item.get("baseline_error_tests", [])),
            )
        )
        profile[full_name] = {
            "runtime_seconds": float(item.get("baseline_runtime_seconds", 0.0)),
            "exit_code": int(item.get("baseline_exit_code", -1)),
            "total_tests": total_tests,
        }

    return profile


def apply_runtime_signal(
    record: dict[str, Any],
    runtime_profile: dict[str, dict[str, Any]],
    *,
    slow_runtime_seconds: float,
    hard_exclude_runtime_seconds: float,
) -> None:
    runtime_info = runtime_profile.get(record["full_name"])

    record["slow_runtime_seconds"] = slow_runtime_seconds
    record["hard_exclude_runtime_seconds"] = hard_exclude_runtime_seconds
    record["historical_runtime_known"] = False
    record["historical_baseline_runtime_seconds"] = None
    record["historical_baseline_total_tests"] = 0
    record["historical_baseline_exit_code"] = None
    record["historical_runtime_slow"] = False
    record["historical_runtime_hard_excluded"] = False

    if not runtime_info:
        return

    runtime_seconds = float(runtime_info.get("runtime_seconds", 0.0))
    total_tests = int(runtime_info.get("total_tests", 0))
    exit_code = int(runtime_info.get("exit_code", -1))

    # Missing-dependency and similar bootstrap issues often have 0 collected tests.
    # Treat those as non-actionable so they do not incur heavy-repo penalties.
    actionable_runtime = total_tests > 0 and runtime_seconds > 0.0

    record["historical_runtime_known"] = actionable_runtime
    record["historical_baseline_runtime_seconds"] = runtime_seconds
    record["historical_baseline_total_tests"] = total_tests
    record["historical_baseline_exit_code"] = exit_code
    record["historical_runtime_slow"] = actionable_runtime and (
        runtime_seconds >= slow_runtime_seconds
    )
    record["historical_runtime_hard_excluded"] = actionable_runtime and (
        runtime_seconds >= hard_exclude_runtime_seconds
    )


def dependency_flags_from_local_files(
    files: list[Path],
    *,
    repo_root: Path,
    max_files: int,
) -> list[str]:
    if not files:
        return []

    def sort_key(path: Path) -> tuple[int, str]:
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        if rel == "pyproject.toml":
            return (0, rel)
        if rel.endswith("requirements.txt"):
            return (1, rel)
        if rel.endswith("requirements-dev.txt"):
            return (2, rel)
        return (3, rel)

    selected = sorted(files, key=sort_key)[:max_files]
    flags: set[str] = set()
    for dep_file in selected:
        try:
            text = dep_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            continue
        flags.update(scan_service_dependencies(text))
    return sorted(flags)


def git_capture(repo_root: Path, args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def parse_remote_full_name(remote_url: str | None, local_name: str) -> str:
    if remote_url:
        candidate = remote_url.strip()
        if candidate.endswith(".git"):
            candidate = candidate[:-4]
        match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/]+)$", candidate)
        if match:
            return f"{match.group('owner')}/{match.group('repo')}"

    return local_name.replace("__", "/")


def build_local_candidate_record(
    repo_root: Path,
    *,
    max_dependency_files: int,
    runtime_profile: dict[str, dict[str, Any]],
    slow_runtime_seconds: float,
    hard_exclude_runtime_seconds: float,
    require_cutoff_pass: bool,
    cfg: Any,
) -> dict[str, Any] | None:
    if not (repo_root / ".git").exists():
        return None

    local_name = repo_root.name
    remote_url = git_capture(repo_root, ["config", "--get", "remote.origin.url"])
    full_name = parse_remote_full_name(remote_url, local_name)
    default_branch = git_capture(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]) or "main"
    head_sha = git_capture(repo_root, ["rev-parse", "HEAD"])
    pushed_iso = git_capture(repo_root, ["show", "-s", "--format=%cI", "HEAD"])
    first_commit_sha = git_capture(repo_root, ["rev-list", "--max-parents=0", "HEAD"])
    first_commit_iso = (
        git_capture(repo_root, ["show", "-s", "--format=%cI", first_commit_sha])
        if first_commit_sha
        else None
    )

    python_file_count = 0
    has_test_suite = False
    has_installable_marker = False
    dependency_files: list[Path] = []

    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PATH_PARTS for part in path.parts):
            continue

        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        rel_lower = rel.lower()
        basename = path.name.lower()

        if rel_lower.endswith(".py"):
            python_file_count += 1

        if rel_lower.startswith("tests/") or "/tests/" in rel_lower:
            has_test_suite = True
        elif TEST_FILE_RE.search(rel_lower):
            has_test_suite = True

        if rel_lower in INSTALLABLE_MARKERS:
            has_installable_marker = True

        if rel_lower == "pyproject.toml":
            dependency_files.append(path)
        elif basename == "setup.py":
            dependency_files.append(path)
        elif basename.startswith("requirements") and (
            basename.endswith(".txt") or basename.endswith(".in")
        ):
            dependency_files.append(path)

    first_commit_date: date | None = None
    if first_commit_iso:
        try:
            first_commit_date = datetime.fromisoformat(
                first_commit_iso.replace("Z", "+00:00")
            ).date()
        except Exception:  # noqa: BLE001
            first_commit_date = None

    pushed_date_iso = None
    if pushed_iso:
        try:
            pushed_date_iso = datetime.fromisoformat(pushed_iso.replace("Z", "+00:00")).date().isoformat()
        except Exception:  # noqa: BLE001
            pushed_date_iso = None

    dependency_flags = dependency_flags_from_local_files(
        dependency_files,
        repo_root=repo_root,
        max_files=max_dependency_files,
    )

    record: dict[str, Any] = {
        "full_name": full_name,
        "html_url": f"https://github.com/{full_name}",
        "clone_url": f"https://github.com/{full_name}.git",
        "stars": 0,
        "default_branch": default_branch,
        "pushed_at": pushed_date_iso,
        "created_at": first_commit_iso,
        "python_file_count": python_file_count,
        "first_commit_date": first_commit_date.isoformat() if first_commit_date else None,
        "passes_python_file_filter": python_file_count >= cfg.min_python_files,
        "passes_first_commit_filter": bool(
            first_commit_date and first_commit_date > cfg.first_commit_cutoff
        ),
        "has_test_suite": has_test_suite,
        "has_installable_config": has_installable_marker,
        "ci_status_state": "unknown",
        "ci_green": False,
        "ci_status_context_count": 0,
        "ci_check_run_count": 0,
        "head_sha": head_sha,
        "service_dependency_flags": dependency_flags,
        "service_dependency_flagged": len(dependency_flags) > 0,
        "candidate_source": "local-fallback",
    }

    apply_runtime_signal(
        record,
        runtime_profile,
        slow_runtime_seconds=slow_runtime_seconds,
        hard_exclude_runtime_seconds=hard_exclude_runtime_seconds,
    )

    passes_strict = (
        record["passes_python_file_filter"]
        and record["has_test_suite"]
        and record["has_installable_config"]
        and record["ci_green"]
        and not record.get("historical_runtime_hard_excluded", False)
    )
    if require_cutoff_pass:
        passes_strict = passes_strict and record["passes_first_commit_filter"]

    passes_relaxed = (
        record["passes_python_file_filter"]
        and record["has_test_suite"]
        and record["has_installable_config"]
        and relaxed_ci_pass(record)
        and not record.get("historical_runtime_hard_excluded", False)
    )
    if require_cutoff_pass:
        passes_relaxed = passes_relaxed and record["passes_first_commit_filter"]

    record["passes_strict_filters"] = passes_strict
    record["passes_relaxed_filters"] = passes_relaxed
    record["quality_score"] = quality_score(record)
    return record


def discover_local_fallback_candidates(
    local_repos_dir: Path,
    *,
    max_dependency_files: int,
    runtime_profile: dict[str, dict[str, Any]],
    slow_runtime_seconds: float,
    hard_exclude_runtime_seconds: float,
    require_cutoff_pass: bool,
    cfg: Any,
) -> list[dict[str, Any]]:
    if not local_repos_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for repo_root in sorted(local_repos_dir.iterdir()):
        if not repo_root.is_dir():
            continue
        record = build_local_candidate_record(
            repo_root,
            max_dependency_files=max_dependency_files,
            runtime_profile=runtime_profile,
            slow_runtime_seconds=slow_runtime_seconds,
            hard_exclude_runtime_seconds=hard_exclude_runtime_seconds,
            require_cutoff_pass=require_cutoff_pass,
            cfg=cfg,
        )
        if record is not None:
            records.append(record)
    return records


def count_python_files(
    session: requests.Session, owner: str, repo: str, ref: str
) -> int:
    url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{ref}"
    payload, _ = request_json(session, url, params={"recursive": "1"})
    entries = payload.get("tree", [])
    return sum(
        1
        for entry in entries
        if entry.get("path", "").endswith(".py") and entry.get("type") == "blob"
    )


def latest_commit_sha(
    session: requests.Session, owner: str, repo: str, branch: str
) -> str | None:
    commits_url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    payload, _ = request_json(
        session,
        commits_url,
        params={"sha": branch, "per_page": 1, "page": 1},
    )
    if not payload:
        return None
    return payload[0].get("sha")


def ci_status_on_head(
    session: requests.Session,
    owner: str,
    repo: str,
    head_sha: str,
) -> dict[str, Any]:
    status_state = "unknown"
    status_count = 0
    check_run_count = 0
    has_failed_checks = False
    has_successful_checks = False
    has_pending_checks = False

    status_url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{head_sha}/status"
    try:
        status_payload, _ = request_json(session, status_url)
        status_state = status_payload.get("state", "unknown")
        status_count = int(status_payload.get("total_count", 0))
    except Exception:  # noqa: BLE001
        status_state = "unknown"

    checks_url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{head_sha}/check-runs"
    try:
        checks_payload, _ = request_json(session, checks_url)
        check_runs = checks_payload.get("check_runs", [])
        check_run_count = len(check_runs)
        for run in check_runs:
            status = run.get("status", "")
            conclusion = run.get("conclusion")
            if status != "completed":
                has_pending_checks = True
                continue
            if conclusion in CHECK_RUN_FAILURE_CONCLUSIONS:
                has_failed_checks = True
            if conclusion in CHECK_RUN_SUCCESS_CONCLUSIONS:
                has_successful_checks = True
    except Exception:  # noqa: BLE001
        pass

    if status_state == "success" or (
        check_run_count > 0 and has_successful_checks and not has_failed_checks
    ):
        ci_status_state = "success"
        ci_green = True
    elif status_state in {"failure", "error"} or has_failed_checks:
        ci_status_state = "failure"
        ci_green = False
    elif status_state == "pending" or has_pending_checks:
        ci_status_state = "pending"
        ci_green = False
    elif status_count == 0 and check_run_count == 0:
        ci_status_state = "no_status"
        ci_green = False
    else:
        ci_status_state = status_state or "unknown"
        ci_green = False

    return {
        "ci_status_state": ci_status_state,
        "ci_green": ci_green,
        "status_context_count": status_count,
        "check_run_count": check_run_count,
    }


def quality_score(record: dict[str, Any]) -> float:
    score = float(record.get("stars", 0))
    score += min(float(record.get("python_file_count", 0)), 120.0) * 0.08
    score -= 18.0 * len(record.get("service_dependency_flags", []))
    if not record.get("ci_green", False):
        score -= 120.0
    if record.get("historical_runtime_slow", False):
        runtime_seconds = float(record.get("historical_baseline_runtime_seconds", 0.0))
        slow_runtime_seconds = float(record.get("slow_runtime_seconds", 30.0))
        # Strongly penalize known slow repos so they sink in discovery ranking.
        score -= 320.0 + max(0.0, runtime_seconds - slow_runtime_seconds) * 12.0
    if record.get("historical_runtime_hard_excluded", False):
        score -= 10000.0
    return score


def relaxed_ci_pass(record: dict[str, Any]) -> bool:
    if record.get("ci_green", False):
        return True
    return record.get("ci_status_state") in CI_RELAXED_STATES


def oldest_commit_date(
    session: requests.Session, owner: str, repo: str, branch: str
) -> date | None:
    commits_url = f"{GITHUB_API}/repos/{owner}/{repo}/commits"
    _, response = request_json(
        session,
        commits_url,
        params={"sha": branch, "per_page": 1, "page": 1},
    )
    links = parse_link_header(response.headers.get("Link"))
    if "last" in links:
        last_page_url = links["last"]
        oldest_payload, _ = request_json(session, last_page_url)
    else:
        oldest_payload, _ = request_json(
            session,
            commits_url,
            params={"sha": branch, "per_page": 1, "page": 1},
        )
    if not oldest_payload:
        return None
    commit_date = oldest_payload[0]["commit"]["author"]["date"]
    return datetime.fromisoformat(commit_date.replace("Z", "+00:00")).date()


def discover(args: argparse.Namespace) -> dict[str, Any]:
    ensure_stage_dirs()
    cfg = DEFAULT_CONFIG
    session = github_session()
    runtime_profile = load_runtime_profile(args.runtime_profile)

    query = (
        f"language:Python stars:{cfg.min_stars}..{cfg.max_stars} "
        f"created:>={cfg.first_commit_cutoff.isoformat()} "
        f"pushed:>={cfg.pushed_since.isoformat()} fork:false archived:false"
    )

    raw_candidates: list[dict[str, Any]] = []
    strict_qualified: list[dict[str, Any]] = []
    relaxed_qualified: list[dict[str, Any]] = []
    seen_full_names: set[str] = set()

    for page in range(1, args.max_search_pages + 1):
        search_url = f"{GITHUB_API}/search/repositories"
        search_payload, _ = request_json(
            session,
            search_url,
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": args.per_page,
                "page": page,
            },
        )
        items = search_payload.get("items", [])
        if not items:
            break

        for repo in items:
            owner = repo["owner"]["login"]
            name = repo["name"]
            full_name = repo["full_name"]
            if full_name in seen_full_names:
                continue
            seen_full_names.add(full_name)

            default_branch = repo["default_branch"]
            pushed_at = datetime.fromisoformat(
                repo["pushed_at"].replace("Z", "+00:00")
            ).date()
            record: dict[str, Any] = {
                "full_name": full_name,
                "html_url": repo["html_url"],
                "clone_url": repo["clone_url"],
                "stars": repo["stargazers_count"],
                "default_branch": default_branch,
                "pushed_at": pushed_at.isoformat(),
                "created_at": repo["created_at"],
                "candidate_source": "github-search",
            }

            try:
                tree_entries = fetch_repo_tree(session, owner, name, default_branch)
                tree_info = analyze_tree(tree_entries)
                python_files = int(tree_info["python_file_count"])
                first_commit = None
                if python_files >= cfg.min_python_files:
                    first_commit = oldest_commit_date(
                        session, owner, name, default_branch
                    )
                head_sha = latest_commit_sha(session, owner, name, default_branch)
                ci_info = (
                    ci_status_on_head(session, owner, name, head_sha)
                    if head_sha
                    else {
                        "ci_status_state": "unknown",
                        "ci_green": False,
                        "status_context_count": 0,
                        "check_run_count": 0,
                    }
                )
                dependency_flags = dependency_flags_from_tree(
                    session,
                    owner,
                    name,
                    tree_info["dependency_files"],
                    max_files=args.max_dependency_files,
                )
            except Exception as exc:  # noqa: BLE001
                record["error"] = str(exc)
                raw_candidates.append(record)
                time.sleep(args.sleep_seconds)
                continue

            record["python_file_count"] = python_files
            record["first_commit_date"] = (
                first_commit.isoformat() if first_commit else None
            )
            record["passes_python_file_filter"] = python_files >= cfg.min_python_files
            record["passes_first_commit_filter"] = bool(
                first_commit and first_commit > cfg.first_commit_cutoff
            )
            record["has_test_suite"] = bool(tree_info["has_test_suite"])
            record["has_installable_config"] = bool(tree_info["has_installable_marker"])
            record["ci_status_state"] = ci_info["ci_status_state"]
            record["ci_green"] = bool(ci_info["ci_green"])
            record["ci_status_context_count"] = int(ci_info["status_context_count"])
            record["ci_check_run_count"] = int(ci_info["check_run_count"])
            record["head_sha"] = head_sha
            record["service_dependency_flags"] = dependency_flags
            record["service_dependency_flagged"] = len(dependency_flags) > 0

            apply_runtime_signal(
                record,
                runtime_profile,
                slow_runtime_seconds=args.slow_runtime_seconds,
                hard_exclude_runtime_seconds=args.hard_exclude_runtime_seconds,
            )

            passes_strict = (
                record["passes_python_file_filter"]
                and record["has_test_suite"]
                and record["has_installable_config"]
                and record["ci_green"]
                and not record.get("historical_runtime_hard_excluded", False)
            )
            if args.require_cutoff_pass:
                passes_strict = passes_strict and record["passes_first_commit_filter"]

            passes_relaxed = (
                record["passes_python_file_filter"]
                and record["has_test_suite"]
                and record["has_installable_config"]
                and relaxed_ci_pass(record)
                and not record.get("historical_runtime_hard_excluded", False)
            )
            if args.require_cutoff_pass:
                passes_relaxed = passes_relaxed and record["passes_first_commit_filter"]

            record["passes_strict_filters"] = passes_strict
            record["passes_relaxed_filters"] = passes_relaxed
            record["quality_score"] = quality_score(record)

            raw_candidates.append(record)

            if passes_strict:
                strict_qualified.append(record)

            if passes_relaxed:
                relaxed_qualified.append(record)

            if len(strict_qualified) >= cfg.target_repo_count:
                break

            time.sleep(args.sleep_seconds)

        if len(strict_qualified) >= cfg.target_repo_count:
            break

    local_fallback_candidates: list[dict[str, Any]] = []
    if (
        len(strict_qualified) < cfg.min_repo_count
        and not args.disable_local_fallback
    ):
        local_fallback_candidates = discover_local_fallback_candidates(
            args.local_repos_dir,
            max_dependency_files=args.max_dependency_files,
            runtime_profile=runtime_profile,
            slow_runtime_seconds=args.slow_runtime_seconds,
            hard_exclude_runtime_seconds=args.hard_exclude_runtime_seconds,
            require_cutoff_pass=args.require_cutoff_pass,
            cfg=cfg,
        )

        for record in local_fallback_candidates:
            full_name = record["full_name"]
            if full_name in seen_full_names:
                continue
            seen_full_names.add(full_name)
            raw_candidates.append(record)
            if record.get("passes_strict_filters", False):
                strict_qualified.append(record)
            if record.get("passes_relaxed_filters", False):
                relaxed_qualified.append(record)

    strict_sorted = sorted(
        strict_qualified,
        key=lambda r: (r.get("quality_score", 0.0), r["stars"], r["python_file_count"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = strict_sorted[: cfg.target_repo_count]
    selection_mode = "strict"

    if (
        len(selected) < cfg.min_repo_count
        and not args.disable_relaxed_fallback
        and relaxed_qualified
    ):
        relaxed_sorted = sorted(
            relaxed_qualified,
            key=lambda r: (
                r.get("passes_strict_filters", False),
                r.get("quality_score", 0.0),
                r["stars"],
                r["python_file_count"],
            ),
            reverse=True,
        )
        selected = relaxed_sorted[: cfg.target_repo_count]
        selection_mode = "relaxed"

    for record in selected:
        record["selection_tier"] = (
            "strict" if record.get("passes_strict_filters", False) else "relaxed"
        )

    strict_selected_count = sum(
        1 for record in selected if record.get("selection_tier") == "strict"
    )
    risky_selected_count = sum(
        1 for record in selected if record.get("service_dependency_flagged", False)
    )
    slow_runtime_selected_count = sum(
        1 for record in selected if record.get("historical_runtime_slow", False)
    )
    runtime_hard_excluded_count = sum(
        1
        for record in raw_candidates
        if record.get("historical_runtime_hard_excluded", False)
    )

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "query": query,
        "config": config_as_json_dict(cfg),
        "stats": {
            "raw_candidate_count": len(raw_candidates),
            "strict_qualified_count": len(strict_qualified),
            "relaxed_qualified_count": len(relaxed_qualified),
            "selected_count": len(selected),
            "strict_selected_count": strict_selected_count,
            "selected_with_service_dependency_flags": risky_selected_count,
            "selected_with_historical_slow_runtime": slow_runtime_selected_count,
            "runtime_hard_excluded_candidate_count": runtime_hard_excluded_count,
            "runtime_profile_entries": len(runtime_profile),
            "local_fallback_candidate_count": len(local_fallback_candidates),
            "selection_mode": selection_mode,
            "fallback_used": selection_mode == "relaxed",
            "selected_meets_min_repo_target": len(selected) >= cfg.min_repo_count,
        },
        "selected_repos": selected,
        "raw_candidates": raw_candidates,
    }
    return result


def main() -> None:
    args = parse_args()
    result = discover(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "Wrote "
        f"{len(result['selected_repos'])} selected repos to {args.output} "
        f"(mode={result['stats']['selection_mode']})"
    )


if __name__ == "__main__":
    main()
