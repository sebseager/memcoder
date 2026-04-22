from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import requests
from config import DEFAULT_CONFIG, OUTPUTS_DIR, config_as_json_dict, ensure_stage_dirs

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
    score += min(float(record.get("python_file_count", 0)), 200.0) * 0.25
    score -= 18.0 * len(record.get("service_dependency_flags", []))
    if not record.get("ci_green", False):
        score -= 120.0
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

            passes_strict = (
                record["passes_python_file_filter"]
                and record["has_test_suite"]
                and record["has_installable_config"]
                and record["ci_green"]
            )
            if args.require_cutoff_pass:
                passes_strict = passes_strict and record["passes_first_commit_filter"]

            passes_relaxed = (
                record["passes_python_file_filter"]
                and record["has_test_suite"]
                and record["has_installable_config"]
                and relaxed_ci_pass(record)
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
