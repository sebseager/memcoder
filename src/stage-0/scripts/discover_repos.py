from __future__ import annotations

import argparse
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
    qualified: list[dict[str, Any]] = []

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
                python_files = count_python_files(session, owner, name, default_branch)
                first_commit = None
                if python_files >= cfg.min_python_files:
                    first_commit = oldest_commit_date(
                        session, owner, name, default_branch
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

            raw_candidates.append(record)

            passes = record["passes_python_file_filter"]
            if args.require_cutoff_pass:
                passes = passes and record["passes_first_commit_filter"]
            if passes:
                qualified.append(record)
                if len(qualified) >= cfg.target_repo_count:
                    break

            time.sleep(args.sleep_seconds)

        if len(qualified) >= cfg.target_repo_count:
            break

    qualified_sorted = sorted(
        qualified,
        key=lambda r: (r["stars"], r["python_file_count"]),
        reverse=True,
    )
    selected = qualified_sorted[: cfg.target_repo_count]

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "query": query,
        "config": config_as_json_dict(cfg),
        "stats": {
            "raw_candidate_count": len(raw_candidates),
            "qualified_count": len(qualified_sorted),
            "selected_count": len(selected),
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
    print(f"Wrote {len(result['selected_repos'])} selected repos to {args.output}")


if __name__ == "__main__":
    main()
