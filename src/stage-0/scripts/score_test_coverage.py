from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import DEFAULT_CONFIG, OUTPUTS_DIR, config_as_json_dict, ensure_stage_dirs

SUMMARY_COUNT_PATTERNS = {
    "passed": re.compile(r"(?P<count>\d+)\s+passed"),
    "failed": re.compile(r"(?P<count>\d+)\s+failed"),
    "errors": re.compile(r"(?P<count>\d+)\s+error"),
    "skipped": re.compile(r"(?P<count>\d+)\s+skipped"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline pytest once per repo and score test coverage quality"
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
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "test_coverage.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of repos to baseline",
    )
    parser.add_argument(
        "--runtime-limit-seconds",
        type=float,
        default=DEFAULT_CONFIG.baseline_runtime_limit_seconds,
    )
    parser.add_argument(
        "--error-limit",
        type=int,
        default=DEFAULT_CONFIG.baseline_error_limit,
    )
    parser.add_argument(
        "--relaxed-runtime-limit-seconds",
        type=float,
        default=DEFAULT_CONFIG.relaxed_baseline_runtime_limit_seconds,
    )
    parser.add_argument(
        "--relaxed-error-limit",
        type=int,
        default=DEFAULT_CONFIG.relaxed_baseline_error_limit,
    )
    parser.add_argument(
        "--disable-relaxed-fallback",
        action="store_true",
        help="Disable relaxed threshold fallback if strict filtering yields too few repos",
    )
    return parser.parse_args()


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
            "had_report": False,
        }

    tree = ET.parse(report_path)
    root = tree.getroot()

    passed: set[str] = set()
    failed: set[str] = set()
    errored: set[str] = set()
    skipped: set[str] = set()

    for test_case in root.iter("testcase"):
        node_id = testcase_node_id(test_case)
        has_failure = any(child.tag.endswith("failure") for child in test_case)
        has_error = any(child.tag.endswith("error") for child in test_case)
        has_skipped = any(child.tag.endswith("skipped") for child in test_case)

        if has_error:
            errored.add(node_id)
        elif has_failure:
            failed.add(node_id)
        elif has_skipped:
            skipped.add(node_id)
        else:
            passed.add(node_id)

    all_tests = sorted(passed | failed | errored | skipped)
    return {
        "passed_tests": sorted(passed),
        "failed_tests": sorted(failed),
        "error_tests": sorted(errored),
        "skipped_tests": sorted(skipped),
        "all_tests": all_tests,
        "had_report": True,
    }


def run_baseline(repo_path: Path, venv_path: Path) -> dict[str, Any]:
    pytest_bin = venv_path / "bin" / "pytest"
    if not pytest_bin.exists():
        return {
            "exit_code": 127,
            "runtime_seconds": 0.0,
            "stdout": "",
            "stderr": f"Missing pytest executable at {pytest_bin}",
            "passed_tests": [],
            "failed_tests": [],
            "error_tests": [],
            "skipped_tests": [],
            "all_tests": [],
            "baseline_pass_count": 0,
            "baseline_fail_count": 0,
            "baseline_error_count": 0,
            "baseline_skipped_count": 0,
            "junit_path": None,
            "used_summary_fallback": True,
        }

    junit_path = repo_path / ".stage0_baseline_junit.xml"
    if junit_path.exists():
        junit_path.unlink()

    command = [
        str(pytest_bin),
        "--tb=no",
        "-q",
        "--timeout=30",
        "--ignore=tests/integration",
        "--ignore=tests/e2e",
        "-x",
        f"--junitxml={junit_path}",
    ]

    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=repo_path,
        check=False,
        capture_output=True,
        text=True,
    )
    runtime_seconds = time.monotonic() - started

    parsed = parse_junit_report(junit_path)
    used_summary_fallback = False

    passed_tests = parsed["passed_tests"]
    failed_tests = parsed["failed_tests"]
    error_tests = parsed["error_tests"]
    skipped_tests = parsed["skipped_tests"]
    all_tests = parsed["all_tests"]

    baseline_pass_count = len(passed_tests)
    baseline_fail_count = len(failed_tests)
    baseline_error_count = len(error_tests)
    baseline_skipped_count = len(skipped_tests)

    if not parsed["had_report"]:
        summary_counts = parse_summary_counts(completed.stdout, completed.stderr)
        baseline_pass_count = summary_counts["passed"]
        baseline_fail_count = summary_counts["failed"]
        baseline_error_count = summary_counts["errors"]
        baseline_skipped_count = summary_counts["skipped"]
        used_summary_fallback = True

    return {
        "exit_code": int(completed.returncode),
        "runtime_seconds": round(runtime_seconds, 3),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "error_tests": error_tests,
        "skipped_tests": skipped_tests,
        "all_tests": all_tests,
        "baseline_pass_count": baseline_pass_count,
        "baseline_fail_count": baseline_fail_count,
        "baseline_error_count": baseline_error_count,
        "baseline_skipped_count": baseline_skipped_count,
        "junit_path": str(junit_path),
        "used_summary_fallback": used_summary_fallback,
    }


def strict_disqualification_reasons(
    record: dict[str, Any], args: argparse.Namespace
) -> list[str]:
    reasons: list[str] = []

    if record["baseline_exit_code"] == 2:
        reasons.append("collection_error_exit_code_2")

    if record["baseline_runtime_seconds"] > args.runtime_limit_seconds:
        reasons.append("runtime_exceeds_limit")

    if record["baseline_error_count"] > args.error_limit:
        reasons.append("too_many_test_errors")

    if record["total_test_count"] == 0:
        reasons.append("no_tests_collected")

    return reasons


def relaxed_disqualification_reasons(
    record: dict[str, Any], args: argparse.Namespace
) -> list[str]:
    reasons: list[str] = []

    if record["baseline_exit_code"] == 2:
        reasons.append("collection_error_exit_code_2")

    if record["baseline_runtime_seconds"] > args.relaxed_runtime_limit_seconds:
        reasons.append("runtime_exceeds_relaxed_limit")

    if record["baseline_error_count"] > args.relaxed_error_limit:
        reasons.append("too_many_test_errors_relaxed")

    if record["total_test_count"] == 0:
        reasons.append("no_tests_collected")

    return reasons


def repo_weight(record: dict[str, Any]) -> float:
    weight = 1.0

    fail_count = int(record.get("baseline_fail_count", 0))
    if fail_count > 0:
        weight *= max(0.5, 1.0 - min(fail_count, 10) * 0.05)

    service_flag_count = len(record.get("service_dependency_flags", []))
    if service_flag_count > 0:
        weight *= max(0.55, 1.0 - service_flag_count * 0.08)

    if record.get("selection_tier") == "relaxed":
        weight *= 0.85

    return round(weight, 4)


def select_repos(
    all_repo_records: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], str]:
    strict_pool = [
        r for r in all_repo_records if not r["strict_disqualification_reasons"]
    ]
    strict_sorted = sorted(
        strict_pool,
        key=lambda r: (
            r.get("repo_weight", 0.0),
            r.get("baseline_pass_count", 0),
            -r.get("baseline_runtime_seconds", 0.0),
        ),
        reverse=True,
    )
    strict_selected = strict_sorted[: DEFAULT_CONFIG.target_repo_count]

    if (
        len(strict_selected) >= DEFAULT_CONFIG.min_repo_count
        or args.disable_relaxed_fallback
    ):
        for item in strict_selected:
            item["selection_tier"] = "strict"
            item["repo_weight"] = repo_weight(item)
        return strict_selected, "strict"

    relaxed_pool = [
        r for r in all_repo_records if not r["relaxed_disqualification_reasons"]
    ]
    relaxed_sorted = sorted(
        relaxed_pool,
        key=lambda r: (
            not r["strict_disqualification_reasons"],
            r.get("baseline_pass_count", 0),
            -r.get("baseline_runtime_seconds", 0.0),
        ),
        reverse=True,
    )
    selected = relaxed_sorted[: DEFAULT_CONFIG.target_repo_count]
    for item in selected:
        item["selection_tier"] = (
            "strict" if not item["strict_disqualification_reasons"] else "relaxed"
        )
        item["repo_weight"] = repo_weight(item)
    return selected, "relaxed"


def main() -> None:
    args = parse_args()
    ensure_stage_dirs()

    candidate_payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidate_map = {
        item["full_name"]: item for item in candidate_payload.get("selected_repos", [])
    }

    env_setup_payload = json.loads(args.env_setup.read_text(encoding="utf-8"))
    env_repos = env_setup_payload.get("qualified_repos", [])
    if args.limit > 0:
        env_repos = env_repos[: args.limit]

    repo_records: list[dict[str, Any]] = []

    for env_meta in env_repos:
        full_name = env_meta["full_name"]
        repo_path = Path(env_meta["repo_path"])
        venv_path = Path(env_meta["venv_path"])

        baseline = run_baseline(repo_path, venv_path)

        candidate_meta = candidate_map.get(full_name, {})
        service_flags = candidate_meta.get("service_dependency_flags", [])

        record = {
            "full_name": full_name,
            "repo_path": str(repo_path),
            "venv_path": str(venv_path),
            "env_install_method": env_meta.get("env_install_method"),
            "used_no_build_isolation": bool(
                env_meta.get("used_no_build_isolation", False)
            ),
            "service_dependency_flags": service_flags,
            "baseline_pass_count": baseline["baseline_pass_count"],
            "baseline_fail_count": baseline["baseline_fail_count"],
            "baseline_error_count": baseline["baseline_error_count"],
            "baseline_skipped_count": baseline["baseline_skipped_count"],
            "baseline_exit_code": baseline["exit_code"],
            "baseline_runtime_seconds": baseline["runtime_seconds"],
            "baseline_failed_tests": baseline["failed_tests"],
            "baseline_error_tests": baseline["error_tests"],
            "baseline_passed_tests": baseline["passed_tests"],
            "baseline_skipped_tests": baseline["skipped_tests"],
            "total_test_count": len(baseline["all_tests"]),
            "junit_path": baseline["junit_path"],
            "used_summary_fallback": baseline["used_summary_fallback"],
            "stdout_tail": baseline["stdout"],
            "stderr_tail": baseline["stderr"],
        }

        record["strict_disqualification_reasons"] = strict_disqualification_reasons(
            record, args
        )
        record["relaxed_disqualification_reasons"] = relaxed_disqualification_reasons(
            record, args
        )
        record["selection_tier"] = "disqualified"
        record["repo_weight"] = repo_weight(record)

        repo_records.append(record)
        print(
            f"Baselined {full_name}: exit={record['baseline_exit_code']} "
            f"pass={record['baseline_pass_count']} fail={record['baseline_fail_count']} "
            f"error={record['baseline_error_count']} runtime={record['baseline_runtime_seconds']}s"
        )

    selected_repos, selection_mode = select_repos(repo_records, args)

    selected_names = {item["full_name"] for item in selected_repos}
    strict_selected_count = sum(
        1 for item in selected_repos if item.get("selection_tier") == "strict"
    )

    qualified_repos = []
    for item in selected_repos:
        qualified_repos.append(
            {
                "full_name": item["full_name"],
                "repo_path": item["repo_path"],
                "venv_path": item["venv_path"],
                "env_install_method": item["env_install_method"],
                "used_no_build_isolation": item["used_no_build_isolation"],
                "baseline_pass_count": item["baseline_pass_count"],
                "baseline_fail_count": item["baseline_fail_count"],
                "baseline_error_count": item["baseline_error_count"],
                "baseline_runtime_seconds": item["baseline_runtime_seconds"],
                "baseline_exit_code": item["baseline_exit_code"],
                "baseline_failed_tests": item["baseline_failed_tests"],
                "baseline_error_tests": item["baseline_error_tests"],
                "baseline_passed_tests": item["baseline_passed_tests"],
                "selection_tier": item["selection_tier"],
                "repo_weight": item["repo_weight"],
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": config_as_json_dict(DEFAULT_CONFIG),
        "thresholds": {
            "runtime_limit_seconds": args.runtime_limit_seconds,
            "error_limit": args.error_limit,
            "relaxed_runtime_limit_seconds": args.relaxed_runtime_limit_seconds,
            "relaxed_error_limit": args.relaxed_error_limit,
        },
        "stats": {
            "input_repo_count": len(env_repos),
            "selected_repo_count": len(selected_repos),
            "strict_selected_count": strict_selected_count,
            "selection_mode": selection_mode,
            "fallback_used": selection_mode == "relaxed",
            "meets_min_repo_target": len(selected_repos)
            >= DEFAULT_CONFIG.min_repo_count,
            "strict_pass_pool_count": sum(
                1
                for item in repo_records
                if not item["strict_disqualification_reasons"]
            ),
            "relaxed_pass_pool_count": sum(
                1
                for item in repo_records
                if not item["relaxed_disqualification_reasons"]
            ),
        },
        "qualified_repos": qualified_repos,
        "selected_repos": selected_repos,
        "repos": repo_records,
        "disqualified_repos": [
            item for item in repo_records if item["full_name"] not in selected_names
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Wrote test coverage scoring to {args.output} "
        f"(selected={len(selected_repos)} mode={selection_mode})"
    )


if __name__ == "__main__":
    main()
