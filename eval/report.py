"""Aggregate ``judgments.jsonl`` into a human-readable ``report.md``."""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("memcoder.eval.report")

FAILURE_MODES = (
    "wrong_specifics",
    "missing_information",
    "off_topic",
    "refusal_or_nonresponse",
    "format_failure",
    "other",
)


def write_report(run_dir: Path) -> Path:
    judgments_path = run_dir / "judgments.jsonl"
    if not judgments_path.exists():
        raise FileNotFoundError(f"judgments.jsonl not found at {judgments_path}")

    rows = [
        json.loads(line)
        for line in judgments_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows in {judgments_path}")

    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repo[str(row.get("repo_id") or "<unknown>")].append(row)

    lines: list[str] = []
    lines.append(f"# Eval Report — {run_dir.name}")
    lines.append("")
    lines.append(f"- Total rows: **{len(rows)}**")
    lines.append(f"- Repos: **{len(by_repo)}**")
    lines.append("")

    for repo_id in sorted(by_repo):
        repo_rows = by_repo[repo_id]
        lines.append(f"## Repo: `{repo_id}`")
        lines.append("")
        lines.append(f"Rows: {len(repo_rows)}")
        lines.append("")
        lines.extend(_render_per_condition(repo_rows))
        lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)
    return report_path


def _render_per_condition(rows: list[dict[str, Any]]) -> list[str]:
    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cond[str(row.get("condition") or "<unknown>")].append(row)

    out: list[str] = []
    out.append("### Score summary")
    out.append("")
    out.append("| Condition | N | Mean | % score 5 | 1 | 2 | 3 | 4 | 5 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for cond in sorted(by_cond):
        cond_rows = by_cond[cond]
        scores = [int(r.get("judge", {}).get("score", 0)) for r in cond_rows]
        valid = [s for s in scores if 1 <= s <= 5]
        n = len(cond_rows)
        mean = (sum(valid) / len(valid)) if valid else 0.0
        pct5 = (100.0 * sum(1 for s in valid if s == 5) / len(valid)) if valid else 0.0
        hist = Counter(valid)
        out.append(
            f"| `{cond}` | {n} | {mean:.2f} | {pct5:.1f}% "
            f"| {hist.get(1, 0)} | {hist.get(2, 0)} | {hist.get(3, 0)} "
            f"| {hist.get(4, 0)} | {hist.get(5, 0)} |"
        )
    out.append("")

    out.append("### Failure modes")
    out.append("")
    header = "| Condition | " + " | ".join(FAILURE_MODES) + " |"
    sep = "|---|" + "---|" * len(FAILURE_MODES)
    out.append(header)
    out.append(sep)
    for cond in sorted(by_cond):
        cond_rows = by_cond[cond]
        tag_counts: Counter = Counter()
        for r in cond_rows:
            tags = r.get("judge", {}).get("failure_modes") or []
            tag_counts.update(tags)
        cells = [str(tag_counts.get(m, 0)) for m in FAILURE_MODES]
        out.append(f"| `{cond}` | " + " | ".join(cells) + " |")
    out.append("")
    return out
