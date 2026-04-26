#!/usr/bin/env python3
"""Show model answers per question grouped by condition from eval_results JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse an eval_results JSONL file and show answers for each question "
            "for each condition."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to eval_results JSONL file.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Only include rows with this run_id.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help="Condition filter (repeatable). Example: --condition naive --condition in_context",
    )
    parser.add_argument(
        "--show-expected",
        action="store_true",
        help="Show expected_answer under each question.",
    )
    parser.add_argument(
        "--show-qa-id",
        action="store_true",
        help="Show qa_id for each question block.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"Expected JSON object at line {line_no}")
        records.append(rec)
    return records


def build_question_key(record: dict[str, Any]) -> tuple[str, str]:
    qa_id = str(record.get("qa_id") or "")
    question = str(record.get("question") or "")
    return qa_id, question


def main() -> int:
    args = parse_args()

    try:
        records = load_jsonl(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.run_id is not None:
        records = [r for r in records if r.get("run_id") == args.run_id]

    if args.condition:
        wanted_conditions = set(args.condition)
        records = [r for r in records if str(r.get("condition")) in wanted_conditions]

    if not records:
        print("No records matched the provided filters.")
        return 0

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    condition_order: list[str] = []
    seen_conditions: set[str] = set()

    for rec in records:
        condition = str(rec.get("condition") or "")
        if condition not in seen_conditions:
            seen_conditions.add(condition)
            condition_order.append(condition)

        key = build_question_key(rec)
        if key not in grouped:
            grouped[key] = {
                "qa_id": rec.get("qa_id"),
                "question": rec.get("question"),
                "expected_answer": rec.get("expected_answer"),
                "answers": {},
            }

        grouped[key]["answers"][condition] = rec.get("answer")

    items = sorted(
        grouped.values(),
        key=lambda item: (
            str(item.get("qa_id") or ""),
            str(item.get("question") or ""),
        ),
    )

    for index, item in enumerate(items, start=1):
        print(f"Question {index}: {item.get('question', '')}")
        if args.show_qa_id:
            print(f"  qa_id: {item.get('qa_id', '')}")
        if args.show_expected:
            print(f"  expected: {item.get('expected_answer', '')}")

        answers = item["answers"]
        for condition in condition_order:
            if condition in answers:
                print(f"  [{condition}] {answers[condition]}")
            else:
                print(f"  [{condition}] <missing>")

        if index < len(items):
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
