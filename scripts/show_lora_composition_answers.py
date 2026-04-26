#!/usr/bin/env python3
"""Show individual vs composed LoRA answers grouped by question."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any


DEFAULT_WIDTH = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a LoRA composition eval JSONL file and show each question with "
            "the expected, individual, and composition answers."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to lora_composition_results JSONL.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Only include rows with this run_id.",
    )
    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help="Only include a document_id. Repeatable.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show raw_generation instead of the parsed answer field.",
    )
    parser.add_argument(
        "--hide-scores",
        action="store_true",
        help="Do not show token_f1 / exact_or_contains score lines.",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Only show questions where individual and composition answers differ.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Wrap long text to this width. Default: {DEFAULT_WIDTH}.",
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


def question_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("document_id") or ""),
        str(record.get("qa_id") or ""),
        str(record.get("question") or ""),
    )


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def format_score(record: dict[str, Any]) -> str:
    scores = record.get("scores")
    if not isinstance(scores, dict):
        return "score: <missing>"

    token_f1 = scores.get("token_f1")
    contains = scores.get("exact_or_contains")
    token_f1_text = f"{float(token_f1):.3f}" if isinstance(token_f1, (int, float)) else "<missing>"
    contains_text = str(contains).lower() if isinstance(contains, bool) else "<missing>"
    return f"score: token_f1={token_f1_text}, exact_or_contains={contains_text}"


def print_wrapped(label: str, value: Any, width: int) -> None:
    text = str(value if value is not None else "<missing>")
    prefix = f"  {label}: "
    subsequent = " " * len(prefix)
    print(textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=subsequent))


def main() -> int:
    args = parse_args()

    try:
        records = load_jsonl(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.run_id is not None:
        records = [record for record in records if record.get("run_id") == args.run_id]

    if args.document_id:
        wanted_docs = set(args.document_id)
        records = [record for record in records if str(record.get("document_id")) in wanted_docs]

    if not records:
        print("No records matched the provided filters.")
        return 0

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = question_key(record)
        item = grouped.setdefault(
            key,
            {
                "document_id": record.get("document_id"),
                "topic": record.get("topic"),
                "qa_id": record.get("qa_id"),
                "question": record.get("question"),
                "expected_answer": record.get("expected_answer"),
                "runs": {},
            },
        )
        item["runs"][str(record.get("condition") or "")] = record

    items = sorted(
        grouped.values(),
        key=lambda item: (
            str(item.get("document_id") or ""),
            str(item.get("qa_id") or ""),
            str(item.get("question") or ""),
        ),
    )

    shown = 0
    for item in items:
        runs = item["runs"]
        individual = runs.get("individual")
        composition = runs.get("composition")
        individual_answer = normalize_text((individual or {}).get("answer"))
        composition_answer = normalize_text((composition or {}).get("answer"))

        if args.only_changed and individual_answer == composition_answer:
            continue

        shown += 1
        title = f"Question {shown}"
        topic = item.get("topic")
        if topic:
            title += f" [{topic}]"
        print(title)
        print(f"  document_id: {item.get('document_id', '')}")
        print(f"  qa_id: {item.get('qa_id', '')}")
        print_wrapped("question", item.get("question", ""), args.width)
        print_wrapped("expected", item.get("expected_answer", ""), args.width)

        for condition in ("individual", "composition"):
            record = runs.get(condition)
            if record is None:
                print(f"  {condition}: <missing>")
                continue

            answer_field = "raw_generation" if args.show_raw else "answer"
            print_wrapped(condition, record.get(answer_field, ""), args.width)
            if not args.hide_scores:
                print(f"  {condition} {format_score(record)}")

        extra_conditions = sorted(set(runs) - {"individual", "composition", ""})
        for condition in extra_conditions:
            record = runs[condition]
            answer_field = "raw_generation" if args.show_raw else "answer"
            print_wrapped(condition, record.get(answer_field, ""), args.width)
            if not args.hide_scores:
                print(f"  {condition} {format_score(record)}")

        print()

    if shown == 0:
        print("No questions matched the provided filters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
