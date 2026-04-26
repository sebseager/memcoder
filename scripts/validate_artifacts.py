#!/usr/bin/env python3
"""
Validate all artifacts in the artifacts/ directory against their schemas.

Run with:
    uv run scripts/validate_artifacts.py
    uv run scripts/validate_artifacts.py --artifacts artifacts/ --schemas schemas/
    uv run scripts/validate_artifacts.py --fail-fast
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator, SchemaError


# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------

ANSI = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "red":    "\033[31m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "grey":   "\033[90m",
}


def _c(color: str, text: str) -> str:
    """Wrap *text* in the named ANSI colour, resetting afterwards."""
    return f"{ANSI[color]}{text}{ANSI['reset']}"


# ---------------------------------------------------------------------------
# Schema routing
# ---------------------------------------------------------------------------

def _route(path: Path) -> str | None:
    """Return a schema key for *path*, or None to skip."""
    parts = path.parts

    # artifacts/{repo}/{difficulty}/docs/{name}.json  -> design_doc
    if "docs" in parts and path.suffix == ".json":
        return "design_doc"

    # artifacts/{repo}/{difficulty}/qas/{name}.json  -> qa_pairs
    if "qas" in parts and path.suffix == ".json":
        return "qa_pairs"

    # artifacts/{repo}/{difficulty}/lora_store.json  -> lora_store
    if path.name == "lora_store.json":
        return "lora_store"

    # artifacts/{repo}/{difficulty}/eval_results.jsonl  -> eval_result (per line)
    if path.name == "eval_results.jsonl":
        return "eval_result_jsonl"

    # artifacts/{repo}/{difficulty}/*.meta.json  -> eval_run_meta
    if path.name.endswith(".meta.json"):
        return "eval_run_meta"

    return None


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class ValidationError:
    path: str
    line: int | None   # set for JSONL; None for plain JSON
    message: str


@dataclass
class Summary:
    checked: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[ValidationError] = field(default_factory=list)

    def record_pass(self, path: Path):
        self.checked += 1
        self.passed += 1
        print(f"  {_c('green', 'PASS')}  {path}")

    def record_fail(self, path: Path, errs: list[ValidationError]):
        self.checked += 1
        self.failed += 1
        self.errors.extend(errs)
        print(f"  {_c('red', 'FAIL')}  {path}  ({len(errs)} error{'s' if len(errs) != 1 else ''})")
        for e in errs:
            loc = f" line {e.line}" if e.line is not None else ""
            print(f"        {_c('yellow', '-->')}{loc} {e.message}")

    def record_skip(self, path: Path, reason: str):
        self.skipped += 1
        print(f"  {_c('grey', 'SKIP')}  {path}  {_c('grey', f'({reason})')}")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_instance(validator: Draft202012Validator, instance: object,
                        path: Path, line: int | None) -> list[ValidationError]:
    return [
        ValidationError(str(path), line, e.message)
        for e in validator.iter_errors(instance)
    ]


def validate_json(validator: Draft202012Validator, path: Path,
                  summary: Summary, fail_fast: bool) -> bool:
    """Validate a single JSON file. Returns True if valid."""
    try:
        instance = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        errs = [ValidationError(str(path), None, f"JSON parse error: {exc}")]
        summary.record_fail(path, errs)
        if fail_fast:
            raise SystemExit(1)
        return False

    errs = _validate_instance(validator, instance, path, None)
    if errs:
        summary.record_fail(path, errs)
        if fail_fast:
            raise SystemExit(1)
        return False

    summary.record_pass(path)
    return True


def validate_jsonl(validator: Draft202012Validator, path: Path,
                   summary: Summary, fail_fast: bool) -> bool:
    """Validate every non-empty line of a JSONL file. Returns True if all valid."""
    all_ok = True
    file_errors: list[ValidationError] = []

    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            instance = json.loads(raw)
        except json.JSONDecodeError as exc:
            file_errors.append(ValidationError(str(path), lineno,
                                                f"JSON parse error: {exc}"))
            all_ok = False
            continue
        line_errs = _validate_instance(validator, instance, path, lineno)
        if line_errs:
            file_errors.extend(line_errs)
            all_ok = False

    if file_errors:
        summary.record_fail(path, file_errors)
        if fail_fast:
            raise SystemExit(1)
    else:
        summary.record_pass(path)

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate memcoder artifacts against JSON schemas.")
    parser.add_argument("--artifacts", default=str(Path(__file__).parents[1] / "artifacts"), type=Path,
                        help="Root artifacts directory (default: <repo root>/artifacts/).")
    parser.add_argument("--schemas", default=str(Path(__file__).parents[1] / "schemas"), type=Path,
                        help="Schemas directory (default: <repo root>/schemas/).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Exit immediately on the first validation failure.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print skipped files too.")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts)
    schemas_dir = Path(args.schemas)

    if not artifacts_root.exists():
        sys.exit(f"Artifacts directory not found: {artifacts_root}")
    if not schemas_dir.exists():
        sys.exit(f"Schemas directory not found: {schemas_dir}")

    # ---- Load and pre-compile schemas ----
    schema_files = {
        "design_doc":        "design_doc.schema.json",
        "qa_pairs":          "qa_pairs.schema.json",
        "lora_store":        "lora_store.schema.json",
        "eval_result_jsonl": "eval_result.schema.json",   # JSONL, one record per line
        "eval_run_meta":     "eval_run_meta.schema.json",
        "ledger_entry":      "ledger_entry.schema.json",
    }

    validators: dict[str, Draft202012Validator] = {}
    print("Loading schemas ...")
    for key, filename in schema_files.items():
        schema_path = schemas_dir / filename
        if not schema_path.exists():
            print(f"  {_c('yellow', 'WARN')}  Schema missing, skipping key '{key}': {schema_path}")
            continue
        try:
            schema = json.loads(schema_path.read_text())
            Draft202012Validator.check_schema(schema)
            validators[key] = Draft202012Validator(schema)
            print(f"  {_c('green', 'PASS')}  {schema_path}")
        except (json.JSONDecodeError, SchemaError) as exc:
            sys.exit(f"Invalid schema {schema_path}: {exc}")

    print()

    # ---- Walk artifacts ----
    summary = Summary()
    all_files = sorted(artifacts_root.rglob("*"))

    section = None
    for path in all_files:
        if not path.is_file():
            continue

        # Print a section header when we enter a new sub-directory
        parent = str(path.parent)
        if parent != section:
            section = parent
            print(f"\n{_c('bold', str(path.parent))}")

        key = _route(path)

        if key is None:
            if args.verbose:
                summary.record_skip(path, "no schema mapping")
            else:
                summary.skipped += 1
            continue

        if key not in validators:
            summary.record_skip(path, f"schema '{key}' not loaded")
            continue

        v = validators[key]

        if key == "eval_result_jsonl":
            validate_jsonl(v, path, summary, args.fail_fast)
        else:
            validate_json(v, path, summary, args.fail_fast)

    # ---- Summary ----
    total = summary.checked
    print(f"\n{'─' * 60}")
    print(f"Checked: {total}  |  "
          f"{_c('green', f'Passed: {summary.passed}')}  |  "
          f"{_c('red', f'Failed: {summary.failed}')}  |  "
          f"{_c('grey', f'Skipped: {summary.skipped}')}")

    if summary.failed:
        print(f"\n{_c('red', f'Validation failed -- {summary.failed} file(s) have errors.')}")
        sys.exit(1)
    else:
        print(f"\n{_c('green', f'All {summary.passed} checked artifact(s) are valid.')}")


if __name__ == "__main__":
    main()