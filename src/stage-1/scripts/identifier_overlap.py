from __future__ import annotations

import argparse
import ast
import json
import keyword
import re

import pandas as pd
from config import MODEL_ID, TRUNCATION_BUDGET_TOKENS, get_stage1_paths
from helpers import load_instances, truncate_to_budget
from transformers import AutoTokenizer

IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute Stage 1 identifier-overlap diagnostics for condition D"
    )
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--trunc-budget", type=int, default=TRUNCATION_BUDGET_TOKENS)
    p.add_argument("--condition", default="D", choices=["B", "C", "D"])
    return p.parse_args()


def regex_identifiers(text: str) -> set[str]:
    ids = set(IDENT_RE.findall(text or ""))
    return {x for x in ids if not keyword.iskeyword(x)}


def ast_identifiers(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return regex_identifiers(text)

    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            names.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            names.add(node.attr)
            self.generic_visit(node)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            names.add(node.name)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            names.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            names.add(node.name)
            self.generic_visit(node)

        def visit_arg(self, node: ast.arg) -> None:
            names.add(node.arg)
            self.generic_visit(node)

    Visitor().visit(tree)
    return {x for x in names if not keyword.iskeyword(x)}


def completion_identifiers(masked_function: str, predicted_body: str) -> set[str]:
    sig = (masked_function or "def _f():").splitlines()[0]
    src = f"{sig}\n{predicted_body or ''}\n"
    ids = ast_identifiers(src)
    if ids:
        return ids

    fallback_src = "def _tmp():\n" + (predicted_body or "") + "\n"
    return ast_identifiers(fallback_src)


def main() -> int:
    args = parse_args()
    paths = get_stage1_paths(args.model_id)
    paths.analysis.mkdir(parents=True, exist_ok=True)

    completions_path = paths.completions / f"condition_{args.condition}.jsonl"
    if not completions_path.exists():
        raise FileNotFoundError(f"Missing completions file: {completions_path}")

    instances = load_instances()
    by_id = {row["instance_id"]: row for row in instances}

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    rows = []
    with completions_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue

            iid = rec["instance_id"]
            inst = by_id.get(iid)
            if inst is None:
                continue

            full_file = inst.get("full_file", "")
            truncated = truncate_to_budget(full_file, tokenizer, args.trunc_budget)

            comp_ids = completion_identifiers(
                rec.get("masked_function", ""),
                rec.get("predicted_body", ""),
            )
            full_ids = ast_identifiers(full_file)
            trunc_ids = ast_identifiers(truncated)

            novel_ids = sorted(comp_ids.intersection(full_ids.difference(trunc_ids)))
            novel_count = len(novel_ids)
            comp_count = len(comp_ids)
            novel_rate = 0.0 if comp_count == 0 else novel_count / comp_count

            rows.append(
                {
                    "instance_id": iid,
                    "condition": args.condition,
                    "file_key": rec.get("file_key", ""),
                    "repo": rec.get("repo", ""),
                    "file_path": rec.get("file_path", ""),
                    "completion_identifier_count": comp_count,
                    "novel_identifier_count": novel_count,
                    "novel_identifier_rate": novel_rate,
                    "novel_identifiers": novel_ids,
                }
            )

    out_jsonl = paths.analysis / f"identifier_overlap_condition_{args.condition}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    flat = []
    for row in rows:
        flat.append(
            {
                "instance_id": row["instance_id"],
                "condition": row["condition"],
                "file_key": row["file_key"],
                "repo": row["repo"],
                "file_path": row["file_path"],
                "completion_identifier_count": row["completion_identifier_count"],
                "novel_identifier_count": row["novel_identifier_count"],
                "novel_identifier_rate": row["novel_identifier_rate"],
                "novel_identifiers": ";".join(row["novel_identifiers"]),
            }
        )

    out_csv = paths.analysis / f"identifier_overlap_condition_{args.condition}.csv"
    pd.DataFrame(flat).to_csv(out_csv, index=False)

    summary = {
        "condition": args.condition,
        "n_rows": len(rows),
        "mean_novel_identifier_rate": float(
            pd.Series([r["novel_identifier_rate"] for r in rows]).mean()
        )
        if rows
        else 0.0,
        "mean_novel_identifier_count": float(
            pd.Series([r["novel_identifier_count"] for r in rows]).mean()
        )
        if rows
        else 0.0,
        "input_jsonl": str(completions_path),
        "output_jsonl": str(out_jsonl),
        "output_csv": str(out_csv),
    }
    with (
        paths.analysis / f"identifier_overlap_condition_{args.condition}.summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Identifier overlap complete: n={summary['n_rows']}, "
        f"mean_novel_identifier_rate={summary['mean_novel_identifier_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
