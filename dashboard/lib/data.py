from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from eval.artifacts import _load_design_doc, _load_qa_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


@dataclass(frozen=True)
class DocumentView:
    repo_id: str
    repo_root: str
    document_id: str
    topic: str
    topic_slug: str
    difficulty: str
    description: str
    generator: str
    token_count: int
    doc_text: str
    doc_metadata: dict[str, Any]
    qa_pairs: list[dict[str, Any]]
    example_questions: list[str]
    files: dict[str, Any]
    lora_path: str | None
    lora_exists: bool


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


@st.cache_data(show_spinner=False)
def list_repo_ids() -> list[str]:
    if not ARTIFACTS_ROOT.exists():
        return []
    return sorted(
        p.name
        for p in ARTIFACTS_ROOT.iterdir()
        if p.is_dir() and (p / "ledger.json").exists()
    )


@st.cache_data(show_spinner=False)
def load_repo_meta(repo_id: str) -> dict[str, Any]:
    repo_root = ARTIFACTS_ROOT / repo_id
    path = repo_root / "repo.json"
    if path.exists():
        return read_json(path)
    return {"repo_id": repo_id}


@st.cache_data(show_spinner=False)
def load_ledger(repo_id: str) -> dict[str, Any]:
    return read_json(ARTIFACTS_ROOT / repo_id / "ledger.json")


@st.cache_data(show_spinner=False)
def load_documents(repo_id: str) -> list[DocumentView]:
    repo_root = ARTIFACTS_ROOT / repo_id
    meta = load_repo_meta(repo_id)
    ledger = load_ledger(repo_id)
    docs = ledger.get("documents") or {}
    out: list[DocumentView] = []

    for document_id, entry in docs.items():
        if not isinstance(entry, dict):
            continue
        files = entry.get("files") or {}
        doc_rel = files.get("doc")
        qa_rel = files.get("qa")
        if not isinstance(doc_rel, str) or not isinstance(qa_rel, str):
            continue

        doc_path = repo_root / doc_rel
        qa_path = repo_root / qa_rel
        if not doc_path.exists() or not qa_path.exists():
            continue

        doc_text, doc_metadata = _load_design_doc(doc_path)
        qa_pairs = [
            {
                "qa_id": qa.qa_id,
                "question": qa.question,
                "answer": qa.expected_answer,
                "metadata": qa.metadata,
            }
            for qa in _load_qa_pairs(qa_path)
        ]
        example_questions = _load_example_questions(repo_root, files.get("qa_examples"))
        lora_rel = files.get("lora")
        lora_path = str(repo_root / lora_rel) if isinstance(lora_rel, str) else None

        out.append(
            DocumentView(
                repo_id=str(meta.get("repo_id") or repo_id),
                repo_root=str(repo_root),
                document_id=str(entry.get("document_id") or document_id),
                topic=str(entry.get("topic") or ""),
                topic_slug=str(entry.get("topic_slug") or ""),
                difficulty=str(entry.get("difficulty") or ""),
                description=str(entry.get("description") or ""),
                generator=str(doc_metadata.get("generator") or entry.get("generator") or ""),
                token_count=_token_count(doc_text, doc_metadata),
                doc_text=doc_text,
                doc_metadata=doc_metadata,
                qa_pairs=qa_pairs,
                example_questions=example_questions,
                files=files,
                lora_path=lora_path,
                lora_exists=bool(lora_path and Path(lora_path).exists()),
            )
        )

    return sorted(out, key=lambda d: (d.difficulty, d.topic, d.document_id))


def documents_for_difficulty(repo_id: str, difficulty: str) -> list[DocumentView]:
    docs = load_documents(repo_id)
    if difficulty and difficulty != "all":
        docs = [doc for doc in docs if doc.difficulty == difficulty]
    return docs


def list_difficulties(repo_id: str) -> list[str]:
    difficulties = sorted({doc.difficulty for doc in load_documents(repo_id) if doc.difficulty})
    return ["all", *difficulties] if difficulties else ["all"]


def document_by_id(repo_id: str, document_id: str) -> DocumentView | None:
    for doc in load_documents(repo_id):
        if doc.document_id == document_id:
            return doc
    return None


def documents_table(docs: list[DocumentView]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "document_id": doc.document_id,
                "topic": doc.topic,
                "difficulty": doc.difficulty,
                "generator": doc.generator,
                "token_count": doc.token_count,
                "qa_count": len(doc.qa_pairs),
                "lora": "present" if doc.lora_exists else ("missing" if doc.lora_path else "none"),
            }
            for doc in docs
        ]
    )


def qa_table(doc: DocumentView) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "qa_id": row["qa_id"],
                "question": row["question"],
                "answer": _short_text(row.get("answer")),
            }
            for row in doc.qa_pairs
        ]
    )


def lora_options(repo_id: str, difficulty: str) -> list[DocumentView]:
    return [
        doc
        for doc in documents_for_difficulty(repo_id, difficulty)
        if doc.lora_path or doc.example_questions or doc.qa_pairs
    ]


def question_options(doc: DocumentView) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qa in doc.qa_pairs:
        rows.append(
            {
                "label": f"QA: {qa['question']}",
                "question": qa["question"],
                "answer": qa.get("answer"),
                "qa_id": qa.get("qa_id"),
                "source": "ground truth QA",
            }
        )
    for idx, question in enumerate(doc.example_questions, start=1):
        rows.append(
            {
                "label": f"Example: {question}",
                "question": question,
                "answer": None,
                "qa_id": f"example_{idx}",
                "source": "example question",
            }
        )
    return rows


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", compact) if part.strip()]


@st.cache_data(show_spinner=False)
def discover_result_logs(repo_id: str) -> list[dict[str, str]]:
    repo_root = ARTIFACTS_ROOT / repo_id
    if not repo_root.exists():
        return []
    candidates: list[dict[str, str]] = []
    for path in sorted(repo_root.rglob("*.jsonl")):
        name = path.name
        if not any(bit in name for bit in ("judgments", "eval_results", "lora_composition_results", "predictions")):
            continue
        rel = path.relative_to(PROJECT_ROOT)
        kind = "judged" if "judgments" in name else "predictions"
        candidates.append({"label": f"{rel} ({kind})", "path": str(path), "kind": kind})
    return candidates


@st.cache_data(show_spinner=False)
def load_result_rows(repo_id: str, paths: tuple[str, ...]) -> pd.DataFrame:
    docs = {doc.document_id: doc for doc in load_documents(repo_id)}
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            rows.append(_flatten_result_row(row, path, docs))
    return pd.DataFrame(rows)


def _flatten_result_row(row: dict[str, Any], path: Path, docs: dict[str, DocumentView]) -> dict[str, Any]:
    judge = row.get("judge") or {}
    scores = row.get("scores") or {}
    document_id = str(row.get("document_id") or (row.get("qa_metadata") or {}).get("document_id") or "")
    doc = docs.get(document_id)
    raw_condition = str(row.get("condition") or "")
    score = judge.get("score")
    score_kind = "judge_score"
    if score is None:
        score = scores.get("token_f1")
        score_kind = "token_f1"

    return {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "run_id": row.get("run_id"),
        "repo_id": row.get("repo_id") or (doc.repo_id if doc else ""),
        "document_id": document_id,
        "topic": row.get("topic") or (doc.topic if doc else ""),
        "difficulty": row.get("difficulty") or (doc.difficulty if doc else ""),
        "qa_id": row.get("qa_id"),
        "condition": raw_condition,
        "condition_display": normalize_condition(raw_condition),
        "routing_strategy": (row.get("routing") or {}).get("strategy")
        or row.get("composition_method")
        or "n/a",
        "question": row.get("question"),
        "expected_answer": row.get("expected_answer"),
        "answer": row.get("answer"),
        "score": score,
        "score_kind": score_kind,
        "judge_reasoning": judge.get("reasoning"),
        "failure_modes": judge.get("failure_modes") or [],
        "failure_mode_notes": judge.get("failure_mode_notes"),
        "routing": row.get("routing") or {},
        "raw": row,
    }


def normalize_condition(condition: str) -> str:
    lower = condition.lower()
    if lower == "naive":
        return "Naive"
    if lower == "in_context":
        return "In-Context"
    if lower.startswith("shine"):
        return "SHINE"
    if lower == "individual":
        return "Individual LoRA"
    if lower == "composition":
        return "Composed LoRA"
    return condition or "Unknown"


def _load_example_questions(repo_root: Path, relpath: Any) -> list[str]:
    if not isinstance(relpath, str):
        return []
    path = repo_root / relpath
    if not path.exists():
        return []
    payload = read_json(path)
    if isinstance(payload, dict):
        examples = payload.get("example_questions") or payload.get("questions") or []
    else:
        examples = payload
    if not isinstance(examples, list):
        return []
    return [str(q) for q in examples if str(q).strip()]


def _token_count(text: str, metadata: dict[str, Any]) -> int:
    for key in ("token_count", "tokens", "num_tokens"):
        value = metadata.get(key)
        if isinstance(value, int):
            return value
    return len(re.findall(r"\S+", text))


def _short_text(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 1] + "..."
