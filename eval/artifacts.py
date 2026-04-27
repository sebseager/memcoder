"""Walk repo ledgers and yield document records for the eval harness."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import ArtifactSelector, RunConfig

LOGGER = logging.getLogger("memcoder.eval.artifacts")


@dataclass
class QAPair:
    qa_id: str
    question: str
    expected_answer: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentRecord:
    repo_id: str
    repo_root: Path           # absolute path to artifacts/<repo>/
    document_id: str
    topic: str | None
    topic_slug: str | None
    difficulty: str
    description: str | None
    doc_text: str
    doc_metadata: dict[str, Any]
    qa_pairs: list[QAPair]
    lora_path: Path | None    # absolute path to the pre-baked .pt, or None if not yet baked
    lora_relpath: str | None  # relative path as recorded in the ledger, or None


def iter_documents(cfg: RunConfig) -> Iterator[DocumentRecord]:
    """Yield one ``DocumentRecord`` per ledger entry that passes the filters."""
    for selector in cfg.artifacts:
        yield from _iter_for_selector(selector)


def _iter_for_selector(selector: ArtifactSelector) -> Iterator[DocumentRecord]:
    repo_root = selector.root
    ledger_path = repo_root / "ledger.json"
    if not ledger_path.exists():
        raise FileNotFoundError(f"ledger.json not found at {ledger_path}")

    ledger = _read_json(ledger_path)
    documents = ledger.get("documents")
    if not isinstance(documents, dict):
        raise ValueError(f"ledger.json missing 'documents' map: {ledger_path}")

    repo_meta_path = repo_root / "repo.json"
    repo_meta = _read_json(repo_meta_path) if repo_meta_path.exists() else {}
    repo_id = str(repo_meta.get("repo_id") or repo_root.name)

    diff_filter = set(selector.difficulties)
    id_filter = set(selector.document_ids)
    topic_filter = set(selector.topics)

    for document_id, entry in documents.items():
        if not isinstance(entry, dict):
            continue
        difficulty = str(entry.get("difficulty") or "")
        topic = entry.get("topic")
        topic_slug = entry.get("topic_slug")
        description = entry.get("description")

        if diff_filter and difficulty not in diff_filter:
            continue
        if id_filter and document_id not in id_filter:
            continue
        if topic_filter and (topic is None or topic not in topic_filter):
            continue

        files = entry.get("files") or {}
        doc_rel = files.get("doc")
        qa_rel = files.get("qa")
        lora_rel = files.get("lora")
        if not (isinstance(doc_rel, str) and isinstance(qa_rel, str)):
            raise ValueError(
                f"ledger entry {document_id!r} missing files.doc/files.qa in {ledger_path}"
            )

        doc_path = repo_root / doc_rel
        qa_path = repo_root / qa_rel
        for label, p in (("files.doc", doc_path), ("files.qa", qa_path)):
            if not p.exists():
                raise FileNotFoundError(
                    f"ledger entry {document_id!r} {label} -> {p} does not exist"
                )

        lora_path: Path | None
        if isinstance(lora_rel, str):
            candidate = repo_root / lora_rel
            if candidate.exists():
                lora_path = candidate
            else:
                LOGGER.warning(
                    "ledger entry %r references files.lora=%s but the file does "
                    "not exist; the shine condition will be skipped for this doc",
                    document_id,
                    candidate,
                )
                lora_path = None
                lora_rel = None
        else:
            lora_path = None
            lora_rel = None

        doc_text, doc_metadata = _load_design_doc(doc_path)
        qa_pairs = _load_qa_pairs(qa_path)

        yield DocumentRecord(
            repo_id=repo_id,
            repo_root=repo_root,
            document_id=str(document_id),
            topic=str(topic) if isinstance(topic, str) else None,
            topic_slug=str(topic_slug) if isinstance(topic_slug, str) else None,
            difficulty=difficulty,
            description=str(description) if isinstance(description, str) else None,
            doc_text=doc_text,
            doc_metadata=doc_metadata,
            qa_pairs=qa_pairs,
            lora_path=lora_path,
            lora_relpath=lora_rel,
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}, got {type(data).__name__}")
    return data


def _load_design_doc(path: Path) -> tuple[str, dict[str, Any]]:
    """Mirror ``run_shine_eval.load_design_doc``: text or {document: ...}."""
    payload = _load_json_or_text(path)
    if isinstance(payload, str):
        return payload, {"document_path": str(path)}
    if not isinstance(payload, dict):
        raise ValueError(f"design doc must be text or a JSON object: {path}")
    text = payload.get("document")
    if isinstance(text, str) and text.strip():
        return text, payload
    raise ValueError(f"could not find design doc text in {path}; expected `document`")


def _load_qa_pairs(path: Path) -> list[QAPair]:
    """Mirror ``run_shine_eval.load_qa_pairs`` but as ``QAPair`` records."""
    payload = _load_json_or_text(path)
    if isinstance(payload, dict):
        records = payload.get("qa_pairs")
        if records is None:
            raise ValueError(f"could not find `qa_pairs` in {path}")
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"qa pairs must be JSON or JSONL: {path}")

    out: list[QAPair] = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"qa record {idx} in {path} is not an object")
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"qa record {idx} in {path} has no non-empty `question`")
        out.append(
            QAPair(
                qa_id=str(record.get("qa_id") or record.get("id") or f"qa_{idx + 1:04d}"),
                question=question,
                expected_answer=record.get("answer"),
                metadata=record,
            )
        )
    return out


def _load_json_or_text(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return path.read_text(encoding="utf-8")
