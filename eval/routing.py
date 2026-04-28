"""Routing strategies for selecting the LoRA(s) used by the ``shine`` condition."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .artifacts import DocumentRecord, QAPair

LOGGER = logging.getLogger("memcoder.eval.routing")


@dataclass
class RoutingDecision:
    """The LoRA(s) chosen for a ``(doc, qa)`` pair under the ``shine`` condition.

    ``lora_paths`` is always a list — even at ``top_k=1`` it is a singleton.
    The runner averages the listed LoRAs into one composed adapter via
    ``eval.composition.compose_top_k`` (rank_average, scale=1.0).
    """

    lora_paths: list[Path]
    selected_lora_ids: list[str]
    strategy: str


class Router(Protocol):
    strategy: str

    def select(self, doc: DocumentRecord, qa: QAPair) -> RoutingDecision | None: ...


@dataclass
class OracleRouter:
    """Returns the LoRA baked from the question's source document.

    Returns ``None`` if the document has no pre-baked LoRA.
    """

    strategy: str = "oracle"

    def select(self, doc: DocumentRecord, qa: QAPair) -> RoutingDecision | None:
        if doc.lora_path is None:
            return None
        return RoutingDecision(
            lora_paths=[doc.lora_path],
            selected_lora_ids=[doc.document_id],
            strategy=self.strategy,
        )


class EmbeddingRouter:
    """Per-question routing fed by a pre-computed cosine-retrieval JSONL.

    Consumes the output of ``scripts/embedding_router.py``. Each row carries a
    ``qa_id`` and a ranked list of LoRAs with absolute ``lora_path`` fields;
    the router returns the top-``top_k`` paths for a queried ``qa.qa_id``.
    Returns ``None`` if the qa_id has no entry — the runner skips ``shine``
    for that pair and writes only the other conditions.
    """

    strategy: str = "embedding"

    def __init__(
        self,
        routing_results_paths: list[Path],
        top_k: int = 1,
    ) -> None:
        if not routing_results_paths:
            raise ValueError(
                "EmbeddingRouter requires embedding.routing_results to list at "
                "least one JSONL path"
            )
        self._top_k = max(1, int(top_k))
        self._by_qa: dict[str, dict[str, Any]] = {}
        for raw in routing_results_paths:
            self._load(Path(raw))
        if not self._by_qa:
            raise ValueError(
                "EmbeddingRouter loaded zero qa_id rows from "
                f"{[str(p) for p in routing_results_paths]}"
            )

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"routing-results JSONL not found: {path}")
        base = path.parent
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qa_id = row.get("qa_id")
            if not qa_id:
                continue
            ranked = row.get("ranked_loras") or []
            if not ranked:
                LOGGER.warning("%s:%d has no ranked_loras for qa_id=%s", path, lineno, qa_id)
                continue

            requested = self._top_k
            top = ranked[:requested]
            lora_paths: list[Path] = []
            lora_ids: list[str] = []
            for rank_idx, entry in enumerate(top):
                lora_path_raw = entry.get("lora_path")
                if not lora_path_raw:
                    LOGGER.warning(
                        "%s:%d ranked_lora rank=%d missing lora_path for qa_id=%s; "
                        "truncating top-k here",
                        path,
                        lineno,
                        rank_idx + 1,
                        qa_id,
                    )
                    break
                resolved = Path(lora_path_raw)
                if not resolved.is_absolute():
                    resolved = (base / resolved).resolve()
                lora_paths.append(resolved)
                lora_ids.append(str(entry.get("lora_id") or ""))

            if not lora_paths:
                LOGGER.warning(
                    "%s:%d top-1 ranked_lora missing lora_path for qa_id=%s",
                    path,
                    lineno,
                    qa_id,
                )
                continue

            if len(lora_paths) < requested:
                LOGGER.debug(
                    "%s:%d realized k=%d (< requested top_k=%d) for qa_id=%s",
                    path,
                    lineno,
                    len(lora_paths),
                    requested,
                    qa_id,
                )

            unique_paths = {str(p) for p in lora_paths}
            if len(unique_paths) < len(lora_paths):
                LOGGER.warning(
                    "%s:%d duplicate lora_path within top-k for qa_id=%s; "
                    "duplicates will be averaged with equal weights",
                    path,
                    lineno,
                    qa_id,
                )

            qa_id_str = str(qa_id)
            if qa_id_str in self._by_qa:
                LOGGER.warning(
                    "duplicate qa_id %s in routing-results; later entry from %s wins",
                    qa_id_str,
                    path,
                )
            self._by_qa[qa_id_str] = {
                "lora_paths": lora_paths,
                "selected_lora_ids": lora_ids,
            }

    def select(self, doc: DocumentRecord, qa: QAPair) -> RoutingDecision | None:
        row = self._by_qa.get(qa.qa_id)
        if row is None:
            return None
        return RoutingDecision(
            lora_paths=list(row["lora_paths"]),
            selected_lora_ids=list(row["selected_lora_ids"]),
            strategy=self.strategy,
        )


def make_router(
    strategy: str,
    *,
    routing_results_paths: list[Path] | None = None,
    top_k: int = 1,
) -> Router:
    if strategy == "oracle":
        return OracleRouter()
    if strategy == "embedding":
        return EmbeddingRouter(
            routing_results_paths=list(routing_results_paths or []),
            top_k=top_k,
        )
    raise ValueError(f"unknown routing strategy: {strategy!r}")
