"""Routing strategies for selecting the LoRA used by the ``shine`` condition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .artifacts import DocumentRecord, QAPair


@dataclass
class RoutingDecision:
    """The LoRA chosen for a ``(doc, qa)`` pair under the ``shine`` condition."""

    lora_path: Path
    selected_lora_ids: list[str]
    strategy: str


class Router(Protocol):
    strategy: str

    def select(self, doc: DocumentRecord, qa: QAPair) -> RoutingDecision: ...


@dataclass
class OracleRouter:
    """Returns the LoRA baked from the question's source document.

    This is the only router exercised in v0. It is fully deterministic and
    does no retrieval — the doc is known by construction since the eval
    harness iterates ``(doc, qa)`` from the ledger.
    """

    strategy: str = "oracle"

    def select(self, doc: DocumentRecord, qa: QAPair) -> RoutingDecision:
        return RoutingDecision(
            lora_path=doc.lora_path,
            selected_lora_ids=[doc.document_id],
            strategy=self.strategy,
        )


class EmbeddingRouter:
    """Placeholder for cosine-similarity LoRA retrieval.

    Will embed the question with the configured embedder and rank against
    ``doc_embedding`` vectors stored alongside each ledger entry. Not used
    in v0.
    """

    strategy: str = "embedding"

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "EmbeddingRouter is a placeholder; populate ledger doc_embedding "
            "fields and implement eval.embedding.load_embedder first"
        )

    def select(self, doc: DocumentRecord, qa: QAPair) -> RoutingDecision:  # pragma: no cover
        raise NotImplementedError


def make_router(strategy: str) -> Router:
    if strategy == "oracle":
        return OracleRouter()
    if strategy == "embedding":
        return EmbeddingRouter()
    raise ValueError(f"unknown routing strategy: {strategy!r}")
