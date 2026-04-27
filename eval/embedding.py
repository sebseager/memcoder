"""Embedding-model loader.

Placeholder: the embedding routing path is not exercised in v0. The function
exists so :mod:`eval.routing` has a clear seam — when embedding routing is
wired up, this is the only module that needs to learn how to load and call a
sentence embedder.
"""

from __future__ import annotations


def load_embedder(name: str | None):
    raise NotImplementedError(
        "embedding pipeline is a placeholder; populate ledger doc_embedding "
        "fields and implement eval.embedding.load_embedder before enabling "
        "routing: embedding"
    )
