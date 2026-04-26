"""Chunked internalization helper for doc-to-lora experiments.

The vendor's ``model.internalize(raw_text)`` sends the entire document as a
single context chunk.  When the checkpoint was trained with
``max_ctx_chunk_len > 0``, the context encoder only ever saw short chunks
during training.  Feeding a 1500-token single chunk degrades quality.

This module provides ``internalize_chunked()`` which:
  1. Tokenises the document the same way the training pipeline does.
  2. Optionally splits into chunks matching ``split_too_long_ctx()``
     (with proper CTX_AFFIXES per model).
  3. Pads multi-chunk tensors and builds ``ctx_attn_mask`` / ``ctx_position_ids``.
  4. Stores the actual ``n_ctx_chunks`` on the model so that ``generate()``
     can be called with the correct chunk count.
"""

from __future__ import annotations

from math import ceil

import torch
from ctx_to_lora.data.definitions import CTX_AFFIXES
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer
from torch.nn.utils.rnn import pad_sequence


def _chunk_ctx_ids(
    ctx_ids: list[int],
    model_name_or_path: str,
    max_chunk_len: int,
) -> list[list[int]]:
    """Split *already-tokenised* ctx_ids into chunks with proper affixes.

    Mirrors ``split_too_long_ctx`` from ``ctx_to_lora.data.processing``.
    """
    if max_chunk_len <= 0 or len(ctx_ids) <= max_chunk_len:
        return [ctx_ids]

    n_chunks = max(1, ceil(len(ctx_ids) / max_chunk_len))
    avg_len = ceil(len(ctx_ids) / n_chunks)
    chunks = [ctx_ids[i : i + avg_len] for i in range(0, len(ctx_ids), avg_len)]

    affixes = CTX_AFFIXES[model_name_or_path]
    prefix = affixes["prefix"]
    suffix = affixes["suffix"]

    # First chunk keeps its own prefix (from tokenize_ctx_text) but needs a suffix
    chunks[0] = chunks[0] + suffix
    for i in range(1, len(chunks) - 1):
        chunks[i] = prefix + chunks[i] + suffix
    # Last chunk needs a prefix but keeps its own suffix (from tokenize_ctx_text)
    if len(chunks) > 1:
        chunks[-1] = prefix + chunks[-1]

    return chunks


def internalize_chunked(
    model,
    doc_text: str,
    max_chunk_len: int = -1,
) -> int:
    """Internalize a document with optional chunking.

    Args:
        model: A ``ModulatedPretrainedModel`` instance.
        doc_text: Raw document text.
        max_chunk_len: Maximum tokens per chunk.  ``-1`` disables chunking
            (equivalent to ``model.internalize(doc_text)``).

    Returns:
        The number of context chunks used.
    """
    ctx_tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    # Step 1: tokenize exactly like the training pipeline
    ctx_ids_nested = tokenize_ctx_text(dict(context=[doc_text]), ctx_tokenizer)[
        "ctx_ids"
    ]
    ctx_ids_flat: list[int] = ctx_ids_nested[0]  # single sample → flat list

    # Step 2: optionally chunk
    ctx_model_name = model.ctx_encoder.base_model.name_or_path
    chunks = _chunk_ctx_ids(ctx_ids_flat, ctx_model_name, max_chunk_len)
    n_chunks = len(chunks)

    # Step 3: build padded tensors for _internalize_from_ids
    chunk_tensors = [torch.tensor(c, dtype=torch.long) for c in chunks]
    if n_chunks == 1:
        ctx_ids_t = chunk_tensors[0].unsqueeze(0).to(model.device)
        ctx_attn_mask = torch.ones_like(ctx_ids_t)
        ctx_position_ids = None  # _internalize_from_ids handles this
    else:
        # Pad to equal length; build matching attention mask + position_ids
        ctx_ids_t = pad_sequence(chunk_tensors, batch_first=True, padding_value=0).to(
            model.device
        )
        ctx_attn_mask = (ctx_ids_t != 0).long()
        # Position ids: 0-based within each chunk, 0 for padding
        ctx_position_ids = torch.zeros_like(ctx_ids_t)
        for i in range(n_chunks):
            seq_len = chunk_tensors[i].shape[0]
            ctx_position_ids[i, :seq_len] = torch.arange(seq_len)
        ctx_position_ids = ctx_position_ids.to(model.device)

    # Step 4: internalize — generate LoRA weights from context
    model.patch_lora_forward()
    generated_loras, _ = model.generate_weights(
        ctx_ids_t, ctx_attn_mask, ctx_position_ids
    )
    model.generated_loras = generated_loras

    # Store chunk count so callers can pass it to generate()
    model._n_ctx_chunks = n_chunks

    return n_chunks


def generate_with_chunks(model, n_ctx_chunks: int | None = None, **generate_kwargs):
    """Call model.generate() with the correct n_ctx_chunks from internalization.

    Wraps ``model.generate()`` and injects ``n_ctx_chunks`` from the most
    recent ``internalize_chunked()`` call if not explicitly provided.
    """
    if n_ctx_chunks is None:
        n_ctx_chunks = getattr(model, "_n_ctx_chunks", 1)

    return model.generate(
        n_ctx_chunks=torch.tensor((n_ctx_chunks,), device=model.device),
        **generate_kwargs,
    )
