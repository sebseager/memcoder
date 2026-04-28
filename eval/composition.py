"""Compose multiple LoRA dicts into a single adapter via rank-concatenation.

Ported from ``scripts/run_routed_lora_eval.py`` (the legacy collaborator
script). The math: a LoraLinear layer computes ``(x @ A) @ B + C``. Two
adapters' deltas sum, so concatenating each pair's A along its rank axis
(scaled by per-adapter weight) and B along its rank axis yields a single
A'/B' whose contribution equals the weighted sum of the originals'. The
optional bias C is averaged with the same weights.
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any:
    """Return the torch module loaded by ``eval.model.ensure_runtime``.

    Composition only runs inside the predictions phase, after the runner has
    already called ``ensure_runtime``. Pulling torch through ``model_module``
    keeps this module importable without torch present (mirrors the lazy
    pattern in the rest of ``eval/``).
    """
    from . import model as model_module

    if model_module.torch is None:
        raise RuntimeError(
            "torch is not loaded; call eval.model.ensure_runtime() before "
            "using compose_lora_dicts/compose_top_k"
        )
    return model_module.torch


def composition_weights(count: int, method: str, scale: float) -> list[float]:
    if count <= 0:
        raise ValueError("At least one LoRA is required for composition")
    if method == "rank_sum":
        return [scale] * count
    if method == "rank_average":
        return [scale / count] * count
    raise ValueError(f"Unknown composition method: {method}")


def compose_lora_dicts(lora_dicts: list[Any], weights: list[float]) -> Any:
    """Compose LoRAs by rank-concat while weighting each adapter's output delta.

    LoraLinear computes ``(x @ A) @ B + C``. Concatenating A/B ranks sums
    deltas, so averaging means scaling one side of each adapter pair, plus
    C, by 1/N.
    """
    if len(lora_dicts) != len(weights):
        raise ValueError("lora_dicts and weights must have the same length")
    if not lora_dicts:
        raise ValueError("Cannot compose an empty LoRA list")

    torch = _torch()

    def _compose(nodes: list[Any], path: str) -> Any:
        first = nodes[0]
        if isinstance(first, dict) and "A" in first and "B" in first:
            if not all(isinstance(node, dict) and "A" in node and "B" in node for node in nodes):
                raise TypeError(f"{path}: all leaves must be LoRA A/B dictionaries")

            a_tensors = [node["A"] for node in nodes]
            b_tensors = [node["B"] for node in nodes]
            c_tensors = [node.get("C", None) for node in nodes]

            for idx, (a_tensor, b_tensor) in enumerate(zip(a_tensors, b_tensors, strict=True)):
                if a_tensor is None or b_tensor is None:
                    raise ValueError(f"{path}: A/B cannot be None for adapter {idx}")
                if a_tensor.shape[0] != a_tensors[0].shape[0]:
                    raise ValueError(f"{path}.A: Lb mismatch for adapter {idx}")
                if b_tensor.shape[0] != b_tensors[0].shape[0]:
                    raise ValueError(f"{path}.B: Lb mismatch for adapter {idx}")
                if a_tensor.shape[1] != a_tensors[0].shape[1]:
                    raise ValueError(f"{path}.A: in_features mismatch for adapter {idx}")
                if b_tensor.shape[2] != b_tensors[0].shape[2]:
                    raise ValueError(f"{path}.B: out_features mismatch for adapter {idx}")
                if a_tensor.shape[2] != b_tensor.shape[1]:
                    raise ValueError(f"{path}: rank mismatch inside adapter {idx}")

            has_c = [c_tensor is not None for c_tensor in c_tensors]
            if any(has_c) and not all(has_c):
                raise ValueError(f"{path}.C: either all adapters must have C or none may have C")

            return {
                "A": torch.cat(
                    [a_tensor * weight for a_tensor, weight in zip(a_tensors, weights, strict=True)],
                    dim=2,
                ),
                "B": torch.cat(b_tensors, dim=1),
                "C": None
                if not any(has_c)
                else sum(c_tensor * weight for c_tensor, weight in zip(c_tensors, weights, strict=True)),
            }

        if isinstance(first, dict):
            keys = set(first.keys())
            for idx, node in enumerate(nodes):
                if not isinstance(node, dict):
                    raise TypeError(f"{path}: adapter {idx} is not a dictionary")
                if set(node.keys()) != keys:
                    raise ValueError(f"{path}: key mismatch for adapter {idx}")
            return {
                key: _compose(
                    [node[key] for node in nodes],
                    path=f"{path}.{key}" if path else str(key),
                )
                for key in first.keys()
            }

        raise TypeError(f"{path}: unsupported LoRA node type {type(first)}")

    return _compose(lora_dicts, "")


def compose_top_k(loaded_dicts: list[Any]) -> Any:
    """Average N pre-loaded LoRA dicts (rank_average, scale=1.0).

    For ``N == 1`` returns the singleton unchanged so the predictions for
    oracle routing and ``top_k=1`` embedding routing are bit-for-bit
    identical to the pre-composition harness behavior.
    """
    if not loaded_dicts:
        raise ValueError("compose_top_k requires at least one LoRA dict")
    if len(loaded_dicts) == 1:
        return loaded_dicts[0]
    weights = composition_weights(len(loaded_dicts), "rank_average", 1.0)
    return compose_lora_dicts(loaded_dicts, weights)
