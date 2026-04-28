"""Unit tests for ``eval.composition``.

Run as: ``python -m tests.test_composition`` (no pytest needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from eval import model as model_module  # noqa: E402

# Composition uses torch lazily through model_module; populate the slot so
# ``compose_lora_dicts`` can resolve it without going through
# ``ensure_runtime`` (which would try to set up the full Qwen runtime).
model_module.torch = torch

from eval import composition  # noqa: E402


def _build_leaf(rank: int, in_features: int, out_features: int, lb: int = 1) -> dict:
    """Hand-build a single LoRA leaf (A, B, C) matching LoraQwen.LoraLinear shapes.

    From vendor/SHINE/LoraQwen.py:42-44:
        A: [Lb, in, r], B: [Lb, r, out], C: [Lb, out] (broadcast over batch/seq).
    """
    a = torch.randn(lb, in_features, rank, dtype=torch.float64)
    b = torch.randn(lb, rank, out_features, dtype=torch.float64)
    c = torch.randn(lb, out_features, dtype=torch.float64)
    return {"A": a, "B": b, "C": c}


def test_singleton_values_match() -> None:
    """compose_lora_dicts([X], [1.0]) preserves X's tensor values."""
    torch.manual_seed(0)
    x = _build_leaf(rank=2, in_features=4, out_features=4)
    composed = composition.compose_lora_dicts([x], [1.0])
    assert composed["A"].shape == x["A"].shape, composed["A"].shape
    assert composed["B"].shape == x["B"].shape, composed["B"].shape
    assert composed["C"].shape == x["C"].shape, composed["C"].shape
    assert torch.allclose(composed["A"], x["A"]), "A mismatch"
    assert torch.allclose(composed["B"], x["B"]), "B mismatch"
    assert torch.allclose(composed["C"], x["C"]), "C mismatch"


def test_two_lora_average_forward_matches() -> None:
    """compose_lora_dicts([X, Y], [0.5, 0.5]) → forward equals 0.5*(X) + 0.5*(Y)."""
    torch.manual_seed(1)
    in_features, out_features, rank = 4, 4, 2
    x = _build_leaf(rank=rank, in_features=in_features, out_features=out_features)
    y = _build_leaf(rank=rank, in_features=in_features, out_features=out_features)

    weights = [0.5, 0.5]
    composed = composition.compose_lora_dicts([x, y], weights)

    # Composed shapes: ranks concatenate; C is averaged.
    assert composed["A"].shape == (1, in_features, 2 * rank), composed["A"].shape
    assert composed["B"].shape == (1, 2 * rank, out_features), composed["B"].shape
    assert composed["C"].shape == (1, out_features), composed["C"].shape

    # Forward (per Lb=1, single beam): (x_in @ A[0]) @ B[0] + C[0].
    x_in = torch.randn(3, in_features, dtype=torch.float64)

    def fwd(leaf: dict, weight: float) -> torch.Tensor:
        delta = (x_in @ leaf["A"][0]) @ leaf["B"][0] + leaf["C"][0]
        return weight * delta

    expected = fwd(x, weights[0]) + fwd(y, weights[1])
    actual = (x_in @ composed["A"][0]) @ composed["B"][0] + composed["C"][0]
    assert torch.allclose(actual, expected, atol=1e-9), \
        f"forward mismatch: max diff {(actual - expected).abs().max().item()}"


def test_compose_top_k_singleton_is_identity() -> None:
    """compose_top_k([X]) returns the singleton unchanged (no copy)."""
    torch.manual_seed(2)
    x = _build_leaf(rank=2, in_features=4, out_features=4)
    out = composition.compose_top_k([x])
    assert out is x, "compose_top_k must short-circuit at N=1 to preserve identity"


def test_compose_top_k_two_uses_uniform_weights() -> None:
    """compose_top_k([X, Y]) equals compose_lora_dicts([X, Y], [0.5, 0.5])."""
    torch.manual_seed(3)
    in_features, out_features, rank = 4, 4, 2
    x = _build_leaf(rank=rank, in_features=in_features, out_features=out_features)
    y = _build_leaf(rank=rank, in_features=in_features, out_features=out_features)
    via_top_k = composition.compose_top_k([x, y])
    via_explicit = composition.compose_lora_dicts([x, y], [0.5, 0.5])
    assert torch.allclose(via_top_k["A"], via_explicit["A"])
    assert torch.allclose(via_top_k["B"], via_explicit["B"])
    assert torch.allclose(via_top_k["C"], via_explicit["C"])


def test_nested_dict_structure_preserved() -> None:
    """Composition recurses into nested dicts (e.g. {layer_0: {q: leaf}})."""
    torch.manual_seed(4)
    leaf = _build_leaf(rank=2, in_features=4, out_features=4)
    nested = {"layer_0": {"q": leaf}}
    composed = composition.compose_lora_dicts([nested], [1.0])
    assert "layer_0" in composed and "q" in composed["layer_0"]
    assert torch.allclose(composed["layer_0"]["q"]["A"], leaf["A"])


def test_shape_mismatch_raises() -> None:
    """Mismatched in_features should raise."""
    torch.manual_seed(5)
    x = _build_leaf(rank=2, in_features=4, out_features=4)
    y = _build_leaf(rank=2, in_features=8, out_features=4)
    raised = False
    try:
        composition.compose_lora_dicts([x, y], [0.5, 0.5])
    except ValueError:
        raised = True
    assert raised, "expected ValueError on in_features mismatch"


def test_composition_weights_methods() -> None:
    assert composition.composition_weights(3, "rank_average", 1.0) == [1 / 3, 1 / 3, 1 / 3]
    assert composition.composition_weights(2, "rank_sum", 1.0) == [1.0, 1.0]
    raised = False
    try:
        composition.composition_weights(0, "rank_average", 1.0)
    except ValueError:
        raised = True
    assert raised, "count <= 0 must raise"


def main() -> int:
    tests = [
        test_singleton_values_match,
        test_two_lora_average_forward_matches,
        test_compose_top_k_singleton_is_identity,
        test_compose_top_k_two_uses_uniform_weights,
        test_nested_dict_structure_preserved,
        test_shape_mismatch_raises,
        test_composition_weights_methods,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        else:
            print(f"PASS {fn.__name__}")
    if failed:
        print(f"\n{failed}/{len(tests)} test(s) failed")
        return 1
    print(f"\nall {len(tests)} test(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
