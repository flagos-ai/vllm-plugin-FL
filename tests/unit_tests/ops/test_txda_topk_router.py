# Copyright (c) 2026 BAAI. All rights reserved.
"""TXDA empty-build routing against an independent scalar probability oracle."""

import __future__

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/ops/fused_moe/router.py"


@pytest.fixture
def router():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {"_is_txda_tensor", "_txda_native_fused_topk", "fused_topk"}
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in names
        or isinstance(node, ast.ClassDef)
        and node.name == "FusedTopKRouterFL"
    ]
    padding = Mock(return_value=None)
    vendor = Mock(side_effect=AssertionError("vendor routing must not run on TXDA"))
    upstream = Mock(side_effect=AssertionError("empty build has no _moe_C router"))
    namespace = {
        "torch": torch,
        "_get_padding_mask": padding,
        "_topk_softmax": vendor,
        "upstream_fused_topk": upstream,
        "FusedTopKRouter": object,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(SOURCE),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return SimpleNamespace(
        native=namespace["_txda_native_fused_topk"],
        fused=namespace["fused_topk"],
        cls=namespace["FusedTopKRouterFL"],
        namespace=namespace,
        padding=padding,
        vendor=vendor,
        upstream=upstream,
    )


def oracle(logits, topk, scoring, renormalize):
    rows = logits.float().tolist()
    all_weights, all_ids = [], []
    for row in rows:
        if scoring == "softmax":
            maximum = max(row)
            numerators = [math.exp(value - maximum) for value in row]
            denominator = sum(numerators)
            scores = [value / denominator for value in numerators]
        else:
            scores = [1 / (1 + math.exp(-value)) for value in row]
        ids = sorted(range(len(row)), key=lambda index: (-scores[index], index))[:topk]
        weights = [scores[index] for index in ids]
        if renormalize:
            denominator = sum(weights) or 1
            weights = [value / denominator for value in weights]
        all_weights.append(weights)
        all_ids.append(ids)
    return torch.tensor(all_weights, dtype=torch.float32), torch.tensor(all_ids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("experts,topk", [(8, 1), (256, 8), (257, 8)])
def test_probabilities_ids_and_source_rows_match_scalar_oracle(
    router, dtype, scoring, renormalize, experts, topk
):
    generator = torch.Generator().manual_seed(783)
    logits = torch.randn(3, experts, generator=generator, dtype=torch.float32).to(dtype)
    hidden = torch.empty(3, 4, dtype=torch.bfloat16)
    weights, ids, source = router.native(
        hidden, logits, topk, renormalize, None, scoring
    )
    expected_weights, expected_ids = oracle(logits, topk, scoring, renormalize)
    torch.testing.assert_close(weights, expected_weights, rtol=2e-6, atol=3e-8)
    torch.testing.assert_close(ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    assert weights.dtype == torch.float32
    assert ids.dtype == source.dtype == torch.int32
    assert weights.device == ids.device == source.device == hidden.device
    assert source.tolist() == [
        [rank * 3 + row for rank in range(topk)] for row in range(3)
    ]


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("indices_type", [None, torch.int32, torch.int64])
@pytest.mark.parametrize("padding", [[False] * 3, [True, False, True], [True] * 3])
def test_padding_masks_only_expert_ids_and_keeps_weights_and_source_map(
    router, scoring, indices_type, padding
):
    hidden = torch.empty(3, 4)
    logits = torch.tensor([[0.4, 0.2, -0.7, 1.8]]).expand(3, -1)
    unmasked = router.native(hidden, logits, 2, True, indices_type, scoring)
    # Mimic V2's always-present mask, including its all-false form that
    # previously sent TXDA empty builds into the missing _moe_C operator.
    router.padding.return_value = torch.tensor(padding)
    router.namespace["_is_txda_tensor"] = lambda _: True
    weights, ids, source = router.fused(
        hidden, logits, 2, True, indices_type=indices_type, scoring_func=scoring
    )
    torch.testing.assert_close(weights, unmasked[0], rtol=0, atol=0)
    torch.testing.assert_close(source, unmasked[2], rtol=0, atol=0)
    expected_ids = unmasked[1].clone()
    for row, is_padding in enumerate(padding):
        if is_padding:
            expected_ids[row].fill_(-1)
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    router.upstream.assert_not_called()
    router.vendor.assert_not_called()


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
def test_tied_scores_choose_distinct_lower_ids_first(router, scoring):
    weights, ids, _ = router.native(
        torch.empty(2, 4), torch.zeros(2, 256), 8, True, torch.int64, scoring
    )
    assert ids.tolist() == [list(range(8))] * 2
    torch.testing.assert_close(weights, torch.full((2, 8), 1 / 8), rtol=0, atol=0)


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [False, True])
def test_nonfinite_scores_and_zero_denominator_follow_upstream_contract(
    router, scoring, renormalize
):
    logits = torch.tensor([[float("nan")] * 4, [float("-inf")] * 4])
    weights, ids, _ = router.native(
        torch.empty(2, 3), logits, 4, renormalize, None, scoring
    )
    assert bool(torch.isfinite(weights).all())
    assert weights.tolist() == [[0.0] * 4] * 2
    assert ids.tolist() == [list(range(4))] * 2


def test_sigmoid_infinities_are_finite_probabilities(router):
    weights, ids, _ = router.native(
        torch.empty(1, 3),
        torch.tensor([[float("inf"), float("-inf"), float("nan"), 0.0]]),
        4,
        False,
        None,
        "sigmoid",
    )
    assert weights.tolist() == [[1.0, 0.5, 0.0, 0.0]]
    assert ids.tolist() == [[0, 3, 1, 2]]


@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
def test_empty_batch_keeps_all_three_typed_outputs(router, scoring):
    router.padding.return_value = torch.empty(0, dtype=torch.bool)
    weights, ids, source = router.native(
        torch.empty(0, 5), torch.empty(0, 256), 8, True, torch.int64, scoring
    )
    assert weights.shape == ids.shape == source.shape == (0, 8)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int64
    assert source.dtype == torch.int32


def test_router_method_forwards_routing_options_and_discards_source_rows(router):
    native = Mock(return_value=(object(), object(), object()))
    router.namespace["_is_txda_tensor"] = lambda _: True
    router.namespace["_txda_native_fused_topk"] = native
    instance = router.cls()
    instance.top_k, instance.renormalize, instance.scoring_func = 8, False, "sigmoid"
    hidden, logits = torch.empty(2, 3), torch.empty(2, 256)
    result = instance._compute_routing(hidden, logits, torch.int64, input_ids=object())
    native.assert_called_once_with(hidden, logits, 8, False, torch.int64, "sigmoid")
    assert result == native.return_value[:2]


@pytest.mark.parametrize("scoring,padded", [("softmax", True), ("sigmoid", False)])
def test_other_devices_keep_upstream_masked_or_sigmoid_routing(router, scoring, padded):
    hidden, logits = torch.empty(2, 3), torch.empty(2, 4)
    result = object()
    router.upstream.side_effect = None
    router.upstream.return_value = result
    if padded:
        router.padding.return_value = torch.zeros(2, dtype=torch.bool)
    assert router.fused(hidden, logits, 2, True, torch.int64, scoring) is result
    router.upstream.assert_called_once_with(
        hidden_states=hidden,
        gating_output=logits,
        topk=2,
        renormalize=True,
        indices_type=torch.int64,
        scoring_func=scoring,
    )
    router.vendor.assert_not_called()


def test_other_devices_keep_unpadded_softmax_provider(router):
    hidden, logits = torch.empty(2, 3), torch.empty(2, 4)
    router.vendor.side_effect = lambda weights, ids, *_: (weights, ids)
    weights, ids, source = router.fused(hidden, logits, 2, True)
    assert weights.shape == ids.shape == source.shape == (2, 2)
    assert router.vendor.call_count == 1
    router.upstream.assert_not_called()


def test_invalid_scoring_and_mismatched_batches_fail_before_provider_calls(router):
    hidden, logits = torch.empty(2, 3), torch.empty(2, 4)
    router.namespace["_is_txda_tensor"] = lambda _: True
    with pytest.raises(ValueError, match="Unsupported scoring function"):
        router.fused(hidden, logits, 2, True, scoring_func="sqrtsoftplus")
    with pytest.raises(AssertionError, match="Number of tokens mismatch"):
        router.fused(hidden, logits[:1], 2, True)
    router.vendor.assert_not_called()
    router.upstream.assert_not_called()
