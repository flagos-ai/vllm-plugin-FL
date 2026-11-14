# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contracts for Sunrise's v0.32 padded softmax router."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def router(monkeypatch):
    # Execute the actual functions without importing accelerator-only modules.
    path = Path(__file__).resolve().parents[3] / "vllm_fl/ops/fused_moe/router.py"
    tree = ast.parse(path.read_text())
    names = {"fused_topk", "_is_sunrise_ptpu_tensor"}
    source = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    state = SimpleNamespace(mask=None, sunrise=True, native_kernel=True)
    upstream = Mock(return_value="upstream")

    def kernel(weights, ids, source_rows, logits, renormalize):
        scores = logits.float().softmax(-1)
        values, indices = scores.topk(weights.shape[-1], dim=-1)
        if renormalize:
            values = values / values.sum(-1, keepdim=True)
        weights.copy_(values)
        ids.copy_(indices)
        rows = torch.arange(logits.shape[0])[:, None]
        columns = torch.arange(weights.shape[-1])[None, :]
        source_rows.copy_(columns * logits.shape[0] + rows)
        return weights, ids

    kernel_mock = Mock(side_effect=kernel)
    scope = dict(
        torch=torch,
        _get_padding_mask=lambda count: state.mask,
        upstream_fused_topk=upstream,
        _topk_softmax=kernel_mock,
    )
    exec(compile(source, str(path), "exec"), scope)
    scope["_is_sunrise_ptpu_tensor"] = lambda value: state.sunrise
    scope["_has_native_moe_topk"] = lambda: state.native_kernel
    return SimpleNamespace(
        call=scope["fused_topk"], state=state, upstream=upstream, kernel=kernel_mock
    )


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("indices_type", [None, torch.int32, torch.int64])
@pytest.mark.parametrize("padding", [None, [False, True, False], [True, True, True]])
def test_sunrise_padding_preserves_scores_and_source_rows(
    router, renormalize, indices_type, padding
):
    logits = torch.tensor(
        [[-1.0, 2.0, 0.5, -3.0], [3.0, 1.0, -2.0, 0.0], [0.0, -2.0, 1.0, 4.0]]
    )
    if padding is not None:
        router.state.mask = torch.tensor(padding)
    weights, ids, source_rows = router.call(
        torch.zeros(3, 8), logits, 2, renormalize, indices_type
    )
    scores = logits.softmax(-1)
    expected_weights, expected_ids = scores.topk(2, dim=-1)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    if padding is not None:
        expected_ids[router.state.mask] = -1
    torch.testing.assert_close(weights, expected_weights)
    assert ids.dtype == (torch.int32 if indices_type is None else indices_type)
    torch.testing.assert_close(ids.long(), expected_ids)
    torch.testing.assert_close(
        source_rows, torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32)
    )
    router.upstream.assert_not_called()
    router.kernel.assert_called_once()


def test_other_device_keeps_upstream_padding_route(router):
    router.state.sunrise = False
    router.state.mask = torch.tensor([False, True])
    assert router.call(torch.zeros(2, 4), torch.zeros(2, 8), 2, True) == "upstream"
    router.upstream.assert_called_once()
    router.kernel.assert_not_called()


def test_sigmoid_keeps_existing_upstream_contract(router):
    router.state.mask = torch.tensor([False, True])
    assert (
        router.call(
            torch.zeros(2, 4), torch.zeros(2, 8), 2, True, scoring_func="sigmoid"
        )
        == "upstream"
    )
    assert router.upstream.call_args.kwargs["scoring_func"] == "sigmoid"
    router.kernel.assert_not_called()
