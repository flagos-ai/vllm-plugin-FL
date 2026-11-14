# SPDX-License-Identifier: Apache-2.0
"""Numerical routing contracts executed without importing accelerator runtimes."""

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASCEND_MOE_MODULE = "vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe"


def _load_fused_topk(source):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fused_topk"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace


@pytest.fixture
def router_functions():
    return _load_fused_topk(_REPO_ROOT / "vllm_fl/ops/fused_moe/router.py")


@pytest.fixture
def ascend_fused_topk(monkeypatch):
    source = _REPO_ROOT / (_ASCEND_MOE_MODULE.replace(".", "/") + ".py")
    module = ModuleType(_ASCEND_MOE_MODULE)
    module.fused_topk = _load_fused_topk(source)["fused_topk"]
    monkeypatch.setitem(sys.modules, _ASCEND_MOE_MODULE, module)
    return module.fused_topk


@pytest.mark.usefixtures("ascend_fused_topk")
@pytest.mark.parametrize("scoring", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("index_dtype", [None, torch.int64])
@pytest.mark.parametrize(
    "padding", [None, [False, False, False], [False, True, False], [True, True, True]]
)
def test_npu_routing_preserves_scores_and_drops_padding(
    router_functions, scoring, renormalize, input_dtype, index_dtype, padding
):
    logits = torch.tensor(
        [[0, 2, -2, 1], [4, -1, 3, -2], [-3, -1, 2, 3]], dtype=input_dtype
    )
    mask = None if padding is None else torch.tensor(padding)
    router_functions["_get_padding_mask"] = lambda size: mask
    router_functions["_is_ascend_npu_tensor"] = lambda value: True

    def unavailable(*args, **kwargs):
        raise AssertionError("NPU router called a CUDA operator")

    router_functions["upstream_fused_topk"] = unavailable
    router_functions["_topk_softmax"] = unavailable
    weights, ids, scratch = router_functions["fused_topk"](
        torch.zeros(3, 2),
        logits,
        2,
        renormalize,
        indices_type=index_dtype,
        scoring_func=scoring,
    )
    expected_ids = torch.tensor([[1, 3], [0, 2], [3, 2]])
    if scoring == "softmax":
        scores = (logits.float() - logits.float().max(-1, keepdim=True).values).exp()
        scores /= scores.sum(-1, keepdim=True)
    else:
        scores = 1 / (1 + (-logits.float()).exp())
    expected = scores.gather(1, expected_ids)
    if renormalize:
        expected /= expected.sum(-1, keepdim=True)
    if mask is not None:
        expected[mask] = 0
        expected_ids[mask] = -1
    assert ids.dtype == (torch.int32 if index_dtype is None else index_dtype)
    assert weights.dtype == torch.float32
    torch.testing.assert_close(ids.long(), expected_ids)
    torch.testing.assert_close(weights, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        scratch, torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int32)
    )


def test_other_platform_keeps_upstream_padded_dispatch(router_functions):
    mask = torch.tensor([True])
    router_functions["_get_padding_mask"] = lambda size: mask
    router_functions["_is_ascend_npu_tensor"] = lambda value: False
    router_functions["_has_native_moe_topk"] = lambda: True
    outputs = (object(), object(), object())
    calls = []

    def upstream(**kwargs):
        calls.append(kwargs)
        return outputs

    router_functions["upstream_fused_topk"] = upstream
    result = router_functions["fused_topk"](
        torch.zeros(1, 2), torch.zeros(1, 4), 2, True
    )
    assert result is outputs
    assert calls[0]["scoring_func"] == "softmax"


def test_sigmoid_underflow_does_not_produce_nan(ascend_fused_topk):
    weights, _, _ = ascend_fused_topk(
        torch.full((2, 4), -1000.0),
        2,
        True,
        None,
        "sigmoid",
        torch.tensor([False, True]),
    )
    assert torch.equal(weights, torch.zeros_like(weights))
