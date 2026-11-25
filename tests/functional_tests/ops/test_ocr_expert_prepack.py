# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NPU integration coverage for the load/reload contract of TP1 OCR experts."""

from types import SimpleNamespace

import pytest
import torch


def test_ocr_expert_prepack_reload_and_restore(monkeypatch):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe import (
        prepack_ocr_expert_weights,
    )

    def make_weight(shape):
        return torch.nn.Parameter(
            torch.zeros(shape, device="npu", dtype=torch.bfloat16), requires_grad=False
        )

    def read_gmm(weight):
        gmm = weight.transpose(1, 2)
        x = torch.zeros(1, gmm.shape[1], device=gmm.device, dtype=gmm.dtype)
        x[0, 3] = 1
        groups = torch.zeros(gmm.shape[0], device=gmm.device, dtype=torch.int64)
        groups[0] = 1
        output = torch_npu.npu_grouped_matmul(
            x=[x],
            weight=[gmm],
            group_list=groups,
            split_item=2,
            group_list_type=1,
            group_type=0,
        )[0]
        return output[0, 5].item()

    layer = SimpleNamespace(
        w13_weight=make_weight((64, 1792, 1280)),
        w2_weight=make_weight((64, 1280, 896)),
    )
    weights = (layer.w13_weight, layer.w2_weight)
    source_ptrs = [w.data_ptr() for w in weights]
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    cast = torch_npu.npu_format_cast

    def fail_second(view, fmt):
        if view.shape[-1] == 1280:
            raise RuntimeError("second cast failed")
        return cast(view, fmt)

    with monkeypatch.context() as patch:
        patch.setattr(torch_npu, "npu_format_cast", fail_second)
        with pytest.raises(RuntimeError, match="second cast failed"):
            prepack_ocr_expert_weights(layer)
    assert torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == previous
    assert [w.data_ptr() for w in weights] == source_ptrs
    assert not getattr(layer.w13_weight, "_fl_ocr_prepacked", False)

    for weight in weights:
        weight[0, 5, 3] = 2
    prepack_ocr_expert_weights(layer)
    assert layer.w13_weight._fl_ocr_prepacked
    packed_ptrs = [w.data_ptr() for w in weights]
    for weight in weights:
        gmm = weight.transpose(1, 2)
        assert gmm.is_contiguous() and torch_npu.get_npu_format(gmm) == 29
        assert read_gmm(weight) == 2
        weight[0, 5, 3] = 7
    prepack_ocr_expert_weights(layer)
    assert [w.data_ptr() for w in weights] == packed_ptrs
    assert all(read_gmm(w) == 7 for w in weights)
    assert torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT") == previous


@pytest.mark.parametrize("prepacked", [False, True])
def test_local_route_reuse_scope(monkeypatch, prepacked):
    native = pytest.importorskip("torch_npu")
    from vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe import (
        _torch_fused_experts_impl,
    )

    # Isolate routing wire-up: the native producer owns the permutation.
    ids = torch.tensor([[1], [0]], dtype=torch.int32)
    permutation = torch.tensor([1, 0], dtype=torch.int32)
    hidden = torch.zeros((2, 8))
    weight = torch.zeros((2, 8, 8))
    if prepacked:
        weight._fl_ocr_prepacked = True
    monkeypatch.setattr(
        native,
        "npu_moe_init_routing_v2",
        lambda *a, **k: (hidden, permutation, torch.ones(2, dtype=torch.int64), None),
    )
    monkeypatch.setattr(native, "npu_grouped_matmul", lambda **k: [hidden])
    monkeypatch.setattr(native, "npu_swiglu", lambda t: t)

    def unpermute(*, permuted_tokens, sorted_indices, probs):
        assert (sorted_indices is permutation) == prepacked
        torch.testing.assert_close(sorted_indices, permutation)
        return permuted_tokens

    monkeypatch.setattr(native, "npu_moe_token_unpermute", unpermute)
    _torch_fused_experts_impl(hidden, weight, weight, torch.ones(2, 1), ids)
