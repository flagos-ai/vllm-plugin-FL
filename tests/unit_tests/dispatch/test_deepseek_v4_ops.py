# Copyright (c) 2026 BAAI. All rights reserved.

"""Tests for DeepSeek-V4 operator dispatch."""

from unittest.mock import Mock

import torch

from vllm_fl.dispatch.backends.reference.impl.deepseek_v4 import (
    deepseek_v4_inv_rope_quant_int8_torch,
)
from vllm_fl.dispatch.types import BackendImplKind
from vllm_fl.ops import deepseek_v4_int8_woa


def test_reference_inv_rope_quant_int8():
    o = torch.tensor(
        [[[1, 2, 3, 4], [-1, -2, 5, 6]]],
        dtype=torch.bfloat16,
    )
    positions = torch.tensor([0], dtype=torch.int32)
    cos_sin_cache = torch.tensor([[0, 1]], dtype=torch.float32)

    quantized, scales = deepseek_v4_inv_rope_quant_int8_torch(
        o,
        positions,
        cos_sin_cache,
        n_groups=1,
        heads_per_group=2,
        nope_dim=2,
        rope_dim=2,
    )

    expected = torch.tensor(
        [[[21, 42, 85, -64, -21, -42, 127, -106]]],
        dtype=torch.int8,
    )
    assert torch.equal(quantized, expected)
    torch.testing.assert_close(
        scales,
        torch.tensor([[[6 / 127]]], dtype=torch.float32),
    )


def test_frontend_dispatches_through_cached_op(monkeypatch):
    expected = (Mock(), Mock())
    dispatch = Mock(return_value=expected)
    monkeypatch.setattr(
        deepseek_v4_int8_woa,
        "_dispatch_inv_rope_quant_int8",
        dispatch,
    )
    args = (
        Mock(),
        Mock(),
        Mock(),
        2,
        4,
        64,
        64,
    )

    actual = deepseek_v4_int8_woa.fused_inv_rope_quant_int8(*args)

    assert actual is expected
    dispatch.assert_called_once_with(*args)


def test_all_backends_register_deepseek_v4_op(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops as flaggems_ops
    from vllm_fl.dispatch.backends.reference import register_ops as reference_ops
    from vllm_fl.dispatch.backends.vendor.cuda import register_ops as cuda_ops

    registered = []

    class Registry:
        def register_many(self, impls):
            registered.extend(impls)

    monkeypatch.setattr(
        flaggems_ops,
        "use_flaggems_op",
        lambda op_name: op_name == deepseek_v4_int8_woa.DSV4_INV_ROPE_QUANT_INT8_OP,
    )
    registry = Registry()
    flaggems_ops.register_builtins(registry)
    cuda_ops.register_builtins(registry)
    reference_ops.register_builtins(registry)

    implementations = [
        impl
        for impl in registered
        if impl.op_name == deepseek_v4_int8_woa.DSV4_INV_ROPE_QUANT_INT8_OP
    ]
    assert {impl.impl_id for impl in implementations} == {
        "default.flagos",
        "vendor.cuda",
        "reference.torch",
    }
    assert {impl.kind for impl in implementations} == {
        BackendImplKind.DEFAULT,
        BackendImplKind.VENDOR,
        BackendImplKind.REFERENCE,
    }
