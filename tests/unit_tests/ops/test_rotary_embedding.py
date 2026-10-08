# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for rotary embedding ops.
"""

import os
from unittest.mock import patch

import pytest
import torch


def _make_apply_rotary_emb(*, is_neox_style=True, enable_fp32_compute=False):
    # The CustomOp constructor requires a running vLLM config; these tests
    # exercise its backend methods without model construction.
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    op = ApplyRotaryEmb.__new__(ApplyRotaryEmb)
    torch.nn.Module.__init__(op)
    op.is_neox_style = is_neox_style
    op.enable_fp32_compute = enable_fp32_compute
    return op


@pytest.mark.parametrize("shape", [(5, 2, 80), (2, 5, 2, 80)])
@pytest.mark.parametrize("rotary_dim", [78, 80])
@pytest.mark.parametrize("is_neox_style", [True, False])
def test_apply_rotary_emb_reference_preserves_unrotated_tail(
    shape, rotary_dim, is_neox_style
):
    from vllm_fl.dispatch.backends.reference.impl.rotary import apply_rotary_emb_torch

    op = _make_apply_rotary_emb(is_neox_style=is_neox_style)
    x = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(
        shape
    )
    cos = torch.zeros(shape[-3], rotary_dim // 2)
    sin = torch.ones_like(cos)

    actual = apply_rotary_emb_torch(op, x, cos, sin)
    rotary_input = x[..., :rotary_dim]
    if is_neox_style:
        first, second = rotary_input.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
    else:
        first, second = rotary_input[..., ::2], rotary_input[..., 1::2]
        rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    expected = torch.cat((rotated, x[..., rotary_dim:]), dim=-1)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[..., rotary_dim:], x[..., rotary_dim:])


def test_apply_rotary_emb_patch_routes_through_dispatch(monkeypatch):
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    from vllm_fl.attention.utils import patch_oot_apply_rotary_emb
    from vllm_fl.dispatch import PREFER_REFERENCE, SelectionPolicy, policy_context
    from vllm_fl.ops.rotary_embedding import apply_rotary_emb_dispatch

    monkeypatch.setattr(ApplyRotaryEmb, "forward_oot", ApplyRotaryEmb.forward_oot)
    patch_oot_apply_rotary_emb()
    assert ApplyRotaryEmb.forward_oot is apply_rotary_emb_dispatch

    op = _make_apply_rotary_emb()
    x = torch.arange(80, dtype=torch.float32).reshape(1, 1, 80)
    cos = torch.zeros(1, 39)
    sin = torch.ones_like(cos)
    with policy_context(SelectionPolicy(prefer=PREFER_REFERENCE)):
        actual = op.forward_oot(x, cos, sin)

    expected = torch.cat((-x[..., 39:78], x[..., :39], x[..., 78:]), dim=-1)
    torch.testing.assert_close(actual, expected)


def test_apply_rotary_emb_has_separate_dispatch_backends():
    from vllm_fl.dispatch import PREFER_DEFAULT, PREFER_REFERENCE, PREFER_VENDOR
    from vllm_fl.dispatch.backends.flaggems.register_ops import (
        register_builtins as register_flaggems,
    )
    from vllm_fl.dispatch.backends.reference.register_ops import (
        register_builtins as register_reference,
    )
    from vllm_fl.dispatch.backends.vendor.cuda.register_ops import (
        register_builtins as register_cuda,
    )
    from vllm_fl.dispatch.manager import OpManager
    from vllm_fl.dispatch.policy import SelectionPolicy, policy_context
    from vllm_fl.dispatch.registry import OpRegistry

    registry = OpRegistry()
    with patch(
        "vllm_fl.dispatch.backends.flaggems.register_ops.use_flaggems_op",
        return_value=True,
    ):
        register_flaggems(registry)
    register_reference(registry)
    register_cuda(registry)
    # Availability is bound at registration time. Make each implementation
    # selectable here, regardless of which vendor runs the unit test.
    for impl in registry.get_implementations("apply_rotary_emb"):
        impl.fn._is_available = lambda: True
    manager = OpManager(registry)
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()

    for prefer, expected in (
        (PREFER_DEFAULT, "default.flagos"),
        (PREFER_VENDOR, "vendor.cuda"),
        (PREFER_REFERENCE, "reference.torch"),
    ):
        with policy_context(SelectionPolicy(prefer=prefer)):
            assert manager._resolve_impl("apply_rotary_emb").impl_id == expected

    # The FlagGems rotary_embedding switch keeps its existing meaning for
    # both Q/K RoPE and the single-tensor interface.
    disabled = OpRegistry()
    with patch(
        "vllm_fl.dispatch.backends.flaggems.register_ops.use_flaggems_op",
        side_effect=lambda op_name: op_name != "rotary_embedding",
    ):
        register_flaggems(disabled)
    assert disabled.get_implementation("apply_rotary_emb", "default.flagos") is None


@pytest.mark.gpu
@pytest.mark.parametrize(
    "shape,rotary_dim,is_neox_style,cos_rank,enable_fp32_compute",
    [
        ((5, 2, 80), 78, True, 2, False),
        ((5, 2, 80), 80, False, 2, False),
        ((2, 5, 2, 80), 78, False, 3, False),
        ((2, 5, 2, 80), 80, True, 3, True),
    ],
)
def test_apply_rotary_emb_flaggems_matches_reference(
    shape, rotary_dim, is_neox_style, cos_rank, enable_fp32_compute
):
    pytest.importorskip("flag_gems")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlagGems RoPE")

    from vllm_fl.dispatch.backends.flaggems.impl.rotary import (
        apply_rotary_emb_flaggems,
    )
    from vllm_fl.dispatch.backends.reference.impl.rotary import apply_rotary_emb_torch

    op = _make_apply_rotary_emb(
        is_neox_style=is_neox_style, enable_fp32_compute=enable_fp32_compute
    )
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    original = x.clone()
    angles = torch.randn(shape[-3], rotary_dim // 2, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    if cos_rank == 3:
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

    actual = apply_rotary_emb_flaggems(op, x, cos, sin)
    expected = apply_rotary_emb_torch(op, x, cos, sin)

    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(actual[..., rotary_dim:], original[..., rotary_dim:])
    torch.testing.assert_close(x, original)


class TestRotaryEmbeddingFL:
    """Test RotaryEmbeddingFL class behavior."""

    @pytest.fixture
    def mock_cached_op(self):
        with patch("vllm_fl.ops.rotary_embedding._rotary_embedding") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        with patch(
            "vllm_fl.ops.rotary_embedding.RotaryEmbedding.__init__", return_value=None
        ):
            yield

    def test_forward_oot_dispatches_correctly(self, mock_parent_init, mock_cached_op):
        """Test forward_oot calls dispatch system with correct arguments."""
        from vllm_fl.ops.rotary_embedding import RotaryEmbeddingFL

        layer = RotaryEmbeddingFL(
            head_size=64,
            rotary_dim=32,
            max_position_embeddings=2048,
            base=10000.0,
            is_neox_style=True,
            dtype=torch.float32,
        )

        # Manually set attributes that parent __init__ would set
        layer.head_size = 64
        layer.rotary_dim = 32
        layer.is_neox_style = True
        layer.cos_sin_cache = torch.randn(2048, 64)

        mock_cached_op.return_value = (
            torch.randn(4, 8, 32),
            torch.randn(4, 8, 32),
        )

        positions = torch.tensor([0, 1, 2, 3])
        query = torch.randn(4, 8, 64)
        key = torch.randn(4, 8, 64)

        layer.forward_oot(positions, query, key)

        mock_cached_op.assert_called_once()
        call_args = mock_cached_op.call_args
        assert call_args[0][0] is layer
