# Copyright (c) 2026 BAAI. All rights reserved.
"""Extra coverage for the reference backend left by w9b."""

import builtins
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.reference import reference as ref
from vllm_fl.dispatch.backends.reference import register_ops as ref_register
from vllm_fl.dispatch.backends.reference.impl import (
    activation as ref_activation,
    fused_moe as ref_fused_moe,
    normalization as ref_norm,
    rotary as ref_rotary,
)


class _Registry:
    def __init__(self):
        self.impls = []

    def register_many(self, impls):
        self.impls.extend(impls)


def test_register_builtins_skips_absent_optional_ops(monkeypatch):
    class PartialBackend:
        _available = True

        def is_available(self):
            return True

        def silu_and_mul(self, obj, x):
            return x

    monkeypatch.setattr(ref, "ReferenceBackend", PartialBackend)
    registry = _Registry()
    ref_register.register_builtins(registry)
    names = {impl.op_name for impl in registry.impls}
    assert "silu_and_mul" in names
    assert "w8a8_moe_experts" in names
    assert "rms_norm" not in names
    assert "topk_softmax" not in names


def test_is_available_import_failure_returns_false(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(ref.ReferenceBackend, "_available", None)
    assert ref.ReferenceBackend().is_available() is False


def test_moe_align_drops_out_of_range_ids_exactly():
    backend = ref.ReferenceBackend()
    ids = torch.tensor([[0, -1], [3, 9]], dtype=torch.int32)
    _, expert_ids, ntpp = backend.moe_align_block_size(ids, 2, 4)
    # Only ids 0 and 3 survive: one block of 2 tokens per expert == 4 tokens.
    assert ntpp.item() == 4
    assert set(expert_ids.tolist()) <= {0, 3, -1}


def test_moe_align_mapped_ids_are_screened_when_bucketing():
    backend = ref.ReferenceBackend()
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    expert_map = torch.tensor([-1, 0, 1, 2], dtype=torch.int32)
    sorted_ids, expert_ids, ntpp = backend.moe_align_block_size(
        ids, 4, 4, expert_map=expert_map, ignore_invalid_experts=True
    )
    # Expert 0 maps to -1 and must be dropped; three local experts remain.
    assert ntpp.item() == 12
    assert sorted_ids.max().item() == ids.numel()
    assert set(expert_ids.tolist()) <= {0, 1, 2, -1}


def test_moe_align_applies_expert_map_after_bucketing():
    backend = ref.ReferenceBackend()
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    expert_map = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    _, expert_ids, _ = backend.moe_align_block_size(
        ids, 4, 4, expert_map=expert_map, ignore_invalid_experts=False
    )
    live = expert_ids[expert_ids >= 0]
    assert live.numel() > 0
    assert live.min().item() >= 10
    assert live.max().item() <= 13


def test_reference_topk_softmax_without_renormalize():
    gating = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]])
    weights = torch.zeros((2, 2))
    indices = torch.zeros((2, 2), dtype=torch.int32)
    token_indices = torch.zeros((2, 2), dtype=torch.int32)
    ref_fused_moe.topk_softmax(
        weights, indices, token_indices, gating, renormalize=False
    )
    expected = torch.softmax(gating, dim=-1).topk(2, dim=-1).values
    torch.testing.assert_close(weights, expected)
    # Slot-major layout: token index advances within an expert slot.
    assert token_indices.tolist() == [[0, 2], [1, 3]]


def test_gelu_and_mul_defaults_to_none_approximation():
    x = torch.randn(2, 8)
    out = ref_activation.gelu_and_mul_torch(SimpleNamespace(), x)
    expected = torch.nn.functional.gelu(x[..., :4]) * x[..., 4:]
    torch.testing.assert_close(out, expected)


def test_rms_norm_variance_override_and_weightless():
    class Obj:
        def __init__(self):
            self.weight = torch.ones(4)
            self.variance_epsilon = 1e-6
            self.variance_size_override = 2
            self.pass_weight = False

    x = torch.randn(2, 4)
    out = ref_norm.rms_norm_torch(Obj(), x)
    var = x[..., :2].square().mean(dim=-1, keepdim=True)
    expected = x * torch.rsqrt(var + 1e-6)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_rotary_rejects_mismatched_sin_shape():
    q = torch.randn(3, 2, 4)
    with pytest.raises(ValueError, match="same-shaped 2D caches"):
        ref_rotary.rotary_embedding_torch(
            None,
            q,
            q,
            torch.randn(3, 4),
            torch.randn(2, 4),
            torch.arange(3),
        )


def test_rotary_interleaved_full_width_matches_manual_math():
    q = torch.randn(2, 2, 4)
    cos = torch.randn(2, 4)
    sin = torch.randn(2, 4)
    pos = torch.arange(2)
    qe, ke = ref_rotary.rotary_embedding_torch(
        None, q, q, cos, sin, pos, rotary_interleaved=True
    )
    x1 = q[..., ::2]
    x2 = q[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    expected = q * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
    torch.testing.assert_close(qe, expected)
    torch.testing.assert_close(ke, expected)


def test_rotary_4d_half_width_interleaved():
    q = torch.randn(1, 2, 3, 4)
    cos = torch.randn(3, 2)
    sin = torch.randn(3, 2)
    qe, ke = ref_rotary.rotary_embedding_torch(
        None,
        q,
        q,
        cos,
        sin,
        torch.arange(3),
        rotary_interleaved=True,
    )
    assert qe.shape == q.shape and ke.shape == q.shape
