# Coverage tests: vllm_fl/ops/fla/utils.py, chunk.py, fused_recurrent.py.

import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_fl.ops.fla.chunk as chunk_mod
import vllm_fl.ops.fla.fused_recurrent as fr_mod
import vllm_fl.ops.fla.utils as fla_utils


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _fake_torch(monkeypatch):
    fake = SimpleNamespace(
        Tensor=torch.Tensor,
        cuda=SimpleNamespace(device=lambda index: _Ctx()),
    )
    monkeypatch.setattr(fla_utils, "torch", fake)
    return fake


def test_input_guard_makes_tensors_contiguous(monkeypatch):
    _fake_torch(monkeypatch)
    seen = {}

    @fla_utils.input_guard
    def fn(x, y=None):
        seen["x_contiguous"] = x.is_contiguous()
        seen["y_contiguous"] = y.is_contiguous()
        return x + y

    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).t()
    y = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    assert not x.is_contiguous()

    result = fn(x, y=y)
    assert seen["x_contiguous"] is True
    assert seen["y_contiguous"] is True
    assert result.shape == x.shape


def test_input_guard_handles_keyword_only_tensor(monkeypatch):
    _fake_torch(monkeypatch)
    seen = {}

    @fla_utils.input_guard
    def fn(*, x):
        seen["contiguous"] = x.is_contiguous()
        return x

    x = torch.arange(8, dtype=torch.float32).reshape(2, 4).t()
    fn(x=x)
    assert seen["contiguous"] is True


def test_input_guard_without_tensor_uses_nullcontext(monkeypatch):
    fake = _fake_torch(monkeypatch)

    @fla_utils.input_guard
    def fn(value):
        return value * 2

    assert fn(3) == 6


def _chunk_context():
    return SimpleNamespace(scale=None, use_qk_l2norm_in_kernel=None)


def _chunk_args(dtype=torch.bfloat16, batch=1, seq=2, heads=2, kdim=4):
    q = torch.randn(batch, seq, heads, kdim, dtype=dtype)
    k = torch.randn(batch, seq, heads, kdim, dtype=dtype)
    v = torch.randn(batch, seq, heads, kdim, dtype=dtype)
    g = torch.randn(batch, seq, heads, dtype=dtype)
    beta = torch.randn(batch, seq, heads, dtype=dtype)
    return q, k, v, g, beta


def test_chunk_function_forward_records_ctx(monkeypatch):
    _fake_torch(monkeypatch)
    captured = {}
    q, k, v, g, beta = _chunk_args()

    def fake_forward(**kwargs):
        captured.update(kwargs)
        return kwargs["g"], kwargs["v"], "A", "final", "w", "h", "v_new"

    monkeypatch.setattr(
        chunk_mod,
        "_chunk_impl",
        SimpleNamespace(chunk_gated_delta_rule_fwd=fake_forward),
    )
    ctx = _chunk_context()
    result, final_state = chunk_mod.ChunkGatedDeltaRuleFunction.forward(
        ctx, q, k, v, g, beta, 0.5, None, False, None, False
    )

    assert result.dtype == q.dtype
    assert result.shape == v.shape
    assert final_state == "final"
    assert ctx.scale == 0.5
    assert ctx.use_qk_l2norm_in_kernel is False
    assert captured["scale"] == 0.5


def test_chunk_function_forward_applies_l2norm(monkeypatch):
    _fake_torch(monkeypatch)
    calls = []
    q, k, v, g, beta = _chunk_args()

    monkeypatch.setattr(
        chunk_mod, "l2norm_fwd", lambda tensor: calls.append(tensor) or tensor
    )
    monkeypatch.setattr(
        chunk_mod,
        "_chunk_impl",
        SimpleNamespace(
            chunk_gated_delta_rule_fwd=lambda **kwargs: (
                kwargs["g"],
                kwargs["v"],
                None,
                None,
                None,
                None,
                None,
            )
        ),
    )
    chunk_mod.ChunkGatedDeltaRuleFunction.forward(
        _chunk_context(), q, k, v, g, beta, 1.0, None, True, None, True
    )
    assert len(calls) == 2


def _patch_chunk_apply(monkeypatch, captured):
    def apply(*args):
        captured["args"] = args
        return torch.zeros(1, 2, 2, 4), "final"

    monkeypatch.setattr(
        chunk_mod, "ChunkGatedDeltaRuleFunction", SimpleNamespace(apply=apply)
    )


def test_chunk_op_forward_native_defaults_scale(monkeypatch):
    captured = {}
    _patch_chunk_apply(monkeypatch, captured)
    op = object.__new__(chunk_mod.ChunkGatedDeltaRuleOp)
    op.output_final_state = False
    op.use_qk_l2norm_in_kernel = False
    q, k, v, g, beta = _chunk_args()

    output, final = chunk_mod.ChunkGatedDeltaRuleOp.forward_native(
        op, q, k, v, g, beta
    )
    assert output.shape == (1, 2, 2, 4)
    assert final == "final"
    assert captured["args"][5] == k.shape[-1] ** -0.5


def test_chunk_op_forward_native_explicit_scale(monkeypatch):
    captured = {}
    _patch_chunk_apply(monkeypatch, captured)
    op = object.__new__(chunk_mod.ChunkGatedDeltaRuleOp)
    op.output_final_state = True
    op.use_qk_l2norm_in_kernel = True
    q, k, v, g, beta = _chunk_args()
    chunk_mod.ChunkGatedDeltaRuleOp.forward_native(op, q, k, v, g, beta, scale=0.25)
    assert captured["args"][5] == 0.25
    assert captured["args"][7] is True


def test_chunk_op_forward_native_validations(monkeypatch):
    captured = {}
    _patch_chunk_apply(monkeypatch, captured)
    op = object.__new__(chunk_mod.ChunkGatedDeltaRuleOp)
    op.output_final_state = False
    op.use_qk_l2norm_in_kernel = False

    q, k, v, g, beta = _chunk_args(dtype=torch.float32)
    with pytest.raises(AssertionError):
        chunk_mod.ChunkGatedDeltaRuleOp.forward_native(op, q, k, v, g, beta)

    q, k, v, g, beta = _chunk_args()
    with pytest.raises(AssertionError):
        chunk_mod.ChunkGatedDeltaRuleOp.forward_native(
            op, q, k, v, g, beta.reshape(1, 2, 2, 1)
        )

    q, k, v, g, beta = _chunk_args(batch=2, heads=1)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        chunk_mod.ChunkGatedDeltaRuleOp.forward_native(
            op, q, k, v, g, beta, cu_seqlens=torch.tensor([0, 2])
        )

    q, k, v, g, beta = _chunk_args(batch=1, heads=1)
    with pytest.raises(ValueError, match="number of initial states"):
        chunk_mod.ChunkGatedDeltaRuleOp.forward_native(
            op,
            q,
            k,
            v,
            g,
            beta,
            initial_state=torch.zeros(2, 1, 4, 4),
            cu_seqlens=torch.tensor([0, 2]),
        )


def test_chunk_op_forward_native_warns_on_head_first_layout(monkeypatch):
    captured = {}
    _patch_chunk_apply(monkeypatch, captured)
    op = object.__new__(chunk_mod.ChunkGatedDeltaRuleOp)
    op.output_final_state = False
    op.use_qk_l2norm_in_kernel = False
    q, k, v, g, beta = _chunk_args(seq=2, heads=4)

    with pytest.warns(UserWarning):
        chunk_mod.ChunkGatedDeltaRuleOp.forward_native(op, q, k, v, g, beta)


def test_fused_recurrent_forward_contiguous(monkeypatch):
    captured = {}
    q, k, v, g, beta = _chunk_args()

    def fake(**kwargs):
        captured.update(kwargs)
        return kwargs["v"], "final"

    monkeypatch.setattr(fr_mod, "fused_recurrent_gated_delta_rule_fwd", fake)
    ctx = SimpleNamespace()
    o, final = fr_mod.FusedRecurrentFunction.forward(
        ctx, q, k, v, g, beta, 0.5, None, True, None, None, None, False
    )
    assert o.shape == v.shape
    assert final == "final"
    assert captured["g"].is_contiguous()
    assert captured["beta"].is_contiguous()
    assert captured["inplace_final_state"] is True


def test_fused_recurrent_op_forward_native(monkeypatch):
    captured = {}

    def apply(*args):
        captured["args"] = args
        return torch.zeros(1, 2, 2, 4), "final"

    monkeypatch.setattr(
        fr_mod, "FusedRecurrentFunction", SimpleNamespace(apply=apply)
    )
    op = object.__new__(fr_mod.FusedRecurrentGatedDeltaRuleOp)
    op.inplace_final_state = False
    op.use_qk_l2norm_in_kernel = True
    q, k, v, g, beta = _chunk_args()

    output, final = fr_mod.FusedRecurrentGatedDeltaRuleOp.forward_native(
        op, q, k, v, g
    )
    assert output.shape == (1, 2, 2, 4)
    assert final == "final"
    args = captured["args"]
    assert torch.equal(args[4], torch.ones_like(q[..., 0]))
    assert args[5] == k.shape[-1] ** -0.5
    assert args[7] is False
    assert args[11] is True


def test_fused_recurrent_op_forward_native_scale_and_validation(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        fr_mod,
        "FusedRecurrentFunction",
        SimpleNamespace(
            apply=lambda *args: (captured.__setitem__("args", args) or (torch.zeros(1, 2, 2, 4), None))
        ),
    )
    op = object.__new__(fr_mod.FusedRecurrentGatedDeltaRuleOp)
    op.inplace_final_state = True
    op.use_qk_l2norm_in_kernel = False
    q, k, v, g, beta = _chunk_args()

    fr_mod.FusedRecurrentGatedDeltaRuleOp.forward_native(
        op, q, k, v, g, beta, scale=0.25
    )
    assert captured["args"][5] == 0.25

    with pytest.raises(AssertionError, match="scale must be positive"):
        fr_mod.FusedRecurrentGatedDeltaRuleOp.forward_native(
            op, q, k, v, g, beta, scale=-1.0
        )

    q2, k2, v2, g2, beta2 = _chunk_args(batch=2, heads=1)
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        fr_mod.FusedRecurrentGatedDeltaRuleOp.forward_native(
            op, q2, k2, v2, g2, beta2, cu_seqlens=torch.tensor([0, 2])
        )
