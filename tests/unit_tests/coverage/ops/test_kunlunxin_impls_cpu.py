# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin core operator implementations.

Covers ``impl/activation.py``, ``impl/normalization.py``, ``impl/rotary.py``
and ``impl/fused_gdn_gating.py`` through ``xtorch_ops`` stubs: each test
installs a stub whose kernel applies the documented operator semantics in
plain PyTorch, then checks the wrapper's plumbing (argument forwarding,
layout handling, output allocation) against an independent oracle computed
in the test body.

Order independence: stubs live in ``sys.modules`` only inside a context
manager that restores every touched entry, and modules that import
``xtorch_ops`` at module scope are re-imported through their real package
path inside that manager so relative imports resolve.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.supplementary]

_IMPL = "vllm_fl.dispatch.backends.vendor.kunlunxin.impl"

#: Every impl module below imports xtorch_ops at MODULE scope, so it can
#: only be (re-)imported while a stub is installed.  The whole subtree is
#: popped so each import starts fresh, and the parent packages stay loaded
#: for relative-import resolution.
_MODULE_SCOPE_IMPLS = [
    f"{_IMPL}.fused_gdn_gating",
]


@contextlib.contextmanager
def _xtorch_ops(**ops):
    """Install an ``xtorch_ops`` stub module with the given callables."""
    stub = ModuleType("xtorch_ops")
    for name, fn in ops.items():
        setattr(stub, name, fn)
    saved = sys.modules.get("xtorch_ops")
    sys.modules["xtorch_ops"] = stub
    try:
        yield stub
    finally:
        if saved is None:
            sys.modules.pop("xtorch_ops", None)
        else:
            sys.modules["xtorch_ops"] = saved


@contextlib.contextmanager
def _xtorch_ops_reimport(module_names, **ops):
    """Install a stub AND re-import the module-scope impl modules fresh.

    Saves the ``sys.modules`` entries for the target modules plus the stub
    itself, pops the targets, installs the stub, and restores every saved
    entry on exit -- no stub or freshly-imported module ever leaks out.
    """
    stub = ModuleType("xtorch_ops")
    for name, fn in ops.items():
        setattr(stub, name, fn)
    names = [*module_names, "xtorch_ops"]
    saved = {name: sys.modules.get(name) for name in names}
    for name in module_names:
        sys.modules.pop(name, None)
    sys.modules["xtorch_ops"] = stub
    try:
        modules = {name: importlib.import_module(name) for name in module_names}
        yield stub, modules
    finally:
        for name, entry in saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry


# ---------------------------------------------------------------------------
# activation.silu_and_mul_kunlunxin
# ---------------------------------------------------------------------------


class TestSiluAndMul:
    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize("shape", [(4, 12), (2, 3, 10), (8,)])
    def test_returns_half_width_and_matches_oracle(self, shape):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.activation import (
            silu_and_mul_kunlunxin,
        )

        torch.manual_seed(11)
        x = torch.randn(*shape)
        assert x.shape[-1] % 2 == 0  # even last dim: equal gate/up split
        half = x.shape[-1] // 2
        gate, up = x[..., :half], x[..., half:]
        # swiglu contract: out = silu(gate) * up, elementwise, width = half
        oracle = gate * torch.sigmoid(gate) * up

        received = []

        def swiglu(x_in, out):
            received.append((x_in, out))
            g, u = x_in[..., :half], x_in[..., half:]
            out.copy_(g * torch.sigmoid(g) * u)

        with _xtorch_ops(swiglu=swiglu):
            result = silu_and_mul_kunlunxin(object(), x)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (*shape[:-1], half)
        assert result.dtype == x.dtype
        # the wrapper passes the caller's tensor and its own output buffer
        assert len(received) == 1
        assert received[0][0] is x
        assert received[0][1] is result
        torch.testing.assert_close(result, oracle)

    @pytest.mark.platforms("kunlunxin")
    def test_kernel_result_is_returned_not_the_input(self):
        # The wrapper must return the buffer it handed to the kernel, so a
        # kernel writing garbage shows up verbatim (no hidden recompute).
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.activation import (
            silu_and_mul_kunlunxin,
        )

        x = torch.randn(3, 8)

        def swiglu(x_in, out):
            out.fill_(3.5)

        with _xtorch_ops(swiglu=swiglu):
            result = silu_and_mul_kunlunxin(object(), x)

        assert result.shape == (3, 4)
        assert torch.all(result == 3.5)


# ---------------------------------------------------------------------------
# normalization.rms_norm_kunlunxin
# ---------------------------------------------------------------------------


def _rms_oracle(x, weight, eps):
    x32 = x.float()
    norm = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (norm * weight.float()).to(x.dtype)


class TestRmsNorm:
    @pytest.mark.platforms("kunlunxin")
    def test_residual_branch_fused_add_contract(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.normalization import (
            rms_norm_kunlunxin,
        )

        torch.manual_seed(5)
        x = torch.randn(4, 6)
        residual = torch.randn(4, 6)
        weight = torch.randn(6)
        eps = 1e-6
        obj = type("Norm", (), {})()
        obj.weight = weight
        obj.variance_epsilon = eps

        add_oracle = x + residual
        norm_oracle = _rms_oracle(add_oracle, weight, eps)
        x_snapshot = x.clone()
        residual_snapshot = residual.clone()
        calls = []

        def add_rmsnorm(**kwargs):
            # record the pre-write values: the wrapper passes its own output
            # buffers as x/y, so snapshot them before any in-place store.
            calls.append(
                {
                    "x": kwargs["x"].clone(),
                    "y": kwargs["y"].clone(),
                    "x_obj": kwargs["x"],
                    "y_obj": kwargs["y"],
                    "weight": kwargs["weight"],
                    "eps": kwargs["eps"],
                    "output": kwargs["output"],
                    "residual_output": kwargs["residual_output"],
                    "store_output_before_norm": kwargs["store_output_before_norm"],
                }
            )
            # vLLM's fused_add_rms_norm contract (verified against
            # RMSNorm.forward_native on the reference path):
            # residual_output = x + y; output = rmsnorm(x + y).
            # y and residual_output alias, so compute the sum exactly once
            # before any in-place store.
            s = kwargs["x"] + kwargs["y"]
            kwargs["residual_output"].copy_(s)
            kwargs["output"].copy_(
                _rms_oracle(s, kwargs["weight"], kwargs["eps"])
            )

        with _xtorch_ops(add_rmsnorm=add_rmsnorm):
            out_x, out_residual = rms_norm_kunlunxin(obj, x, residual)

        assert len(calls) == 1
        call = calls[0]
        assert call["x_obj"] is x
        assert call["y_obj"] is residual
        torch.testing.assert_close(call["x"], x_snapshot)
        torch.testing.assert_close(call["y"], residual_snapshot)
        assert call["weight"] is weight
        assert call["eps"] == eps
        assert call["output"] is x
        assert call["residual_output"] is residual
        assert call["store_output_before_norm"] is True
        # the branch returns the very tensors it was given
        assert out_x is x
        assert out_residual is residual
        torch.testing.assert_close(out_residual, add_oracle)
        torch.testing.assert_close(out_x, norm_oracle)

    @pytest.mark.platforms("kunlunxin")
    def test_strided_2d_view_is_passed_asis(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.normalization import (
            rms_norm_kunlunxin,
        )

        torch.manual_seed(7)
        base = torch.randn(6, 12)
        x = base[:, ::3]  # shape (6, 4), stride (12, 3): stride(-2) != size(-1)
        assert x.stride(-2) != x.size(-1)
        weight = torch.randn(4)
        eps = 1e-5
        obj = type("Norm", (), {})()
        obj.weight = weight
        obj.variance_epsilon = eps
        oracle = _rms_oracle(x, weight, eps)
        received = []

        def rmsnorm(x_in, w, out, epsilon):
            received.append((x_in, w, out, epsilon))
            out.copy_(_rms_oracle(x_in, w, epsilon))

        with _xtorch_ops(rmsnorm=rmsnorm):
            out = rms_norm_kunlunxin(obj, x)

        assert len(received) == 1
        x_in, w, out_buf, epsilon = received[0]
        # non-contiguous 2D views are handed to the strided-capable kernel
        assert x_in is x
        assert w is weight
        assert epsilon == eps
        assert out is not None and out is out_buf
        assert out.shape == x.shape
        torch.testing.assert_close(out, oracle)

    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize("shape", [(3, 5), (2, 4, 6)])
    def test_contiguous_input_reaches_kernel_unchanged(self, shape):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.normalization import (
            rms_norm_kunlunxin,
        )

        torch.manual_seed(9)
        x = torch.randn(*shape)
        weight = torch.randn(x.shape[-1])
        eps = 1e-6
        obj = type("Norm", (), {})()
        obj.weight = weight
        obj.variance_epsilon = eps
        oracle = _rms_oracle(x, weight, eps)
        received = []

        def rmsnorm(x_in, w, out, epsilon):
            received.append((x_in, w, out, epsilon))
            out.copy_(_rms_oracle(x_in, w, epsilon))

        with _xtorch_ops(rmsnorm=rmsnorm):
            out = rms_norm_kunlunxin(obj, x)

        x_in, w, out_buf, epsilon = received[0]
        # contiguous tensors reach the kernel as the same object
        assert x_in is x
        assert w is weight
        assert epsilon == eps
        assert out is out_buf
        torch.testing.assert_close(out, oracle)

    @pytest.mark.platforms("kunlunxin")
    def test_dense_but_noncontiguous_2d_gets_a_copy(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.normalization import (
            rms_norm_kunlunxin,
        )

        # stride(-2) == size(-1) but stride(-1) != 1: the else-branch calls
        # .contiguous(), so the kernel sees a dense copy, not the strided view.
        torch.manual_seed(3)
        base = torch.randn(24)
        x = torch.as_strided(base, (3, 4), (4, 2))
        assert x.stride(-2) == x.size(-1) and not x.is_contiguous()
        weight = torch.randn(4)
        eps = 1e-6
        obj = type("Norm", (), {})()
        obj.weight = weight
        obj.variance_epsilon = eps
        oracle = _rms_oracle(x.contiguous(), weight, eps)
        received = []

        def rmsnorm(x_in, w, out, epsilon):
            received.append(x_in)
            out.copy_(_rms_oracle(x_in, w, epsilon))

        with _xtorch_ops(rmsnorm=rmsnorm):
            out = rms_norm_kunlunxin(obj, x)

        x_in = received[0]
        assert x_in is not x
        assert x_in.is_contiguous()
        torch.testing.assert_close(x_in, x.contiguous())
        torch.testing.assert_close(out, oracle)


# ---------------------------------------------------------------------------
# rotary.rotary_embedding_kunlunxin
# ---------------------------------------------------------------------------


def _rope_row(seg, cos, sin, neox):
    """Reference rotary update of one head's ``seg`` (length D)."""
    half = seg.shape[-1] // 2
    out = torch.empty_like(seg)
    if neox:
        d1, d2 = seg[..., :half], seg[..., half:]
        out[..., :half] = d1 * cos - d2 * sin
        out[..., half:] = d1 * sin + d2 * cos
    else:
        even, odd = seg[..., 0::2], seg[..., 1::2]
        out[..., 0::2] = even * cos - odd * sin
        out[..., 1::2] = even * sin + odd * cos
    return out


def _rope_tensor(x, positions, cos, sin, neox):
    """Rotate ``x`` of shape [T, H, D] per position with cos/sin [T, D//2]."""
    out = torch.empty_like(x)
    for t in range(x.shape[0]):
        for h in range(x.shape[1]):
            out[t, h] = _rope_row(x[t, h], cos[t], sin[t], neox)
    return out


def _make_rotary_stub(calls):
    def rotary_embedding(position_ids, q_flat, k_flat, head_size, cos_sin_cache, is_neox):
        calls.append(
            {
                "position_ids": position_ids.clone(),
                "q_flat": q_flat.clone(),
                "k_flat": k_flat.clone(),
                "q_flat_obj": q_flat,
                "head_size": head_size,
                "cos_sin_cache": cos_sin_cache.clone(),
                "is_neox": is_neox,
            }
        )
        half = head_size // 2
        for flat in (q_flat, k_flat):
            for t, pos in enumerate(position_ids.tolist()):
                cos = cos_sin_cache[pos, :half]
                sin = cos_sin_cache[pos, half:]
                heads = flat[t].view(-1, head_size)
                for h in range(heads.shape[0]):
                    heads[h] = _rope_row(heads[h], cos, sin, is_neox)

    return rotary_embedding


class TestRotaryEmbedding:
    @pytest.fixture
    def rope_inputs(self):
        torch.manual_seed(13)
        q = torch.randn(3, 2, 8)
        k = torch.randn(3, 1, 8)
        cos = torch.randn(6, 4)
        sin = torch.randn(6, 4)
        positions = torch.tensor([0, 2, 5])
        return q, k, cos, sin, positions

    @pytest.mark.platforms("kunlunxin")
    def test_neox_rotation_inplace_on_contiguous_query(self, rope_inputs):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.rotary import (
            rotary_embedding_kunlunxin,
        )

        q, k, cos, sin, positions = rope_inputs
        q_before = q.clone()
        k_before = k.clone()
        q_oracle = _rope_tensor(q, positions, cos[positions], sin[positions], True)
        k_oracle = _rope_tensor(k, positions, cos[positions], sin[positions], True)
        calls = []
        stub = _make_rotary_stub(calls)

        with _xtorch_ops(rotary_embedding=stub):
            q_embed, k_embed = rotary_embedding_kunlunxin(
                object(), q, k, cos, sin, positions
            )

        call = calls[0]
        # cache layout: first half cos, second half sin
        torch.testing.assert_close(call["cos_sin_cache"], torch.cat([cos, sin], dim=-1))
        # flattening contract: [T, H*D] contiguous views of the inputs
        assert call["q_flat"].shape == (3, 16)
        assert call["k_flat"].shape == (3, 8)
        torch.testing.assert_close(call["q_flat"], q_before.reshape(3, -1))
        torch.testing.assert_close(call["k_flat"], k_before.reshape(3, -1))
        assert call["head_size"] == 8
        # default rotary_interleaved=False maps to is_neox=True
        assert call["is_neox"] is True
        # shapes restored
        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape
        # contiguous inputs are rotated truly in place
        assert q_embed.data_ptr() == q.data_ptr()
        assert k_embed.data_ptr() == k.data_ptr()
        torch.testing.assert_close(q_embed, q_oracle)
        torch.testing.assert_close(k_embed, k_oracle)
        # and the caller's tensors see the rotation
        torch.testing.assert_close(q, q_oracle)
        torch.testing.assert_close(k, k_oracle)

    @pytest.mark.platforms("kunlunxin")
    def test_interleaved_flag_flips_is_neox(self, rope_inputs):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.rotary import (
            rotary_embedding_kunlunxin,
        )

        q, k, cos, sin, positions = rope_inputs
        q_oracle = _rope_tensor(q, positions, cos[positions], sin[positions], False)
        k_oracle = _rope_tensor(k, positions, cos[positions], sin[positions], False)
        calls = []
        stub = _make_rotary_stub(calls)

        with _xtorch_ops(rotary_embedding=stub):
            q_embed, k_embed = rotary_embedding_kunlunxin(
                object(), q, k, cos, sin, positions, rotary_interleaved=True
            )

        assert calls[0]["is_neox"] is False
        torch.testing.assert_close(q_embed, q_oracle)
        torch.testing.assert_close(k_embed, k_oracle)

    @pytest.mark.platforms("kunlunxin")
    def test_noncontiguous_query_gets_a_contiguous_copy(self, rope_inputs):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.rotary import (
            rotary_embedding_kunlunxin,
        )

        _, _, cos, sin, positions = rope_inputs
        full = torch.randn(3, 4, 8)
        q = full[:, ::2, :]  # shape (3, 2, 8), non-contiguous
        k = torch.randn(3, 1, 8)
        assert not q.is_contiguous()
        q_oracle = _rope_tensor(q, positions, cos[positions], sin[positions], True)
        q_before = q.clone()
        calls = []
        stub = _make_rotary_stub(calls)

        with _xtorch_ops(rotary_embedding=stub):
            q_embed, k_embed = rotary_embedding_kunlunxin(
                object(), q, k, cos, sin, positions
            )

        # the kernel received a dense copy, not the strided view
        assert calls[0]["q_flat_obj"].is_contiguous()
        torch.testing.assert_close(calls[0]["q_flat"], q.reshape(3, -1))
        # the returned embedding is a new dense tensor; the input is untouched
        assert q_embed.data_ptr() != q.data_ptr()
        assert q_embed.is_contiguous()
        torch.testing.assert_close(q, q_before)
        torch.testing.assert_close(q_embed, q_oracle)

    @pytest.mark.platforms("kunlunxin")
    def test_position_ids_forwarded_verbatim(self, rope_inputs):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.rotary import (
            rotary_embedding_kunlunxin,
        )

        q, k, cos, sin, positions = rope_inputs
        calls = []
        stub = _make_rotary_stub(calls)

        with _xtorch_ops(rotary_embedding=stub):
            rotary_embedding_kunlunxin(object(), q, k, cos, sin, positions)

        torch.testing.assert_close(calls[0]["position_ids"], positions)


# ---------------------------------------------------------------------------
# fused_gdn_gating.fused_gdn_gating_kunlunxin
# ---------------------------------------------------------------------------


class TestFusedGdnGating:
    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize(
        ("beta", "threshold"),
        [(1.0, 20.0), (2.0, 15.0)],
        ids=["defaults-shaped", "custom-softplus-params"],
    )
    def test_kernel_contract_and_output_layout(self, beta, threshold):
        torch.manual_seed(17)
        heads = 3
        A_log = torch.randn(heads)
        a = torch.randn(2, heads)
        b = torch.randn(2, heads)
        dt_bias = torch.randn(heads)

        # Independent oracle: g = -exp(A_log) * softplus(a + dt_bias)
        g_oracle = (
            -torch.exp(A_log) * F.softplus(a + dt_bias, beta=beta, threshold=threshold)
        ).float()
        beta_oracle = b.sigmoid().unsqueeze(0)

        calls = []

        def fused_gdn_gating(A_log_in, a_in, dt_bias_in, beta_in, threshold_in):
            calls.append((A_log_in, a_in, dt_bias_in, beta_in, threshold_in))
            return (
                -torch.exp(A_log_in)
                * F.softplus(a_in + dt_bias_in, beta=beta_in, threshold=threshold_in)
            ).float()

        with _xtorch_ops_reimport(
            _MODULE_SCOPE_IMPLS, fused_gdn_gating=fused_gdn_gating
        ) as (_, modules):
            fn = modules[f"{_IMPL}.fused_gdn_gating"].fused_gdn_gating_kunlunxin
            g, beta_out = fn(A_log, a, b, dt_bias, beta=beta, threshold=threshold)

        assert len(calls) == 1
        A_log_in, a_in, dt_bias_in, beta_in, threshold_in = calls[0]
        assert A_log_in is A_log
        assert a_in is a
        assert dt_bias_in is dt_bias
        assert beta_in == beta
        assert threshold_in == threshold
        # output layout: [1, batch, heads]
        assert g.shape == (1, 2, heads)
        assert beta_out.shape == (1, 2, heads)
        assert g.dtype == torch.float32
        torch.testing.assert_close(g, g_oracle.unsqueeze(0))
        torch.testing.assert_close(beta_out, beta_oracle)

    @pytest.mark.platforms("kunlunxin")
    def test_default_softplus_parameters(self):
        torch.manual_seed(19)
        heads = 2
        A_log = torch.randn(heads)
        a = torch.randn(1, heads)
        b = torch.randn(1, heads)
        dt_bias = torch.randn(heads)
        calls = []

        def fused_gdn_gating(A_log_in, a_in, dt_bias_in, beta_in, threshold_in):
            calls.append((beta_in, threshold_in))
            return (-torch.exp(A_log_in) * F.softplus(a_in + dt_bias_in)).float()

        with _xtorch_ops_reimport(
            _MODULE_SCOPE_IMPLS, fused_gdn_gating=fused_gdn_gating
        ) as (_, modules):
            fn = modules[f"{_IMPL}.fused_gdn_gating"].fused_gdn_gating_kunlunxin
            g, beta_out = fn(A_log, a, b, dt_bias)

        assert calls == [(1.0, 20.0)]
        assert beta_out.shape == (1, 1, heads)
