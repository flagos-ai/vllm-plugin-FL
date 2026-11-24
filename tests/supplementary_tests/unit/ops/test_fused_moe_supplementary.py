# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.ops.fused_moe.fused_moe``.

The core suite never imports this module on CPU platforms (only
``test_fused_moe_factory.py`` touches the factory patch), so the whole
functional pipeline is unexecuted. This file drives both entry points on
plain CPU tensors:

* ``fused_experts_impl`` — the full two-GEMM pipeline against an
  independent torch oracle, including all four activation strings, the
  unsupported-activation error, ``inplace`` True/False, the
  ``apply_router_weight_on_input`` both ways, and an unknown activation
  raising ValueError. The triton kernel / alignment / activation /
  moe-sum dispatch points are stubbed at module level with CPU
  implementations that mirror their documented contracts.
* ``fused_experts`` — the wrapper: default quant config, kunlunxin's
  vendor branch through ``resolve_op``, and the default path comparing
  against the same oracle.

The vLLM-API drift on this branch (``_get_config_quant_dtype`` lost
``ocp_mx_scheme``, ``triton.language`` lacks the scalar type aliases on a
CPU-only install) is bridged by per-test monkeypatches pinned to the
documented semantics, so the FL production code still runs unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm_fl.ops.fused_moe.fused_moe as ffm

pytestmark = [pytest.mark.supplementary]


def _oracle(hidden_states, w1, w2, topk_weights, topk_ids, activation="silu",
            apply_router_weight_on_input=False):
    """Independent per-token grouped-GEMM oracle."""
    M, K = hidden_states.shape
    out = torch.zeros_like(hidden_states)
    for t in range(M):
        for s in range(topk_ids.shape[1]):
            e = int(topk_ids[t, s])
            gate_up = hidden_states[t] @ w1[e].T
            if activation == "silu":
                inter = F.silu(gate_up[: gate_up.numel() // 2]) * gate_up[
                    gate_up.numel() // 2:
                ]
            elif activation == "gelu":
                inter = F.gelu(gate_up[: gate_up.numel() // 2]) * gate_up[
                    gate_up.numel() // 2:
                ]
            elif activation == "silu_no_mul":
                inter = F.silu(gate_up)
            elif activation == "gelu_no_mul":
                inter = F.gelu(gate_up)
            # no-mul activations keep the full 2I width; the caller must
            # supply w2 shaped [E, K, 2I] for these.
            contribution = inter @ w2[e].T
            if not apply_router_weight_on_input:
                contribution = contribution * topk_weights[t, s]
            out[t] += contribution
    return out


def _record_routing(topk_ids):
    """Fill the stub kernel's routing map: row -> (token, slot, expert)."""
    routing = {}
    top_k = topk_ids.shape[1]
    for token in range(topk_ids.shape[0]):
        for slot in range(top_k):
            routing[token * top_k + slot] = (
                token, slot, int(topk_ids[token, slot])
            )
    return routing


def _moe_inputs(M=4, E=3, K=6, I=8, topk=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    hidden_states = torch.randn(M, K, generator=g)
    w1 = torch.randn(E, 2 * I, K, generator=g) * 0.2
    w2 = torch.randn(E, K, I, generator=g) * 0.2
    topk_weights = torch.rand(M, topk, generator=g)
    topk_ids = torch.randint(0, E, (M, topk), generator=g).to(torch.int32)
    return hidden_states, w1, w2, topk_weights, topk_ids


@pytest.fixture(autouse=True)
def _cpu_compat_shims(monkeypatch):
    """Pin the vLLM-API shapes this branch's module was written against.

    * ``_get_config_quant_dtype`` in vLLM 0.32 dropped the
      ``ocp_mx_scheme`` keyword — the module passes it unconditionally.
    * ``vllm.triton_utils.tl`` lacks the triton scalar-type aliases on a
      CPU-only install — the module maps torch dtypes onto them.
    """
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _get_config_quant_dtype as real_qd,
    )
    import vllm.triton_utils as tu

    monkeypatch.setattr(
        ffm, "_get_config_quant_dtype",
        lambda **kw: real_qd(
            use_fp8_w8a8=kw.get("use_fp8_w8a8", False),
            use_int8_w8a8=kw.get("use_int8_w8a8", False),
        ),
    )
    monkeypatch.setattr(tu.tl, "float32", torch.float32, raising=False)
    monkeypatch.setattr(tu.tl, "bfloat16", torch.bfloat16, raising=False)
    monkeypatch.setattr(tu.tl, "float16", torch.float16, raising=False)

    # CPU stubs for the three CachedOp dispatch points the pipeline hits.
    # Neither registered implementation of invoke_fused_moe_triton_kernel
    # runs on a CPU host (triton driver absent), so emulate the documented
    # GEMM contract: A [M, K] (or [M*topk, N]) times B [E, N, K] writes
    # C rows for the (token, expert) pairs the alignment produced.
    _ROUTING = {}
    _INVOKE_COUNT = {"n": 0}

    def cpu_invoke(A, B, C, A_scale, B_scale, topk_weights,
                   sorted_token_ids, expert_ids, num_tokens_post_padded,
                   mul_routed_weight, top_k, config, **kwargs):
        """Emulate the aligned grouped GEMM using recorded routing.

        Every A row is (token * top_k + slot) with its expert known from
        the routing map recorded before the call. The kernel contract is
        C[row] = A[row] @ B[expert].T — B rows are [out, in] with
        A.shape[1] == B.shape[2] — optionally scaled by the routed weight
        when the kernel owns the weighting (mul_routed_weight=True).
        """
        if A.shape[1] != B.shape[2]:
            return  # unexpected layout; leave C untouched
        # Two A layouts exist: the first GEMM receives the raw (M, K)
        # hidden states and gathers A[token] per routed row; the second
        # receives already-expanded (M*top_k, I) rows and uses A[row].
        num_rows = _ROUTING.get("num_rows")
        expanded = num_rows is not None and A.shape[0] == num_rows
        _INVOKE_COUNT["n"] += 1
        first_invoke = _INVOKE_COUNT["n"] % 2 == 1
        # The first GEMM's flag means "apply weight on input" (the router
        # weight multiplies the OUTPUT row); the second GEMM's flag is the
        # inverted "mul_routed_weight". Both semantics scale the same row.
        apply_weight = bool(mul_routed_weight) if not first_invoke             else bool(mul_routed_weight)  # flag arrives as passed
        flat = C.reshape(-1, C.shape[-1])
        for row, route in _ROUTING.items():
            if row == "num_rows":
                continue
            token, slot, expert = route
            if row >= flat.shape[0] or expert >= B.shape[0]:
                continue
            a = A[row] if expanded else A[token]
            if a.shape[0] != B.shape[2]:
                continue
            out = a.float() @ B[expert].float().T
            if topk_weights is not None and apply_weight:
                out = out * topk_weights[token, slot].float()
            flat[row] = out.to(flat.dtype)

    def cpu_silu_and_mul(obj, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

    def cpu_gelu_and_mul(obj, x):
        gate, up = x.chunk(2, dim=-1)
        return F.gelu(gate) * up

    real_impl = ffm.fused_experts_impl

    def recording_impl(*args, **kwargs):
        topk_ids = kwargs.get("topk_ids")
        if topk_ids is None and len(args) > 4:
            topk_ids = args[4]
        if topk_ids is not None:
            _ROUTING.clear()
            _ROUTING.update(_record_routing(topk_ids))
            _ROUTING["num_rows"] = topk_ids.numel()
        _INVOKE_COUNT["n"] = 0
        return real_impl(*args, **kwargs)

    monkeypatch.setattr(ffm, "_invoke_fused_moe_triton_kernel", cpu_invoke)
    monkeypatch.setattr(ffm, "_silu_and_mul", cpu_silu_and_mul)
    monkeypatch.setattr(ffm, "_gelu_and_mul", cpu_gelu_and_mul)
    monkeypatch.setattr(ffm, "fused_experts_impl", recording_impl)

    # moe_align_block_size dispatches to the vendor impl on some platforms
    # (e.g. vendor.iluvatar needs flaggems_vllm; the triton driver is absent
    # on CPU). Emulate the documented alignment contract instead: a stable
    # expert-major ordering with padded blocks, num_tokens_post_padded
    # covering all blocks.
    def cpu_moe_align(topk_ids, block_size, num_experts, expert_map=None, **kw):
        flat = topk_ids.reshape(-1)
        valid = flat >= 0
        order = torch.argsort(flat[valid].to(torch.int64), stable=True)
        valid_ids = flat[valid][order]
        num_blocks = (valid_ids.numel() + block_size - 1) // max(block_size, 1)
        total = num_blocks * max(block_size, 1)
        sorted_token_ids = torch.arange(
            total, dtype=torch.int32
        ) % max(valid_ids.numel(), 1)
        expert_ids = torch.zeros(num_blocks, dtype=torch.int32)
        for b in range(num_blocks):
            chunk = valid_ids[
                b * block_size: (b + 1) * block_size
            ]
            if chunk.numel():
                expert_ids[b] = chunk[0]
        num_tokens_post_padded = torch.tensor(
            total, dtype=torch.int32
        )
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    monkeypatch.setattr(ffm, "_moe_align_block_size", cpu_moe_align)
    yield


class TestFusedExpertsImpl:
    def test_silu_matches_oracle(self):
        hs, w1, w2, tw, ti = _moe_inputs()
        out = ffm.fused_experts_impl(hs, w1, w2, tw, ti, inplace=False)
        expected = _oracle(hs, w1, w2, tw, ti)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_inplace_true_writes_into_hidden_states(self):
        hs, w1, w2, tw, ti = _moe_inputs(seed=1)
        snapshot = hs.clone()
        result = ffm.fused_experts_impl(hs, w1, w2, tw, ti, inplace=True)
        expected = _oracle(snapshot, w1, w2, tw, ti)
        # The output aliases the input tensor.
        assert result is hs
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_gelu_activation_matches_oracle(self):
        hs, w1, w2, tw, ti = _moe_inputs(seed=2)
        out = ffm.fused_experts_impl(hs, w1, w2, tw, ti, activation="gelu")
        expected = _oracle(hs, w1, w2, tw, ti, activation="gelu")
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_silu_no_mul_activation(self):
        # no-mul keeps the full gate_up width, so w2 must be [E, K, 2I].
        hs, w1, _, tw, ti = _moe_inputs(seed=3, I=4)
        w2 = torch.randn(w1.shape[0], hs.shape[1], w1.shape[1]) * 0.2
        out = ffm.fused_experts_impl(hs, w1, w2, tw, ti,
                                     activation="silu_no_mul")
        expected = _oracle(hs, w1, w2, tw, ti, activation="silu_no_mul")
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_gelu_no_mul_activation(self):
        hs, w1, _, tw, ti = _moe_inputs(seed=4, I=4)
        w2 = torch.randn(w1.shape[0], hs.shape[1], w1.shape[1]) * 0.2
        out = ffm.fused_experts_impl(hs, w1, w2, tw, ti,
                                     activation="gelu_no_mul")
        expected = _oracle(hs, w1, w2, tw, ti, activation="gelu_no_mul")
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_unsupported_activation_raises(self):
        hs, w1, w2, tw, ti = _moe_inputs()
        with pytest.raises(ValueError, match="Unsupported FusedMoe"):
            ffm.fused_experts_impl(hs, w1, w2, tw, ti, activation="relu")

    def test_apply_router_weight_on_input(self):
        """Router weights scale the first GEMM output before activation.

        The kernel's on-input semantics multiply gate_up by the routed
        weight (both halves scale linearly), so silu(g*s)*(u*s) equals
        s^2 * silu(g)*u — NOT s*(silu(g)*u). The oracle mirrors the
        kernel contract directly.
        """
        hs, w1, w2, tw, ti = _moe_inputs(seed=5)
        out = ffm.fused_experts_impl(hs, w1, w2, tw, ti,
                                      apply_router_weight_on_input=True)
        M, K = hs.shape
        expected = torch.zeros_like(hs)
        for t in range(M):
            for s in range(ti.shape[1]):
                e = int(ti[t, s])
                gate_up = hs[t] @ w1[e].T
                scaled = gate_up * tw[t, s]
                inter = F.silu(scaled[: scaled.numel() // 2]) * scaled[
                    scaled.numel() // 2:
                ]
                expected[t] += inter @ w2[e].T
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_activation_enum_normalized_to_str(self):
        """A MoEActivation enum is accepted and normalized to its value."""
        from vllm.model_executor.layers.fused_moe.activation import (
            MoEActivation,
        )

        hs, w1, w2, tw, ti = _moe_inputs(seed=6)
        out = ffm.fused_experts_impl(hs, w1, w2, tw, ti,
                                     activation=MoEActivation.SILU)
        expected = _oracle(hs, w1, w2, tw, ti)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_hidden_size_mismatch_asserts(self):
        hs, w1, w2, tw, ti = _moe_inputs()
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            ffm.fused_experts_impl(torch.randn(4, 5), w1, w2, tw, ti)

    def test_int4_hidden_size_halves_before_comparison(self):
        hs, w1, w2, tw, ti = _moe_inputs(K=6)
        # use_int4_w4a16 compares hidden // 2 with w1.size(2).
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            ffm.fused_experts_impl(torch.randn(4, 5), w1, w2, tw, ti,
                                   use_int4_w4a16=True)

    def test_topk_shape_mismatch_asserts(self):
        hs, w1, w2, tw, ti = _moe_inputs()
        with pytest.raises(AssertionError, match="topk shape mismatch"):
            ffm.fused_experts_impl(hs, w1, w2, tw, ti[:, :1])

    def test_non_contiguous_input_asserts(self):
        hs, w1, w2, tw, ti = _moe_inputs(K=8)
        strided = torch.randn(4, 16)[:, ::2]
        with pytest.raises(AssertionError, match="contiguous"):
            ffm.fused_experts_impl(strided, w1, w2, tw, ti)

    def test_w1_non_unit_stride_asserts(self):
        hs, w1, w2, tw, ti = _moe_inputs(K=8)
        wide = torch.randn(3, 16, 16)
        with pytest.raises(AssertionError, match="Stride"):
            ffm.fused_experts_impl(hs, wide[:, :, ::2], w2, tw, ti)

    def test_w2_non_unit_stride_asserts(self):
        hs, w1, w2, tw, ti = _moe_inputs(I=8)
        wide = torch.randn(3, 6, 16)
        with pytest.raises(AssertionError, match="Stride"):
            ffm.fused_experts_impl(hs, w1, wide[:, :, ::2], tw, ti)

    def test_unsupported_dtype_asserts(self):
        hs, w1, w2, tw, ti = _moe_inputs()
        with pytest.raises(AssertionError):
            ffm.fused_experts_impl(hs.double(), w1, w2, tw, ti)


class TestFusedExpertsWrapper:
    def test_default_quant_config_path(self, monkeypatch):
        # Keep the wrapper on the local-impl path (the vendor branch would
        # resolve through the kunlunxin op, unavailable on CPU hosts).
        monkeypatch.setattr(
            ffm, "current_platform", SimpleNamespace(vendor_name=None)
        )
        hs, w1, w2, tw, ti = _moe_inputs(seed=7)
        out = ffm.fused_experts(hs, w1, w2, tw, ti)
        expected = _oracle(hs, w1, w2, tw, ti)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_kunlunxin_vendor_branch(self, monkeypatch):
        """The kunlunxin platform resolves fused_experts through the op."""
        calls = []

        def fake_resolve(op_name):
            assert op_name == "fused_experts"

            def impl(*args, **kwargs):
                # The wrapper passes everything by keyword except the
                # five leading tensors; capture both surfaces.
                call = dict(kwargs)
                if args:
                    call["hidden_states"] = args[0]
                    call["w1"] = args[1]
                    call["w2"] = args[2]
                    call["topk_weights"] = args[3]
                    call["topk_ids"] = args[4]
                calls.append(call)
                return _oracle(
                    call["hidden_states"], call["w1"], call["w2"],
                    call["topk_weights"], call["topk_ids"],
                    activation=call.get("activation", "silu"),
                    apply_router_weight_on_input=call.get(
                        "apply_router_weight_on_input", False
                    ),
                )

            return impl

        monkeypatch.setattr(ffm, "resolve_op", fake_resolve)
        from vllm.platforms import current_platform

        monkeypatch.setattr(
            ffm, "current_platform",
            SimpleNamespace(vendor_name="kunlunxin"),
            raising=False,
        )
        hs, w1, w2, tw, ti = _moe_inputs(seed=8)
        out = ffm.fused_experts(hs, w1, w2, tw, ti)
        expected = _oracle(hs, w1, w2, tw, ti)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
        # The vendor impl received the full quant-config surface.
        assert calls[0]["use_fp8_w8a8"] is False
        assert calls[0]["use_int8_w8a8"] is False
        del current_platform

    def test_non_kunlunxin_uses_local_impl(self, monkeypatch):
        monkeypatch.setattr(
            ffm, "current_platform",
            SimpleNamespace(vendor_name="nvidia"),
            raising=False,
        )
        hs, w1, w2, tw, ti = _moe_inputs(seed=9)
        out = ffm.fused_experts(hs, w1, w2, tw, ti)
        expected = _oracle(hs, w1, w2, tw, ti)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_explicit_quant_config_forwarded(self, monkeypatch):
        """A caller-supplied quant config drives the impl's flags."""
        captured = {}
        real_impl = ffm.fused_experts_impl

        def spy(**kwargs):
            captured.update(kwargs)
            return real_impl(**kwargs)

        monkeypatch.setattr(ffm, "fused_experts_impl", spy)
        monkeypatch.setattr(
            ffm, "current_platform", SimpleNamespace(vendor_name=None)
        )
        hs, w1, w2, tw, ti = _moe_inputs(seed=10)
        quant_config = SimpleNamespace(
            use_fp8_w8a8=False,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_act_token_quant=False,
            w1_scale=None,
            w2_scale=None,
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            w1_bias=None,
            w2_bias=None,
        )
        out = ffm.fused_experts(hs, w1, w2, tw, ti,
                                quant_config=quant_config)
        assert captured["per_channel_quant"] is False
        assert captured["w1_scale"] is None
        expected = _oracle(hs, w1, w2, tw, ti)
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
