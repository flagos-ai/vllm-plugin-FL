# Copyright 2025 BAAI. All rights reserved.
"""Vendor-path contracts for ``TritonExpertsFL.apply`` (fused_moe_utils).

``test_fused_moe_utils_supplementary.py`` covers the NPU
``call_op("grouped_moe_experts")`` arm, the kunlunxin ``resolve_op`` arm
and the workspace/moe_sum surface; the two remaining arms of the apply()
dispatch (production ``vllm_fl/ops/fused_moe/fused_moe_utils.py``
L435-635, ~40 statements) were previously unexecuted. This file closes
them and pins the vendor routing matrix across the OOT platforms:

* the CUDA FlagGems fast path — ``current_platform.is_cuda()`` with no
  LoRA context imports ``flaggems_vllm.fused_experts_impl`` lazily and
  copies its result into the caller's output with the full quant-config
  surface and ``inplace=False``;
* the generic Triton two-GEMM pipeline — the fallback every non-CUDA
  vendor takes (iluvatar/hygon report ``device_type=cuda`` but
  ``is_cuda()=False``) and every LoRA context takes even on CUDA: the
  shape guards, the optimal-config lookup, the expert assignment, the
  two kernel invocations (GEMM1 always unweighted, GEMM2 weighted by
  ``not apply_router_weight_on_input``), the activation/quantize steps
  in order and the final ``moe_sum``;
* the routing matrix: which arm each of kunlunxin / iluvatar / hygon
  reaches on a real deployment of that vendor.

The kernel boundary (``flaggems_vllm``, the triton
GEMM/align/sum CachedOps, the vllm config oracle and the upstream
``TritonExperts`` LoRA helpers) is stubbed with recording doubles; the
``TritonExpertsFL.apply`` body itself runs unmodified.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import vllm_fl.ops.fused_moe.fused_moe_utils as fmu

pytestmark = [pytest.mark.supplementary]


def _fake_platform(**predicates):
    """A current_platform double with configurable predicates."""
    vendor = predicates.pop("vendor_name", None)
    device_type = predicates.pop("device_type", "cpu")

    def _false(*args, **kwargs):
        return False

    platform = SimpleNamespace(
        is_cuda=predicates.get("is_cuda", _false),
        device_type=device_type,
        vendor_name=vendor,
    )
    return platform


def _experts(**attrs):
    """Build a TritonExpertsFL double with a full attribute surface.

    Upstream ``TritonExperts`` exposes w1_scale/w2_scale/w1_bias/w2_bias/
    block_shape/per_act_token_quant/quant_dtype as read-only properties
    delegating to ``quant_config``, so everything the FL apply() body
    reads is baked into that namespace; only plain attributes are set on
    the instance.
    """
    quant_config = SimpleNamespace(
        quant_dtype=None,
        weight_quant_dtype=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config_name=lambda dtype: "float32",
        w1_scale=None,
        w2_scale=None,
        w1_bias=None,
        w2_bias=None,
        block_shape=None,
        per_act_token_quant=False,
    )
    experts = object.__new__(fmu.TritonExpertsFL)
    experts.quant_config = quant_config
    experts._lora_context = attrs.pop("lora_context", None)
    experts.moe_config = SimpleNamespace(max_num_tokens=64)
    experts.activation_config = SimpleNamespace(clamp_limit=None)
    experts.quantization_emulation = False
    for key, value in attrs.items():
        setattr(experts, key, value)
    return experts


def _arguments(M=4, E=2, K=4, N=6, I=3, topk=1, seed=0):
    """CPU tensors matching the apply() signature defaults."""
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(E, 2 * I, K, generator=g) * 0.2
    w2 = torch.randn(E, K, I, generator=g) * 0.2
    return {
        "output": torch.empty(M, K),
        "hidden_states": torch.randn(M, K, generator=g),
        "w1": w1,
        "w2": w2,
        "topk_weights": torch.rand(M, topk, generator=g),
        "topk_ids": torch.randint(0, E, (M, topk), generator=g,
                                   dtype=torch.int64),
        "activation": fmu.MoEActivation.SILU,
        "global_num_experts": E,
        "expert_map": None,
        "a1q_scale": None,
        "a2_scale": None,
        "workspace13": torch.empty(M * topk * max(2 * I, K)),
        "workspace2": torch.empty(M * topk * max(N, K)),
        "expert_tokens_meta": None,
        "apply_router_weight_on_input": False,
    }


def _silence_generic_pipeline(monkeypatch):
    """Stub every generic-pipeline dispatch point with silent doubles.

    Returns the list the kernel invocations are recorded into, so tests
    that only care WHICH arm ran can also check the pipeline actually
    executed.
    """
    invocations = []

    def fake_invoke(A, B, C, A_scale, B_scale, topk_weights,
                    sorted_token_ids, expert_ids, num_tokens_post_padded,
                    mul_routed_weight, top_k, config, **kwargs):
        invocations.append({"B": B, "flag": bool(mul_routed_weight)})
        C.reshape(-1, C.shape[-1]).fill_(1.0)

    monkeypatch.setattr(fmu, "try_get_optimal_moe_config",
                        lambda *a, **k: {"BLOCK_SIZE_M": 16})
    monkeypatch.setattr(
        fmu, "_prepare_expert_assignment",
        lambda *a, **k: ("sorted", "experts", "padded"),
    )
    monkeypatch.setattr(fmu, "_invoke_fused_moe_triton_kernel", fake_invoke)
    monkeypatch.setattr(fmu, "apply_moe_activation",
                        lambda *a, **k: None)
    monkeypatch.setattr(
        fmu, "moe_kernel_quantize_input",
        lambda A, A_scale, *a, **k: (A, A_scale),
    )
    monkeypatch.setattr(fmu, "_moe_sum", lambda inp, out: None)
    _pin_triton_scalars(monkeypatch)
    return invocations


def _pin_triton_scalars(monkeypatch):
    """Pin the triton scalar aliases the compute-type lookup reads.

    ``vllm.triton_utils.tl`` is a placeholder without the dtype aliases
    on a CPU-only install; the production module maps torch dtypes onto
    them, so the aliases are pinned the same way the fused_moe tests do.
    """
    import vllm.triton_utils as tu

    monkeypatch.setattr(tu.tl, "float32", torch.float32, raising=False)
    monkeypatch.setattr(tu.tl, "bfloat16", torch.bfloat16, raising=False)
    monkeypatch.setattr(tu.tl, "float16", torch.float16, raising=False)


class TestCudaFlaggemsArm:
    """The is_cuda() fast path through flaggems_vllm.fused_experts_impl."""

    @pytest.fixture
    def flaggems_impl(self, monkeypatch):
        """Install a recording fused_experts_impl double in sys.modules."""
        captured = {}

        def fused_experts_impl(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return torch.full_like(args[0], 3.0)

        fake = ModuleType("flaggems_vllm")
        fake.fused_experts_impl = fused_experts_impl
        fake.__path__ = []
        monkeypatch.setitem(sys.modules, "flaggems_vllm", fake)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            is_cuda=lambda: True,
        ))
        return captured

    def test_calls_flaggems_and_copies_result(self, flaggems_impl):
        experts = _experts()
        arguments = _arguments()
        fmu.TritonExpertsFL.apply(experts, **arguments)

        # The full quant-config surface and the FL flags reached the
        # vendor kernel as keywords.
        kwargs = flaggems_impl["kwargs"]
        assert kwargs["inplace"] is False
        assert kwargs["activation"] == "silu"  # MoEActivation.SILU.value
        assert kwargs["apply_router_weight_on_input"] is False
        assert kwargs["use_fp8_w8a8"] is False
        assert kwargs["use_int8_w8a8"] is False
        assert kwargs["use_int8_w8a16"] is False
        assert kwargs["use_int4_w4a16"] is False
        assert kwargs["per_channel_quant"] is False
        assert kwargs["global_num_experts"] == 2
        assert kwargs["expert_map"] is None
        assert kwargs["w1_scale"] is None
        assert kwargs["w2_scale"] is None
        assert kwargs["a1_scale"] is None
        assert kwargs["a2_scale"] is None
        assert kwargs["block_shape"] is None
        assert kwargs["w1_bias"] is None
        assert kwargs["w2_bias"] is None
        # The five leading tensors arrived positionally.
        args = flaggems_impl["args"]
        assert args[0] is arguments["hidden_states"]
        assert args[1] is arguments["w1"]
        assert args[2] is arguments["w2"]
        assert args[3] is arguments["topk_weights"]
        assert args[4] is arguments["topk_ids"]
        # The kernel's result was copied into the caller's output.
        torch.testing.assert_close(
            arguments["output"], torch.full_like(arguments["output"], 3.0)
        )

    def test_router_weight_on_input_flag_forwarded(self, flaggems_impl):
        experts = _experts()
        arguments = _arguments()
        arguments["apply_router_weight_on_input"] = True
        fmu.TritonExpertsFL.apply(experts, **arguments)
        assert flaggems_impl["kwargs"]["apply_router_weight_on_input"] is True

    def test_lora_context_skips_flaggems_arm(self, flaggems_impl, monkeypatch):
        """Even on CUDA a LoRA context must fall to the generic pipeline."""
        invocations = _silence_generic_pipeline(monkeypatch)
        experts = _experts(lora_context=SimpleNamespace(mark="lora"))
        # The upstream LoRA helpers need a full adapter surface; keep the
        # pipeline quiet with no-op instance doubles so only the ROUTING
        # (flaggems vs pipeline) is what this test pins.
        experts.apply_w13_lora = lambda *a, **k: (None, None, None, None)
        experts.apply_w2_lora = lambda *a, **k: None
        arguments = _arguments()
        fmu.TritonExpertsFL.apply(experts, **arguments)
        # flaggems_vllm.fused_experts_impl was never consulted...
        assert flaggems_impl == {}
        # ...and the generic two-GEMM pipeline ran instead.
        assert [inv["flag"] for inv in invocations] == [False, True]


class TestGenericPipeline:
    """The two-GEMM Triton pipeline every non-CUDA vendor falls back to."""

    @pytest.fixture
    def pipeline_stubs(self, monkeypatch):
        """Recording doubles for every CachedOp dispatch point."""
        captured = {"invoke": [], "activation": [], "quantize": [],
                    "moe_sum": [], "align": None, "config": None}

        def fake_config(w1_shape, w2_shape, top_k, dtype_str, M, *,
                        block_shape=None):
            captured["config"] = (w1_shape, w2_shape, top_k, dtype_str, M)
            return {"BLOCK_SIZE_M": 16}

        def fake_prepare(topk_ids, config, num_tokens, top_k_num,
                         global_num_experts, expert_map, **kwargs):
            captured["align"] = {
                "topk_ids": topk_ids,
                "config": config,
                "num_tokens": num_tokens,
                "top_k_num": top_k_num,
                "global_num_experts": global_num_experts,
                **kwargs,
            }
            return ("sorted", "experts", "padded")

        def fake_invoke(A, B, C, A_scale, B_scale, topk_weights,
                        sorted_token_ids, expert_ids, num_tokens_post_padded,
                        mul_routed_weight, top_k, config, **kwargs):
            captured["invoke"].append({
                "A": A, "B": B, "C": C,
                "A_scale": A_scale, "B_scale": B_scale,
                "topk_weights": topk_weights,
                "sorted_token_ids": sorted_token_ids,
                "expert_ids": expert_ids,
                "mul_routed_weight": mul_routed_weight,
                "top_k": top_k, "config": config,
                **kwargs,
            })
            # Emulate the GEMM writing through to C so downstream steps
            # see plausible buffers.
            C.reshape(-1, C.shape[-1]).fill_(1.0)

        def fake_activation(activation, out, inp, *, activation_config=None,
                            **kwargs):
            captured["activation"].append(
                {"activation": activation, "out": out, "inp": inp}
            )
            out.fill_(2.0)

        def fake_quantize(A, A_scale, quant_dtype, per_act_token_quant,
                          block_shape, *, quantization_emulation=False):
            captured["quantize"].append(
                {"A": A, "A_scale": A_scale,
                 "per_act_token_quant": per_act_token_quant}
            )
            return A, A_scale

        monkeypatch.setattr(fmu, "try_get_optimal_moe_config", fake_config)
        monkeypatch.setattr(fmu, "_prepare_expert_assignment", fake_prepare)
        monkeypatch.setattr(fmu, "_invoke_fused_moe_triton_kernel", fake_invoke)
        monkeypatch.setattr(fmu, "apply_moe_activation", fake_activation)
        monkeypatch.setattr(fmu, "moe_kernel_quantize_input", fake_quantize)
        monkeypatch.setattr(
            fmu, "_moe_sum",
            lambda inp, out: captured["moe_sum"].append((inp, out)),
        )
        _pin_triton_scalars(monkeypatch)
        return captured

    def test_two_gemm_pipeline_orchestration(self, pipeline_stubs, monkeypatch):
        """The generic pipeline runs GEMM1 -> activation -> quantize ->
        GEMM2 -> moe_sum; GEMM1 is always unweighted, GEMM2 owns the
        routed weight unless it was applied on the input."""
        # iluvatar/hygon report device_type=cuda but is_cuda()=False, so
        # both the kunlunxin arm, the NPU arm and the FlagGems arm are
        # skipped and the generic pipeline runs.
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="iluvatar", device_type="cuda",
        ))
        experts = _experts()
        arguments = _arguments()
        fmu.TritonExpertsFL.apply(experts, **arguments)

        # The config oracle was consulted with the real problem shape.
        assert pipeline_stubs["config"] == (
            arguments["w1"].size(),
            arguments["w2"].size(),
            1,  # top_k
            "float32",  # config_name(hidden_states.dtype)
            4,  # num_tokens
        )
        # The expert assignment ran with the documented surface.
        align = pipeline_stubs["align"]
        assert align["num_tokens"] == 4
        assert align["top_k_num"] == 1
        assert align["global_num_experts"] == 2
        assert align["use_int8_w8a16"] is False
        assert align["use_int4_w4a16"] is False

        # Two GEMM invocations.
        assert len(pipeline_stubs["invoke"]) == 2
        first, second = pipeline_stubs["invoke"]
        # GEMM1: raw hidden states, w1, never weighted (None + False).
        assert first["A"] is arguments["hidden_states"]
        assert first["B"] is arguments["w1"]
        assert first["topk_weights"] is None
        assert first["mul_routed_weight"] is False
        assert first["top_k"] == 1
        assert first["use_fp8_w8a8"] is False
        assert first["B_bias"] is None
        # GEMM2: quantized intermediate rows, w2, weighted output because
        # apply_router_weight_on_input is False (the flag inverts).
        assert second["B"] is arguments["w2"]
        assert second["topk_weights"] is arguments["topk_weights"]
        assert second["mul_routed_weight"] is True
        assert second["top_k"] == 1
        # GEMM2's output cache is the moe_sum input.
        assert pipeline_stubs["moe_sum"][0][0] is second["C"]

        assert len(pipeline_stubs["activation"]) == 1
        assert len(pipeline_stubs["quantize"]) == 1
        assert pipeline_stubs["quantize"][0]["A"] is (
            pipeline_stubs["activation"][0]["out"]
        )
        act = pipeline_stubs["activation"][0]
        assert act["activation"] is fmu.MoEActivation.SILU
        # The activation read GEMM1's flat cache and wrote cache2.
        assert act["inp"].shape[0] == 4  # (M*topk, N) view of cache1
        assert act["out"].shape == (4, 3)  # (M*topk, N//2)

        # The final output tensor came from moe_sum over GEMM2's cache.
        assert pipeline_stubs["moe_sum"][0][1] is arguments["output"]

    def test_router_weight_on_input_inverts_gemm2_flag(
            self, pipeline_stubs, monkeypatch):
        """apply_router_weight_on_input=True drops the GEMM2 weighting.

        GEMM1 is hardcoded to the unweighted contract (topk_weights=None,
        flag False), so with the on-input flag set NO GEMM applies the
        routed weight in this pipeline — the flag is only forwarded to
        GEMM2 inverted. This pins today's contract; see the file docstring
        for the suspected production gap this documents.
        """
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="hygon", device_type="cuda",
        ))
        experts = _experts()
        arguments = _arguments()
        arguments["apply_router_weight_on_input"] = True
        fmu.TritonExpertsFL.apply(experts, **arguments)

        first, second = pipeline_stubs["invoke"]
        assert first["mul_routed_weight"] is False
        assert second["mul_routed_weight"] is False

    @pytest.mark.parametrize("vendor", ["iluvatar", "hygon"])
    def test_iluvatar_hygon_reach_generic_pipeline(
            self, monkeypatch, vendor):
        """Neither OOT vendor reaches the CUDA FlagGems fast path."""
        flaggems_seen = []
        fake = ModuleType("flaggems_vllm")
        fake.fused_experts_impl = (
            lambda *a, **k: flaggems_seen.append(1)
        )
        fake.__path__ = []
        monkeypatch.setitem(sys.modules, "flaggems_vllm", fake)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name=vendor, device_type="cuda",
        ))
        invocations = _silence_generic_pipeline(monkeypatch)
        experts = _experts()
        arguments = _arguments()
        fmu.TritonExpertsFL.apply(experts, **arguments)
        # The fast path is gated on current_platform.is_cuda(), which
        # CoreX and Hygon DCUs report as False.
        assert flaggems_seen == []
        # And the generic pipeline ran instead.
        assert len(invocations) == 2

    def test_shape_guards_run_before_pipeline(self, monkeypatch):
        """The generic pipeline's asserts fire on mismatched shapes."""
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="iluvatar", device_type="cuda",
        ))
        # Stub only the config lookup; the asserts must fire first.
        config_called = []
        monkeypatch.setattr(
            fmu, "try_get_optimal_moe_config",
            lambda *a, **k: config_called.append(1) or {"BLOCK_SIZE_M": 16},
        )
        experts = _experts()
        arguments = _arguments()
        arguments["hidden_states"] = torch.randn(4, 5)  # K mismatch
        with pytest.raises(AssertionError, match="Hidden size mismatch"):
            fmu.TritonExpertsFL.apply(experts, **arguments)
        assert config_called == []

    def test_non_contiguous_hidden_states_rejected(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="iluvatar", device_type="cuda",
        ))
        experts = _experts()
        arguments = _arguments(K=8)
        strided = torch.randn(4, 16)[:, ::2]
        assert not strided.is_contiguous()
        arguments["hidden_states"] = strided
        with pytest.raises(AssertionError, match="contiguous"):
            fmu.TritonExpertsFL.apply(experts, **arguments)

    def test_non_unit_weight_stride_rejected(self, monkeypatch):
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="iluvatar", device_type="cuda",
        ))
        experts = _experts()
        arguments = _arguments(K=8)
        wide = torch.randn(2, 6, 16)
        arguments["w1"] = wide[:, :, ::2]
        with pytest.raises(AssertionError, match="Stride"):
            fmu.TritonExpertsFL.apply(experts, **arguments)

    def test_unsupported_dtype_rejected(self, monkeypatch):
        """A dtype outside the guarded list fails the dtype assert before
        the compute-type lookup (the ValueError in that lookup is
        unreachable behind the assert)."""
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name="iluvatar", device_type="cuda",
        ))
        experts = _experts()
        arguments = _arguments()
        arguments["hidden_states"] = arguments["hidden_states"].double()
        with pytest.raises(AssertionError):
            fmu.TritonExpertsFL.apply(experts, **arguments)


class TestVendorRoutingMatrix:
    """Which apply() arm each OOT vendor reaches end to end."""

    @pytest.mark.parametrize("vendor", ["iluvatar", "hygon", "kunlunxin"])
    def test_kunlunxin_routes_to_resolve_others_to_generic(
            self, monkeypatch, vendor):
        """kunlunxin resolves the vendor op; iluvatar/hygon fall back.

        Both arms are driven through recording doubles so the routing
        decision itself (which arm's boundary is consulted) is the
        assertion target — the arm bodies are covered above and in
        test_fused_moe_utils_supplementary.py.
        """
        vendor_calls = {"resolve": [], "flaggems": []}

        def fake_resolve(op_name):
            assert op_name == "fused_experts"

            def impl(*args, **kwargs):
                vendor_calls["resolve"].append((args, kwargs))
                return args[0]

            return impl

        fake = ModuleType("flaggems_vllm")
        fake.fused_experts_impl = (
            lambda *a, **k: vendor_calls["flaggems"].append(1)
        )
        fake.__path__ = []
        monkeypatch.setitem(sys.modules, "flaggems_vllm", fake)
        monkeypatch.setattr(fmu, "resolve_op", fake_resolve)
        monkeypatch.setattr(fmu, "current_platform", _fake_platform(
            vendor_name=vendor, device_type="cuda",
        ))
        invocations = _silence_generic_pipeline(monkeypatch)

        experts = _experts()
        arguments = _arguments()
        fmu.TritonExpertsFL.apply(experts, **arguments)

        if vendor == "kunlunxin":
            # The vendor op resolved and ran with the documented surface.
            assert len(vendor_calls["resolve"]) == 1
            _, kwargs = vendor_calls["resolve"][0]
            assert kwargs["inplace"] is False
            assert kwargs["activation"] == "silu"
            assert kwargs["apply_router_weight_on_input"] is False
            assert kwargs["global_num_experts"] == 2
        else:
            # iluvatar/hygon: no vendor op resolution, no FlagGems fast
            # path — the generic two-GEMM pipeline owns the call.
            assert vendor_calls["resolve"] == []
            assert vendor_calls["flaggems"] == []
            assert len(invocations) == 2
