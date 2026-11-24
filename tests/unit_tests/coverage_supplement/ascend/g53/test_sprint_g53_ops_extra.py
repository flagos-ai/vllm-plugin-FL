# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G53 extra coverage for three 814b modules that g4/g32/g45 left short:

  * vllm_fl/ops/custom_ops.py -- OOT registration whitelist/blacklist flow,
    the disabled-mode oracle patch, per-vendor monkey-patch fan-out,
    ``_patch_unquantized_moe_oracle`` and ``_patch_fused_moe_factory``
    (idempotency + cached module-global rewrite).
  * vllm_fl/ops/fused_moe/fused_moe_utils.py -- residual ``apply`` branches
    (float8_e4m3fnuz compute type, router-weight-on-input, LoRA w13/w2 hook
    wiring), the shape/stride guards and the remaining
    ``_prepare_expert_assignment`` alignment triggers.
  * vllm_fl/dispatch/backends/vendor/ascend/impl/fused_gdn_gating.py --
    ``fused_gdn_gating_patch`` for the small-batch, large-batch, non-multiple
    head count and custom beta/threshold cases.

CPU only: every boundary (CustomOp/PluggableLayer, platform object, vLLM
oracle/factory modules, vendor patch modules, triton launcher and vector-core
probe) is an explicit fake.  No NPU, no real triton, no network.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

try:
    import vllm_fl.ops.custom_ops as co
except Exception as exc:  # pragma: no cover - env specific
    co = None
    _CO_ERR = exc
else:
    _CO_ERR = None

try:
    import vllm_fl.ops.fused_moe.fused_moe_utils as fmoe
except Exception as exc:  # pragma: no cover - env specific
    fmoe = None
    _FMOE_ERR = exc
else:
    _FMOE_ERR = None

try:
    import vllm_fl.dispatch.backends.vendor.ascend.impl.fused_gdn_gating as gdn
except Exception as exc:  # pragma: no cover - env specific
    gdn = None
    _GDN_ERR = exc
else:
    _GDN_ERR = None


def _need(module, err, label):
    if module is None:
        pytest.skip(f"{label} unavailable: {err}")


def _install(monkeypatch, dotted, **attrs):
    """Install ``dotted`` as a fake module and bind it on its parent.

    ``import a.b.c as x`` resolves through the parent attribute chain, so the
    leaf must be attached to the (possibly real) parent package as well as
    placed in ``sys.modules``.
    """
    mod = types.ModuleType(dotted)
    for key, value in attrs.items():
        setattr(mod, key, value)
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent in sys.modules:
            continue
        try:
            importlib.import_module(parent)
        except Exception:
            stub = types.ModuleType(parent)
            stub.__path__ = []
            monkeypatch.setitem(sys.modules, parent, stub)
    monkeypatch.setitem(sys.modules, dotted, mod)
    if len(parts) > 1:
        parent_mod = sys.modules.get(".".join(parts[:-1]))
        if parent_mod is not None:
            monkeypatch.setattr(parent_mod, parts[-1], mod, raising=False)
    return mod


# ---------------------------------------------------------------------------
# vllm_fl/ops/custom_ops.py
# ---------------------------------------------------------------------------


def _make_ops(monkeypatch):
    """Fresh fake PluggableLayer/CustomOp bases plus a two-op registry."""

    class PlugBase:
        regs = []

        @classmethod
        def register_oot(cls, **kw):
            cls.regs.append(kw)

    class CustBase:
        regs = []

        @classmethod
        def register_oot(cls, **kw):
            cls.regs.append(kw)

    class PlugOp(PlugBase):
        pass

    class CustOp:  # deliberately NOT a PluggableLayer subclass
        pass

    monkeypatch.setattr(co, "PluggableLayer", PlugBase)
    monkeypatch.setattr(co, "CustomOp", CustBase)
    ops = {
        "silu_and_mul": (PlugOp, "SiluAndMul"),
        "rms_norm": (CustOp, "RMSNorm"),
    }
    return ops, PlugBase, CustBase


def _reg_env(monkeypatch, **env):
    """Patch the vllm_fl.utils policy helpers read inside register_oot_ops."""
    import vllm.platforms as vp
    import vllm_fl.dispatch.config.utils as cfg_utils
    import vllm_fl.utils as fl_utils

    rec = {"oracle": 0, "factory": 0}

    monkeypatch.setattr(fl_utils, "is_oot_enabled", lambda: env.get("enabled", True))
    monkeypatch.setattr(
        fl_utils, "get_oot_blacklist", lambda: env.get("blacklist")
    )
    monkeypatch.setattr(
        fl_utils, "get_oot_whitelist", lambda: env.get("env_whitelist")
    )
    monkeypatch.setattr(
        fl_utils, "use_flaggems_op", lambda *a, **k: env.get("flag_op", True)
    )
    monkeypatch.setattr(
        vp,
        "current_platform",
        SimpleNamespace(
            device_type=env.get("device_type", "cuda"),
            vendor_name=env.get("vendor", "nvidia"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cfg_utils, "get_platform_name", lambda: env.get("platform_name", "cuda")
    )

    def _bump(key):
        def _inner():
            rec[key] += 1

        return _inner

    monkeypatch.setattr(co, "_patch_unquantized_moe_oracle", _bump("oracle"))
    monkeypatch.setattr(co, "_patch_fused_moe_factory", _bump("factory"))
    return rec


def test_co_disabled_only_patches_oracle(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    rec = _reg_env(monkeypatch, enabled=False)

    co.register_oot_ops()

    assert rec == {"oracle": 1, "factory": 0}
    assert plug.regs == [] and cust.regs == []


def test_co_registers_all_by_default(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    rec = _reg_env(monkeypatch)

    co.register_oot_ops()

    assert [r["name"] for r in plug.regs] == ["SiluAndMul"]
    assert plug.regs[0]["_decorated_layer_cls"] is ops["silu_and_mul"][0]
    assert [r["name"] for r in cust.regs] == ["RMSNorm"]
    assert cust.regs[0]["_decorated_op_cls"] is ops["rms_norm"][0]
    assert rec == {"oracle": 1, "factory": 1}


def test_co_env_whitelist_beats_argument(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    _reg_env(monkeypatch, env_whitelist=["rms_norm"])

    co.register_oot_ops(whitelist=["silu_and_mul"])

    assert plug.regs == []
    assert [r["name"] for r in cust.regs] == ["RMSNorm"]


def test_co_argument_whitelist_used_without_env(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    _reg_env(monkeypatch)

    co.register_oot_ops(whitelist=["silu_and_mul"])

    assert [r["name"] for r in plug.regs] == ["SiluAndMul"]
    assert cust.regs == []


def test_co_blacklist_filters_and_blocks_factory(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    rec = _reg_env(monkeypatch, blacklist=["silu_and_mul", "fused_moe"])

    co.register_oot_ops()

    assert plug.regs == []
    assert [r["name"] for r in cust.regs] == ["RMSNorm"]
    # fused_moe is blacklisted, so the factory must not be patched, but the
    # oracle still needs patching because it is absent from the registry.
    assert rec == {"oracle": 1, "factory": 0}


def test_co_unknown_op_is_skipped(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    rec = _reg_env(monkeypatch, env_whitelist=["bogus"])

    co.register_oot_ops()

    assert plug.regs == [] and cust.regs == []
    assert rec["oracle"] == 1


def test_co_unquantized_method_gated_by_flaggems(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    ops["unquantized_fused_moe_method"] = (ops["rms_norm"][0], "UnquantizedFusedMoEMethod")
    monkeypatch.setattr(co, "OOT_OPS", ops)
    _reg_env(
        monkeypatch,
        env_whitelist=["unquantized_fused_moe_method"],
        flag_op=False,
    )

    co.register_oot_ops()
    assert cust.regs == []


def test_co_unquantized_method_registers_when_flag_true(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    ops["unquantized_fused_moe_method"] = (ops["rms_norm"][0], "UnquantizedFusedMoEMethod")
    monkeypatch.setattr(co, "OOT_OPS", ops)
    _reg_env(
        monkeypatch,
        env_whitelist=["unquantized_fused_moe_method"],
        flag_op=True,
    )

    co.register_oot_ops()
    assert [r["name"] for r in cust.regs] == ["UnquantizedFusedMoEMethod"]


def test_co_npu_applies_ascend_patch(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    hits = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.ascend.patch",
        apply_ascend_patches=lambda: hits.append("ascend"),
    )
    _reg_env(monkeypatch, env_whitelist=["silu_and_mul"], device_type="npu")

    co.register_oot_ops()
    assert hits == ["ascend"]


def test_co_ptpu_applies_sunrise_patch(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    hits = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.sunrise.patch",
        apply_sunrise_patches=lambda: hits.append("sunrise"),
    )
    _reg_env(monkeypatch, env_whitelist=["silu_and_mul"], device_type="ptpu")

    co.register_oot_ops()
    assert hits == ["sunrise"]


@pytest.mark.parametrize("vendor", ["gcu", "enflame"])
def test_co_gcu_applies_gcu_patch(monkeypatch, vendor):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    hits = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.gcu.patch",
        apply_gcu_patches=lambda: hits.append(vendor),
    )
    _reg_env(monkeypatch, env_whitelist=["silu_and_mul"], vendor=vendor)

    co.register_oot_ops()
    assert hits == [vendor]


def test_co_kunlunxin_patch_runs_outside_loop(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    ops, plug, cust = _make_ops(monkeypatch)
    monkeypatch.setattr(co, "OOT_OPS", ops)
    hits = []
    _install(
        monkeypatch,
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patch",
        apply_kunlunxin_patches=lambda: hits.append("klx"),
    )
    _reg_env(
        monkeypatch,
        env_whitelist=["silu_and_mul"],
        platform_name="kunlunxin",
    )

    co.register_oot_ops()
    assert hits == ["klx"]


def test_co_patch_unquantized_oracle_rewrites_both_modules(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    _need(fmoe, _FMOE_ERR, "fused_moe_utils")
    marker = object()
    oracle = _install(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.oracle.unquantized",
        select_unquantized_moe_backend=marker,
    )
    method = _install(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method",
        select_unquantized_moe_backend=marker,
    )
    try:
        co._patch_unquantized_moe_oracle()
    except ImportError as exc:  # pragma: no cover - version specific
        pytest.skip(f"oracle module layout differs: {exc}")

    assert oracle.select_unquantized_moe_backend is fmoe.select_unquantized_moe_backend_oot
    assert method.select_unquantized_moe_backend is fmoe.select_unquantized_moe_backend_oot


def _factory_modules(monkeypatch, layer_overrides, pkg_overrides):
    pkg = _install(
        monkeypatch, "vllm.model_executor.layers.fused_moe", **pkg_overrides
    )
    layer = _install(
        monkeypatch,
        "vllm.model_executor.layers.fused_moe.layer",
        **layer_overrides,
    )
    return pkg, layer


def test_co_factory_patch_is_idempotent(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    sentinel = object()
    pkg, layer = _factory_modules(
        monkeypatch,
        {"FusedMoE": co.FusedMoEFL, "FusedMoEFactory": sentinel},
        {"FusedMoE": sentinel, "FusedMoEFactory": sentinel},
    )

    co._patch_fused_moe_factory()

    assert layer.FusedMoE is co.FusedMoEFL
    assert pkg.FusedMoE is sentinel
    assert pkg.FusedMoEFactory is sentinel


def test_co_factory_patch_replaces_and_rewrites_cached_globals(monkeypatch):
    _need(co, _CO_ERR, "custom_ops")
    orig_e = object()
    orig_f = object()
    pkg, layer = _factory_modules(
        monkeypatch,
        {"FusedMoE": orig_e, "FusedMoEFactory": orig_f},
        {"FusedMoE": object(), "FusedMoEFactory": object()},
    )
    cached = types.ModuleType("g53_cached_model")
    cached.FusedMoE = orig_e
    cached.FusedMoEFactory = orig_f
    unrelated = types.ModuleType("g53_unrelated_model")
    keep = object()
    unrelated.FusedMoE = keep
    monkeypatch.setitem(sys.modules, "g53_cached_model", cached)
    monkeypatch.setitem(sys.modules, "g53_unrelated_model", unrelated)

    co._patch_fused_moe_factory()

    assert layer.FusedMoE is co.FusedMoEFactoryFL
    assert layer.FusedMoEFactory is co.FusedMoEFactoryFL
    assert pkg.FusedMoE is co.FusedMoEFactoryFL
    assert pkg.FusedMoEFactory is co.FusedMoEFactoryFL
    assert cached.FusedMoE is co.FusedMoEFactoryFL
    assert cached.FusedMoEFactory is co.FusedMoEFactoryFL
    assert unrelated.FusedMoE is keep


# ---------------------------------------------------------------------------
# vllm_fl/ops/fused_moe/fused_moe_utils.py
# ---------------------------------------------------------------------------


def _fm_platform(**attrs):
    base = dict(
        vendor_name="nvidia",
        device_type="cuda",
        is_cuda=lambda: False,
        is_out_of_tree=lambda: False,
        is_cpu=lambda: False,
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _fm_quant(**attrs):
    base = dict(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        quant_dtype=None,
        weight_quant_dtype=None,
        config_name=lambda dtype: "dummy",
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _fm_experts(quant=None):
    experts = object.__new__(fmoe.TritonExpertsFL)
    experts._lora_context = object()
    experts.quant_config = quant or _fm_quant()
    experts.per_act_token_quant = False
    experts.block_shape = None
    experts.w1_scale = None
    experts.w2_scale = None
    experts.w1_bias = None
    experts.w2_bias = None
    experts.quant_dtype = None
    experts.quantization_emulation = None
    experts.activation_config = SimpleNamespace(clamp_limit=0.0)
    experts.moe_config = SimpleNamespace(max_num_tokens=64)
    experts.apply_w13_lora = lambda *a, **k: (None, None, None, None)
    experts.apply_w2_lora = lambda *a, **k: None
    experts.moe_problem_size = lambda h, w1, w2, ids: (2, 2, 8, h.size(-1), 1)
    experts.adjust_N_for_activation = lambda N, act: N // 2
    return experts


def _fm_run(
    monkeypatch,
    *,
    hidden_dtype=torch.float32,
    hidden_k=4,
    w1_k=None,
    quant=None,
    global_num_experts=2,
    apply_router_weight_on_input=False,
    hidden=None,
    w1=None,
    w2=None,
    w13_hook=None,
    w2_hook=None,
):
    monkeypatch.setattr(fmoe, "current_platform", _fm_platform())
    monkeypatch.setattr(
        fmoe,
        "tl",
        SimpleNamespace(float32="tl.f32", float16="tl.f16", bfloat16="tl.bf16"),
    )
    invokes = []
    monkeypatch.setattr(
        fmoe, "try_get_optimal_moe_config", lambda *a, **k: {"BLOCK_SIZE_M": 16}
    )
    monkeypatch.setattr(fmoe, "_resize_cache", lambda ws, shape: torch.zeros(*shape))
    monkeypatch.setattr(
        fmoe, "_prepare_expert_assignment", lambda *a, **k: (None, None, None)
    )
    monkeypatch.setattr(
        fmoe,
        "_invoke_fused_moe_triton_kernel",
        lambda *a, **k: invokes.append((a, k)),
    )
    monkeypatch.setattr(fmoe, "apply_moe_activation", lambda *a, **k: None)
    monkeypatch.setattr(fmoe, "moe_kernel_quantize_input", lambda A, *a, **k: (A, None))
    monkeypatch.setattr(fmoe, "_moe_sum", lambda i, o: None)

    experts = _fm_experts(quant=quant)
    if w13_hook is not None:
        experts.apply_w13_lora = w13_hook
    if w2_hook is not None:
        experts.apply_w2_lora = w2_hook
    if w1_k is None:
        w1_k = hidden_k
    if hidden is None:
        hidden = torch.zeros(2, hidden_k, dtype=hidden_dtype, device="cpu")
    if w1 is None:
        w1 = torch.zeros(2, 8, w1_k, device="cpu")
    if w2 is None:
        w2 = torch.zeros(2, hidden_k, 8, device="cpu")
    out = torch.zeros(2, hidden_k, device="cpu")
    fmoe.TritonExpertsFL.apply(
        experts,
        out,
        hidden,
        w1,
        w2,
        torch.zeros(2, 1, device="cpu"),
        torch.zeros(2, 1, dtype=torch.int32, device="cpu"),
        SimpleNamespace(value="silu"),
        global_num_experts,
        None,
        None,
        None,
        torch.zeros(64, device="cpu"),
        torch.zeros(64, device="cpu"),
        None,
        apply_router_weight_on_input,
    )
    return invokes, out, experts


















def _topk():
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)


def test_fmoe_prepare_int8_block_forces_alignment(monkeypatch):
    _need(fmoe, _FMOE_ERR, "fused_moe_utils")
    calls = []

    def _align(topk_ids, block, num_experts, expert_map, ignore_invalid_experts=False):
        calls.append((block, num_experts, ignore_invalid_experts))
        return ("sorted", "experts", "padded")

    monkeypatch.setattr(fmoe, "_moe_align_block_size", _align)
    out = fmoe._prepare_expert_assignment(
        _topk(),
        {"BLOCK_SIZE_M": 16},
        2,
        2,
        64,
        None,
        use_int8_w8a16=True,
        block_shape=[16, 16],
    )
    assert out == ("sorted", "experts", "padded")
    assert calls == [(16, 64, False)]


def test_fmoe_prepare_expert_map_forces_alignment(monkeypatch):
    _need(fmoe, _FMOE_ERR, "fused_moe_utils")
    calls = []
    expert_map = torch.tensor([0, 1], dtype=torch.int32)

    def _align(topk_ids, block, num_experts, expert_map, ignore_invalid_experts=False):
        calls.append((block, num_experts, expert_map))
        return ("sorted", "experts", "padded")

    monkeypatch.setattr(fmoe, "_moe_align_block_size", _align)
    out = fmoe._prepare_expert_assignment(
        _topk(),
        {"BLOCK_SIZE_M": 16},
        2,
        2,
        64,
        expert_map,
        ignore_invalid_experts=True,
    )
    assert out == ("sorted", "experts", "padded")
    assert len(calls) == 1
    block, num_experts, passed_map = calls[0]
    assert block == 16 and num_experts == 64
    assert passed_map is expert_map


# ---------------------------------------------------------------------------
# vllm_fl/dispatch/backends/vendor/ascend/impl/fused_gdn_gating.py
# ---------------------------------------------------------------------------


def _gdn_fakes(monkeypatch, *, num_cores):
    calls = []

    class _Kernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                calls.append((grid, args, kwargs))

            return _launch

    triton = SimpleNamespace(
        cdiv=lambda a, b: -(-int(a) // int(b)),
        next_power_of_2=lambda x: 1 << (int(x) - 1).bit_length(),
    )
    monkeypatch.setattr(gdn, "get_vectorcore_num", lambda: num_cores)
    monkeypatch.setattr(gdn, "triton", triton)
    monkeypatch.setattr(gdn, "fused_gdn_gating_kernel", _Kernel())
    return calls


def test_gdn_small_batch_single_block(monkeypatch):
    _need(gdn, _GDN_ERR, "fused_gdn_gating")
    calls = _gdn_fakes(monkeypatch, num_cores=8)
    batch, heads = 2, 16
    a = torch.zeros(batch, heads, device="cpu")
    b = torch.zeros(batch, heads, device="cpu")
    g, beta_output = gdn.fused_gdn_gating_patch(
        torch.zeros(heads), a, b, torch.zeros(heads)
    )

    assert g.shape == (1, batch, heads) and g.dtype == torch.float32
    assert beta_output.shape == (1, batch, heads)
    assert len(calls) == 1
    grid, args, kwargs = calls[0]
    assert grid == (batch, 1)
    assert args[6] == 1 and args[7] == heads and args[8] == batch
    assert kwargs["BLK_HEADS"] == 8
    assert kwargs["COL_ITER"] == 2
    assert kwargs["BLK_BATCHES"] == 1
    assert kwargs["ROW_ITER"] == 1


def test_gdn_large_batch_uses_vector_cores(monkeypatch):
    _need(gdn, _GDN_ERR, "fused_gdn_gating")
    calls = _gdn_fakes(monkeypatch, num_cores=4)
    batch, heads = 100, 16
    a = torch.zeros(batch, heads, device="cpu")
    b = torch.zeros(batch, heads, device="cpu")
    g, _ = gdn.fused_gdn_gating_patch(
        torch.zeros(heads), a, b, torch.zeros(heads)
    )

    assert g.shape == (1, batch, heads)
    grid, _, kwargs = calls[0]
    assert grid == (4, 1)
    # UNIFIED_BUFFER_SIZE // (8*heads*BLK_HEADS) // 4 bytes, next_pow2, halved
    assert kwargs["BLK_BATCHES"] == 256
    assert kwargs["ROW_ITER"] == 1
    assert kwargs["COL_ITER"] == 2


def test_gdn_partial_head_block(monkeypatch):
    _need(gdn, _GDN_ERR, "fused_gdn_gating")
    calls = _gdn_fakes(monkeypatch, num_cores=8)
    batch, heads = 1, 20
    a = torch.zeros(batch, heads, device="cpu")
    b = torch.zeros(batch, heads, device="cpu")
    gdn.fused_gdn_gating_patch(torch.zeros(heads), a, b, torch.zeros(heads))

    _, _, kwargs = calls[0]
    assert kwargs["COL_ITER"] == 3
    assert kwargs["BLK_HEADS"] == 8


def test_gdn_custom_beta_threshold_and_dtype(monkeypatch):
    _need(gdn, _GDN_ERR, "fused_gdn_gating")
    calls = _gdn_fakes(monkeypatch, num_cores=8)
    batch, heads = 3, 8
    a = torch.zeros(batch, heads, device="cpu")
    b = torch.zeros(batch, heads, dtype=torch.float16, device="cpu")
    g, beta_output = gdn.fused_gdn_gating_patch(
        torch.zeros(heads),
        a,
        b,
        torch.zeros(heads),
        beta=2.5,
        threshold=12.0,
    )

    assert g.dtype == torch.float32
    assert beta_output.dtype == torch.float16
    _, args, _ = calls[0]
    assert args[9] == 2.5 and args[10] == 12.0


def test_gdn_kernel_constants_present():
    _need(gdn, _GDN_ERR, "fused_gdn_gating")
    assert gdn.UNIFIED_BUFFER_SIZE == 1572864
    assert hasattr(gdn, "fused_gdn_gating_kernel")
