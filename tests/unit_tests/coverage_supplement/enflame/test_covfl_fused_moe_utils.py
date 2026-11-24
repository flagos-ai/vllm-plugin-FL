# Copyright (c) 2026 BAAI. All rights reserved.

"""Coverage tests for the Enflame/S60 fused-MoE helpers (commit 030b452f).

Targets (read-only source):
  - vllm_fl/ops/fused_moe/fused_moe_utils.py
  - vllm_fl/ops/fused_moe/router.py

Strategy: every ``vllm``/``vllm_fl`` dependency is replaced with a lightweight
stub module before the two source files are loaded by path, so no real vLLM
install, accelerator, Triton/FlagGems kernel or GCU device is ever touched.
torch tensors are tiny CPU tensors (or attribute fakes where only ``.device``
is inspected).  Assertions focus on executed control flow, return values and
call plumbing -- never on numeric kernel output.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")

_MISSING = object()

_SRC = os.environ.get(
    "VLLM_FL_SRC",
    r"C:\Users\21649\AppData\Local\Temp\opencode\enflame-src-030b",
)
_SRC = os.path.abspath(_SRC)

_UTILS_REL = os.path.join("vllm_fl", "ops", "fused_moe", "fused_moe_utils.py")
_ROUTER_REL = os.path.join("vllm_fl", "ops", "fused_moe", "router.py")

_UTILS_PATH = os.path.join(_SRC, _UTILS_REL)
_ROUTER_PATH = os.path.join(_SRC, _ROUTER_REL)

if not (os.path.isfile(_UTILS_PATH) and os.path.isfile(_ROUTER_PATH)):
    pytest.skip(
        "cannot locate vllm_fl fused_moe sources; set VLLM_FL_SRC",
        allow_module_level=True,
    )


# --------------------------------------------------------------------------- #
# Fake building blocks
# --------------------------------------------------------------------------- #

class _BackendObj:
    """Identity-comparable stand-in for UnquantizedMoeBackend members."""

    def __init__(self, value):
        self.value = value


class _Backends:
    AITER = _BackendObj("aiter")
    TRITON = _BackendObj("triton")
    BATCHED_TRITON = _BackendObj("batched_triton")
    FLASHINFER_TRTLLM = _BackendObj("flashinfer_trtllm")
    FLASHINFER_CUTLASS = _BackendObj("flashinfer_cutlass")
    XPU = _BackendObj("xpu")
    CPU = _BackendObj("cpu")
    TPU = _BackendObj("tpu")


class _CachedOp:
    """Callable op placeholder mirroring vllm_fl.dispatch.CachedOp."""

    def __init__(self, name):
        self.name = name
        self.impl = MagicMock(name=f"cached_op::{name}")

    def __call__(self, *args, **kwargs):
        return self.impl(*args, **kwargs)


class _TritonExperts:
    def workspace_shapes(self, M, N, K, topk, g, e, meta, act):
        return ((M, K), (M, N), (M, K))


class _MoEActivation:
    SWIGLUOAI = SimpleNamespace(value="swigluoai")


def _fake_kernel(supported, reason=None):
    class _K:
        @staticmethod
        def is_supported_config(k_cls, config, a, b, fmt):
            return supported, reason

    return _K


class FakePlatform:
    def __init__(
        self,
        *,
        cuda=False,
        rocm=False,
        xpu=False,
        cpu=False,
        tpu=False,
        oot=False,
        vendor="",
        device_type="cpu",
        family=None,
    ):
        self._cuda = cuda
        self._rocm = rocm
        self._xpu = xpu
        self._cpu = cpu
        self._tpu = tpu
        self._oot = oot
        self.vendor_name = vendor
        self.device_type = device_type
        self.family = family

    def is_cuda(self):
        return self._cuda

    def is_rocm(self):
        return self._rocm

    def is_xpu(self):
        return self._xpu

    def is_cpu(self):
        return self._cpu

    def is_tpu(self):
        return self._tpu

    def is_out_of_tree(self):
        return self._oot

    def is_device_capability_family(self, cap):
        return self.family == cap


# --------------------------------------------------------------------------- #
# sys.modules stub tree + file-path loading
# --------------------------------------------------------------------------- #

_PKGS = {
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.fused_moe.experts",
    "vllm.model_executor.layers.fused_moe.oracle",
    "vllm.model_executor.layers.fused_moe.router",
    "vllm.model_executor.layers.quantization",
    "vllm.model_executor.layers.quantization.utils",
    "vllm_fl",
    "vllm_fl.ops",
    "vllm_fl.ops.fused_moe",
    "vllm_fl.dispatch",
    "vllm_fl.dispatch.backends",
    "vllm_fl.dispatch.backends.vendor",
    "vllm_fl.dispatch.backends.vendor.ascend",
    "vllm_fl.dispatch.backends.vendor.ascend.impl",
}

_AITER_OPS = SimpleNamespace(
    is_rdna_aiter_enabled=MagicMock(return_value=False),
    is_fused_moe_enabled=MagicMock(return_value=False),
    is_fusion_moe_shared_experts_enabled=MagicMock(return_value=False),
)


def _build_stubs():
    mods = {}

    def _new(name, attrs):
        m = types.ModuleType(name)
        if name in _PKGS:
            m.__path__ = []
            m.__package__ = name
        for key, value in (attrs or {}).items():
            setattr(m, key, value)
        mods[name] = m
        return m

    leaf_attrs = {
        "vllm.envs": {
            "is_set": MagicMock(return_value=False),
            "VLLM_ROCM_USE_AITER": False,
            "VLLM_ROCM_USE_AITER_MOE": False,
        },
        "vllm.logger": {"init_logger": MagicMock(return_value=MagicMock())},
        "vllm._aiter_ops": {"rocm_aiter_ops": _AITER_OPS},
        "vllm.platforms": {"current_platform": FakePlatform(cpu=True)},
        "vllm.triton_utils": {
            "tl": SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32"),
            "triton": SimpleNamespace(),
        },
        "vllm.model_executor.layers.fused_moe.modular_kernel": {
            "FusedMoEActivationFormat": SimpleNamespace(
                BatchedExperts=object(), Standard=object()
            ),
            "FusedMoEExperts": type("FusedMoEExperts", (), {}),
            "ExpertTokensMetadata": type("ExpertTokensMetadata", (), {}),
        },
        "vllm.model_executor.layers.fused_moe.activation": {
            "MoEActivation": _MoEActivation
        },
        "vllm.model_executor.layers.fused_moe.config": {
            "FusedMoEConfig": type("FusedMoEConfig", (), {})
        },
        "vllm.model_executor.layers.fused_moe.utils": {
            "_resize_cache": lambda t, shape: torch.empty(tuple(shape)),
            "moe_kernel_quantize_input": lambda *a, **k: (a[0], None),
        },
        "vllm.model_executor.layers.fused_moe.fused_moe": {
            "try_get_optimal_moe_config": MagicMock(
                return_value={"BLOCK_SIZE_M": 32}
            )
        },
        "vllm.model_executor.layers.fused_moe.experts.triton_moe": {
            "TritonExperts": _TritonExperts
        },
        "vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe": {
            "rocm_aiter_grouped_topk": MagicMock()
        },
        "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe": {
            "TrtLlmBf16LoRAExperts": type("TrtLlmBf16LoRAExperts", (), {})
        },
        "vllm.model_executor.layers.fused_moe.oracle.unquantized": {
            "UnquantizedMoeBackend": _Backends,
            "_trtllm_bf16_lora_supported": MagicMock(return_value=False),
            "backend_to_kernel_cls": MagicMock(return_value=[_fake_kernel(True)]),
            "map_unquantized_backend": MagicMock(return_value=_Backends.TRITON),
        },
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router": {
            "FusedTopKBiasRouter": type("FusedTopKBiasRouter", (), {}),
            "fused_topk_bias": MagicMock(),
        },
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router": {
            "FusedTopKRouter": type("FusedTopKRouter", (), {}),
            "_get_padding_mask": MagicMock(return_value=None),
            "fused_topk": MagicMock(),
        },
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router": {
            "GroupedTopKRouter": type("GroupedTopKRouter", (), {})
        },
        # deliberately lacks FlashinferMoeBackend -> forces the shim branch
        "vllm.model_executor.layers.quantization.utils.flashinfer_utils": {},
        "vllm_fl.dispatch": {
            "CachedOp": _CachedOp,
            "call_op": MagicMock(),
            "resolve_op": MagicMock(),
        },
        "vllm_fl.ops.fused_moe.activation": {
            "apply_moe_activation": MagicMock()
        },
        "vllm_fl.utils": {
            "get_oot_blacklist": MagicMock(return_value=None),
            "use_flaggems": MagicMock(return_value=False),
        },
        "vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe": {
            "fused_topk": MagicMock()
        },
        "flaggems_vllm": {"fused_experts_impl": MagicMock()},
    }

    for name in _PKGS | set(leaf_attrs):
        _new(name, leaf_attrs.get(name))

    for name, mod in list(mods.items()):
        if "." in name:
            parent, child = name.rsplit(".", 1)
            if parent in mods:
                setattr(mods[parent], child, mod)
    return mods


STUB = _build_stubs()
STUB_NAMES = list(STUB)


@contextlib.contextmanager
def _installed_stubs():
    saved = {}
    try:
        for name, mod in STUB.items():
            saved[name] = sys.modules.get(name, _MISSING)
            sys.modules[name] = mod
        yield
    finally:
        for name, prev in saved.items():
            if prev is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def _load(target_name, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(target_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[target_name] = module
    spec.loader.exec_module(module)
    return module


with _installed_stubs():
    MOD = _load("covfl_fused_moe_utils", _UTILS_PATH)
    ROUTER = _load("covfl_fused_moe_router", _ROUTER_PATH)


@pytest.fixture(autouse=True)
def _stubs_present(monkeypatch):
    """Expose the stub tree to lazy imports inside the modules under test."""
    for name, mod in STUB.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield


def _cfg(**kw):
    parallel = SimpleNamespace(
        dp_size=kw.pop("dp_size", 1),
        use_ep=kw.pop("use_ep", False),
        use_batched_activation_format=kw.pop("batched", False),
    )
    base = dict(
        moe_parallel_config=parallel,
        activation=SimpleNamespace(value="silu"),
        is_lora_enabled=kw.pop("lora", False),
        moe_backend=kw.pop("backend", "auto"),
        max_num_tokens=kw.pop("max_num_tokens", 64),
    )
    base.update(kw)
    return SimpleNamespace(**base)


# =========================================================================== #
# fused_moe_utils: FlashinferMoeBackend shim
# =========================================================================== #

def test_flashinfer_shim_semantics():
    shim = MOD.FlashinferMoeBackend
    a = shim("  CUTLASS ")
    assert a.value == "cutlass"
    assert str(a) == "cutlass"
    assert a == shim.CUTLASS
    assert a == SimpleNamespace(value="Cutlass")
    assert hash(a) == hash("cutlass")
    assert a == "cutlass"
    assert not (a == "trtllm")


# =========================================================================== #
# _should_use_fl_triton_experts
# =========================================================================== #

def test_should_use_fl_triton_experts_all_true(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cuda=True))
    monkeypatch.setattr(MOD, "use_flaggems", lambda: True)
    monkeypatch.setattr(MOD, "get_oot_blacklist", lambda: None)
    assert MOD._should_use_fl_triton_experts() is True


def test_should_use_fl_triton_experts_out_of_tree(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(oot=True))
    monkeypatch.setattr(MOD, "use_flaggems", lambda: True)
    monkeypatch.setattr(MOD, "get_oot_blacklist", lambda: [])
    assert MOD._should_use_fl_triton_experts() is True


@pytest.mark.parametrize(
    "platform,flaggems,blacklist",
    [
        (FakePlatform(cuda=True), False, None),
        (FakePlatform(cuda=True), True, ["fused_moe"]),
        (FakePlatform(cpu=True), True, None),
        (FakePlatform(), True, None),
    ],
)
def test_should_use_fl_triton_experts_false(monkeypatch, platform, flaggems, blacklist):
    monkeypatch.setattr(MOD, "current_platform", platform)
    monkeypatch.setattr(MOD, "use_flaggems", lambda: flaggems)
    monkeypatch.setattr(MOD, "get_oot_blacklist", lambda: blacklist)
    assert MOD._should_use_fl_triton_experts() is False


# =========================================================================== #
# _get_priority_backends
# =========================================================================== #

def test_priority_backends_rocm(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(rocm=True))
    got = MOD._get_priority_backends(_cfg())
    assert got[0] is _Backends.AITER
    assert got[-1] is _Backends.BATCHED_TRITON


def test_priority_backends_cuda_default(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cuda=True))
    got = MOD._get_priority_backends(_cfg())
    assert got[0] is _Backends.FLASHINFER_TRTLLM
    assert _Backends.TRITON in got


def test_priority_backends_cuda_hopper_reorder(monkeypatch):
    monkeypatch.setattr(
        MOD, "current_platform", FakePlatform(cuda=True, family=90)
    )
    got = MOD._get_priority_backends(_cfg())
    assert got[-2] is _Backends.FLASHINFER_TRTLLM
    assert got[-1] is _Backends.FLASHINFER_CUTLASS


def test_priority_backends_cuda_dp_reorder(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cuda=True))
    got = MOD._get_priority_backends(_cfg(dp_size=2))
    assert got[-1] is _Backends.FLASHINFER_CUTLASS


def test_priority_backends_cuda_swigluoai_reorder(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cuda=True))
    cfg = _cfg()
    cfg.activation = _MoEActivation.SWIGLUOAI
    got = MOD._get_priority_backends(cfg)
    assert got[-1] is _Backends.FLASHINFER_CUTLASS


def test_priority_backends_xpu_cpu_iluvatar(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(xpu=True))
    assert MOD._get_priority_backends(_cfg()) == [_Backends.XPU]

    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    assert MOD._get_priority_backends(_cfg()) == [_Backends.CPU]

    monkeypatch.setattr(
        MOD, "current_platform", FakePlatform(vendor="iluvatar")
    )
    assert MOD._get_priority_backends(_cfg()) == [
        _Backends.TRITON,
        _Backends.BATCHED_TRITON,
    ]


# =========================================================================== #
# select_unquantized_moe_backend_oot
# =========================================================================== #

def test_select_tpu(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(tpu=True))
    assert MOD.select_unquantized_moe_backend_oot(_cfg()) == (_Backends.TPU, None)


def test_select_npu_raises_for_lora_ep_dp_batched(monkeypatch):
    monkeypatch.setattr(
        MOD, "current_platform", FakePlatform(device_type="npu")
    )
    for cfg in (
        _cfg(lora=True),
        _cfg(use_ep=True),
        _cfg(dp_size=2),
        _cfg(batched=True),
    ):
        with pytest.raises(NotImplementedError, match="Ascend grouped MoE"):
            MOD.select_unquantized_moe_backend_oot(cfg)


def test_select_npu_explicit_backend(monkeypatch):
    monkeypatch.setattr(
        MOD, "current_platform", FakePlatform(device_type="npu")
    )
    with pytest.raises(NotImplementedError, match="Unsupported explicit Ascend"):
        MOD.select_unquantized_moe_backend_oot(_cfg(backend="flashinfer"))
    backend, cls = MOD.select_unquantized_moe_backend_oot(_cfg(backend="triton"))
    assert backend is _Backends.TRITON
    assert cls is MOD.TritonExpertsFL


def test_select_fl_triton_experts(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cuda=True))
    monkeypatch.setattr(MOD, "use_flaggems", lambda: True)
    monkeypatch.setattr(MOD, "get_oot_blacklist", lambda: None)
    backend, cls = MOD.select_unquantized_moe_backend_oot(_cfg())
    assert backend is _Backends.TRITON
    assert cls is MOD.TritonExpertsFL


def test_select_lora_trtllm(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(MOD, "_trtllm_bf16_lora_supported", lambda c: True)
    backend, cls = MOD.select_unquantized_moe_backend_oot(_cfg(lora=True))
    assert backend is _Backends.FLASHINFER_TRTLLM
    assert cls is STUB[
        "vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe"
    ].TrtLlmBf16LoRAExperts


def test_select_lora_triton(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(MOD, "_trtllm_bf16_lora_supported", lambda c: False)
    kernel = _fake_kernel(True)
    monkeypatch.setattr(MOD, "backend_to_kernel_cls", lambda b: [kernel])
    backend, cls = MOD.select_unquantized_moe_backend_oot(_cfg(lora=True))
    assert backend is _Backends.TRITON
    assert cls is kernel


def test_select_explicit_backend_success(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(
        MOD, "map_unquantized_backend", lambda b: _Backends.TRITON
    )
    kernel = _fake_kernel(True)
    monkeypatch.setattr(MOD, "backend_to_kernel_cls", lambda b: [kernel])
    backend, cls = MOD.select_unquantized_moe_backend_oot(_cfg(backend="triton"))
    assert backend is _Backends.TRITON and cls is kernel


def test_select_explicit_backend_batched_promotion(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(
        MOD, "map_unquantized_backend", lambda b: _Backends.TRITON
    )
    seen = {}

    def b2k(b):
        seen["backend"] = b
        return [_fake_kernel(True)]

    monkeypatch.setattr(MOD, "backend_to_kernel_cls", b2k)
    backend, _ = MOD.select_unquantized_moe_backend_oot(
        _cfg(backend="triton", batched=True)
    )
    assert seen["backend"] is _Backends.BATCHED_TRITON
    assert backend is _Backends.BATCHED_TRITON


def test_select_explicit_backend_raise_with_reason(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(
        MOD, "map_unquantized_backend", lambda b: _Backends.TRITON
    )
    monkeypatch.setattr(
        MOD, "backend_to_kernel_cls", lambda b: [_fake_kernel(False, "too big")]
    )
    with pytest.raises(ValueError, match="since too big"):
        MOD.select_unquantized_moe_backend_oot(_cfg(backend="triton"))


def test_select_explicit_backend_raise_no_reason(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(
        MOD, "map_unquantized_backend", lambda b: _Backends.TRITON
    )
    monkeypatch.setattr(
        MOD, "backend_to_kernel_cls", lambda b: [_fake_kernel(False, None)]
    )
    with pytest.raises(ValueError, match="does not support the deployment"):
        MOD.select_unquantized_moe_backend_oot(_cfg(backend="triton"))


def test_select_aiter_skipped_removes_backend(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(rocm=True))
    monkeypatch.setattr(
        MOD, "envs", SimpleNamespace(
            is_set=lambda n: True,
            VLLM_ROCM_USE_AITER=False,
            VLLM_ROCM_USE_AITER_MOE=True,
        )
    )
    removed = _Backends.AITER
    calls = {"n": 0}

    def b2k(b):
        calls["n"] += 1
        assert b is not removed
        return [_fake_kernel(True)]

    monkeypatch.setattr(MOD, "backend_to_kernel_cls", b2k)
    backend, _ = MOD.select_unquantized_moe_backend_oot(_cfg())
    assert backend is _Backends.TRITON
    assert calls["n"] >= 1


def test_select_aiter_enabled_returns_aiter(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(rocm=True))
    monkeypatch.setattr(
        MOD, "envs", SimpleNamespace(
            is_set=lambda n: True,
            VLLM_ROCM_USE_AITER=True,
            VLLM_ROCM_USE_AITER_MOE=True,
        )
    )
    aiter_ops = SimpleNamespace(is_rdna_aiter_enabled=lambda: False)
    monkeypatch.setattr(MOD, "rocm_aiter_ops", aiter_ops)
    monkeypatch.setattr(MOD, "backend_to_kernel_cls", lambda b: [_fake_kernel(True)])
    backend, _ = MOD.select_unquantized_moe_backend_oot(_cfg())
    assert backend is _Backends.AITER


def test_select_auto_loop_supported_after_unsupported(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cuda=True))
    monkeypatch.setattr(MOD, "envs", SimpleNamespace(is_set=lambda n: False))
    order = list(_Backends.__dict__.values())

    def b2k(b):
        if b is _Backends.FLASHINFER_TRTLLM:
            return [_fake_kernel(False, "no trtllm"), _fake_kernel(True)]
        return [_fake_kernel(True)]

    monkeypatch.setattr(MOD, "backend_to_kernel_cls", b2k)
    backend, _ = MOD.select_unquantized_moe_backend_oot(_cfg())
    assert backend is _Backends.FLASHINFER_TRTLLM


def test_select_auto_loop_none_supported(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(MOD, "envs", SimpleNamespace(is_set=lambda n: False))
    monkeypatch.setattr(
        MOD, "backend_to_kernel_cls", lambda b: [_fake_kernel(False, None)]
    )
    with pytest.raises(
        NotImplementedError, match="No Unquantized MoE backend supports"
    ):
        MOD.select_unquantized_moe_backend_oot(_cfg())


# =========================================================================== #
# _prepare_expert_assignment
# =========================================================================== #

def test_prepare_assignment_naive(monkeypatch):
    topk_ids = torch.tensor([[0, 1], [2, 0]], dtype=torch.int64)
    cfg = {"BLOCK_SIZE_M": 8}
    sorted_ids, ids, padded = MOD._prepare_expert_assignment(
        topk_ids, cfg, num_tokens=2, top_k_num=2, global_num_experts=64,
        expert_map=None,
    )
    assert sorted_ids is None
    assert ids.shape == (4,)
    assert padded.item() == 4 * 8


def test_prepare_assignment_align_due_to_expert_map(monkeypatch):
    align = MagicMock(return_value=("s", "e", "p"))
    monkeypatch.setattr(MOD, "_moe_align_block_size", align)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    got = MOD._prepare_expert_assignment(
        topk_ids, {"BLOCK_SIZE_M": 8}, 1, 2, 64,
        expert_map=torch.arange(64),
    )
    assert got == ("s", "e", "p")
    align.assert_called_once()


def test_prepare_assignment_align_due_to_many_tokens(monkeypatch):
    align = MagicMock(return_value=("s", "e", "p"))
    monkeypatch.setattr(MOD, "_moe_align_block_size", align)
    topk_ids = torch.zeros((16, 2), dtype=torch.int64)
    MOD._prepare_expert_assignment(topk_ids, {"BLOCK_SIZE_M": 8}, 16, 2, 64, None)
    align.assert_called_once()


def test_prepare_assignment_align_due_to_int8_block(monkeypatch):
    align = MagicMock(return_value=("s", "e", "p"))
    monkeypatch.setattr(MOD, "_moe_align_block_size", align)
    topk_ids = torch.zeros((1, 2), dtype=torch.int64)
    MOD._prepare_expert_assignment(
        topk_ids, {"BLOCK_SIZE_M": 8}, 1, 2, 64, None,
        use_int8_w8a16=True, block_shape=[128, 128],
        ignore_invalid_experts=True,
    )
    align.assert_called_once()
    assert align.call_args.kwargs["ignore_invalid_experts"] is True


def test_prepare_assignment_int4_none_block_stays_naive(monkeypatch):
    topk_ids = torch.zeros((1, 2), dtype=torch.int64)
    sorted_ids, _, _ = MOD._prepare_expert_assignment(
        topk_ids, {"BLOCK_SIZE_M": 8}, 1, 2, 64, None,
        use_int4_w4a16=True, block_shape=None,
    )
    assert sorted_ids is None


# =========================================================================== #
# TritonExpertsFL.workspace_shapes / moe_sum
# =========================================================================== #

def test_workspace_shapes_non_npu(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(device_type="cpu"))
    experts = object.__new__(MOD.TritonExpertsFL)
    experts.moe_config = SimpleNamespace(max_num_tokens=16)
    w13, w2, out = experts.workspace_shapes(4, 8, 6, 2, 3, 3, None, "silu")
    assert w13 == (4, 6)


def test_workspace_shapes_npu_expands(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(device_type="npu"))
    experts = object.__new__(MOD.TritonExpertsFL)
    experts.moe_config = SimpleNamespace(max_num_tokens=16)
    w13, w2, out = experts.workspace_shapes(4, 8, 6, 2, 3, 3, None, "silu")
    assert w13[0] == 16 and w2[0] == 16


def test_moe_sum_delegates(monkeypatch):
    op = MagicMock()
    monkeypatch.setattr(MOD, "_moe_sum", op)
    experts = object.__new__(MOD.TritonExpertsFL)
    a, b = torch.zeros(2, 2), torch.zeros(2, 2)
    experts.moe_sum(a, b)
    op.assert_called_once_with(a, b)


# =========================================================================== #
# TritonExpertsFL.apply -- early dispatch paths
# =========================================================================== #

def _quant_config(**over):
    base = dict(
        quant_dtype=None,
        weight_quant_dtype=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config_name=lambda dt: "cfg",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _bare_experts(quant_config=None):
    experts = object.__new__(MOD.TritonExpertsFL)
    experts._lora_context = None
    experts.quant_config = quant_config or _quant_config()
    experts.w1_bias = None
    experts.w2_bias = None
    experts.activation_config = SimpleNamespace(clamp_limit=None)
    experts.w1_scale = None
    experts.w2_scale = None
    experts.block_shape = None
    experts.per_act_token_quant = False
    experts.quant_dtype = None
    experts.quantization_emulation = False
    return experts


def test_apply_npu_grouped_path(monkeypatch):
    call_op = MagicMock(return_value=torch.zeros(4, 6))
    monkeypatch.setattr(MOD, "call_op", call_op)
    experts = _bare_experts()
    out = torch.empty(4, 6)
    fake_hidden = SimpleNamespace(device=SimpleNamespace(type="npu"))
    experts.apply(
        out, fake_hidden, object(), object(), torch.zeros(4, 2),
        torch.zeros(4, 2, dtype=torch.int64), SimpleNamespace(value="silu"),
        3, None, None, None, torch.empty(1), torch.empty(1), None, False,
    )
    assert torch.equal(out, torch.zeros(4, 6))
    call_op.assert_called_once()
    assert call_op.call_args.args[0] == "grouped_moe_experts"


def test_apply_kunlunxin_path(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(vendor="kunlunxin"))
    resolve_op = MagicMock(return_value=torch.zeros(4, 6))
    monkeypatch.setattr(MOD, "resolve_op", resolve_op)
    experts = _bare_experts()
    out = torch.empty(4, 6)
    experts.apply(
        out, torch.zeros(4, 8), torch.zeros(2, 12, 8), torch.zeros(2, 6, 8),
        torch.zeros(4, 2), torch.zeros(4, 2, dtype=torch.int64),
        SimpleNamespace(value="silu"), 3, None, None, None,
        torch.empty(1), torch.empty(1), None, False,
    )
    resolve_op.assert_called_once_with("fused_experts")
    assert torch.equal(out, torch.zeros(4, 6))


def test_apply_cuda_flaggems_path(monkeypatch):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cuda=True))
    impl = STUB["flaggems_vllm"].fused_experts_impl
    impl.reset_mock()
    impl.return_value = torch.zeros(4, 6)
    experts = _bare_experts()
    out = torch.empty(4, 6)
    experts.apply(
        out, torch.zeros(4, 8), torch.zeros(2, 12, 8), torch.zeros(2, 6, 8),
        torch.zeros(4, 2), torch.zeros(4, 2, dtype=torch.int64),
        SimpleNamespace(value="silu"), 3, None, None, None,
        torch.empty(1), torch.empty(1), None, False,
    )
    impl.assert_called_once()
    assert torch.equal(out, torch.zeros(4, 6))


# =========================================================================== #
# TritonExpertsFL.apply -- LoRA/step-by-step pipeline
# =========================================================================== #

def _install_apply_mocks(monkeypatch, lora=False):
    monkeypatch.setattr(MOD, "current_platform", FakePlatform(cpu=True))
    monkeypatch.setattr(
        MOD, "_resize_cache", lambda t, shape: torch.empty(tuple(shape))
    )
    monkeypatch.setattr(
        MOD, "moe_kernel_quantize_input", lambda *a, **k: (a[0], None)
    )
    monkeypatch.setattr(MOD, "try_get_optimal_moe_config", lambda *a, **k: {"BLOCK_SIZE_M": 8})
    monkeypatch.setattr(MOD, "apply_moe_activation", MagicMock())
    invoke = MagicMock()
    monkeypatch.setattr(MOD, "_invoke_fused_moe_triton_kernel", invoke)
    moe_sum_op = MagicMock()
    monkeypatch.setattr(MOD, "_moe_sum", moe_sum_op)
    return invoke, moe_sum_op


def _run_pipeline(dtype, *, int4=False, global_experts=64, lora=False,
                  workspace_expert_map=None):
    E, num_tokens, N, K, topk = 2, 4, 6, 8, 2
    last = K // 2 if int4 else K
    make = torch.zeros if dtype in (torch.float8_e4m3fn, torch.int8) else torch.randn
    hidden = make(num_tokens, K, dtype=dtype)
    w1 = make(E, 2 * N, last, dtype=dtype)
    w2 = make(E, N, last, dtype=dtype)
    topk_ids = torch.zeros(num_tokens, topk, dtype=torch.int64)
    topk_weights = torch.zeros(num_tokens, topk, dtype=torch.float32)
    out = torch.zeros(num_tokens, K, dtype=dtype)

    experts = _bare_experts(_quant_config(use_int4_w4a16=int4))
    if lora:
        experts._lora_context = MagicMock()
        experts.apply_w13_lora = MagicMock(return_value=(None, None, None, None))
        experts.apply_w2_lora = MagicMock()
    experts.moe_problem_size = lambda h, a, b, c: (E, num_tokens, N, K, topk)
    experts.adjust_N_for_activation = lambda n, act: n

    experts.apply(
        out, hidden, w1, w2, topk_weights, topk_ids,
        SimpleNamespace(value="silu"), global_experts,
        workspace_expert_map, None, None, torch.empty(1), torch.empty(1),
        None, False,
    )
    return experts


def test_apply_pipeline_basic(monkeypatch):
    invoke, moe_sum_op = _install_apply_mocks(monkeypatch)
    experts = _run_pipeline(torch.float16)
    assert invoke.call_count == 2
    moe_sum_op.assert_called_once()
    assert experts._lora_context is None


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.float32, torch.bfloat16]
)
def test_apply_pipeline_compute_types(monkeypatch, dtype):
    invoke, _ = _install_apply_mocks(monkeypatch)
    _run_pipeline(dtype)
    assert invoke.call_count == 2


def test_apply_pipeline_float8(monkeypatch):
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no float8 dtype")
    invoke, _ = _install_apply_mocks(monkeypatch)
    try:
        _run_pipeline(torch.float8_e4m3fn)
    except RuntimeError:
        pytest.skip("float8 tensors unsupported on this CPU build")
    assert invoke.call_count == 2


def test_apply_pipeline_int8(monkeypatch):
    invoke, _ = _install_apply_mocks(monkeypatch)
    _run_pipeline(torch.int8)
    assert invoke.call_count == 2


def test_apply_pipeline_int4_assert(monkeypatch):
    _install_apply_mocks(monkeypatch)
    experts = _run_pipeline(torch.float16, int4=True)
    assert experts.quant_config.use_int4_w4a16 is True


def test_apply_pipeline_global_experts_minus_one(monkeypatch):
    align = MagicMock(return_value=(None, torch.zeros(1, dtype=torch.int64),
                                    torch.zeros(1, dtype=torch.int32)))
    monkeypatch.setattr(MOD, "_moe_align_block_size", align)
    invoke, _ = _install_apply_mocks(monkeypatch)
    _run_pipeline(torch.float16, global_experts=-1)
    assert invoke.call_count == 2
    align.assert_called_once()


def test_apply_pipeline_with_expert_map_align(monkeypatch):
    align = MagicMock(return_value=(None, torch.zeros(1, dtype=torch.int64),
                                    torch.zeros(1, dtype=torch.int32)))
    monkeypatch.setattr(MOD, "_moe_align_block_size", align)
    invoke, _ = _install_apply_mocks(monkeypatch)
    _run_pipeline(torch.float16, workspace_expert_map=torch.arange(64))
    align.assert_called_once()
    assert invoke.call_count == 2


def test_apply_pipeline_lora(monkeypatch):
    invoke, moe_sum_op = _install_apply_mocks(monkeypatch)
    experts = _run_pipeline(torch.float16, lora=True)
    experts.apply_w13_lora.assert_called_once()
    experts.apply_w2_lora.assert_called_once()
    moe_sum_op.assert_called_once()
    assert invoke.call_count == 2


def test_apply_pipeline_bad_dtype_asserts(monkeypatch):
    _install_apply_mocks(monkeypatch)
    with pytest.raises(AssertionError):
        _run_pipeline(torch.complex64)


# =========================================================================== #
# router.py -- helpers
# =========================================================================== #

def test_is_ascend_npu_tensor():
    assert ROUTER._is_ascend_npu_tensor(torch.zeros(2)) is False
    fake = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert ROUTER._is_ascend_npu_tensor(fake) is True


def test_sqrtsoftplus_topk_basic():
    gating = torch.randn(4, 6)
    w, ids = ROUTER._sqrtsoftplus_topk(
        gating, None, 2, False, None, None, None, 1.0
    )
    assert w.shape == (4, 2) and ids.shape == (4, 2)
    assert ids.dtype == torch.int32


def test_sqrtsoftplus_topk_bias_and_renorm_and_dtype():
    gating = torch.randn(3, 5)
    bias = torch.randn(5)
    w, ids = ROUTER._sqrtsoftplus_topk(
        gating, bias, 2, True, torch.int64, None, None, 1.7
    )
    assert ids.dtype == torch.int64
    assert w.dtype == torch.float32


def test_sqrtsoftplus_topk_hash_table_dtype_cast():
    gating = torch.randn(4, 6)
    table = torch.zeros(10, 2, dtype=torch.int64)
    tokens = torch.tensor([1.0, 2.0, 3.0, 4.0])  # float -> cast to int64
    w, ids = ROUTER._sqrtsoftplus_topk(
        gating, None, 2, False, None, tokens, table, 1.0
    )
    assert ids.shape == (4, 2)
    assert ids.dtype == torch.int32


def test_sqrtsoftplus_topk_hash_requires_tokens():
    gating = torch.randn(2, 4)
    table = torch.zeros(5, 2, dtype=torch.int64)
    with pytest.raises(AssertionError):
        ROUTER._sqrtsoftplus_topk(
            gating, None, 2, False, None, None, table, 1.0
        )


def test_has_native_moe_topk(monkeypatch):
    monkeypatch.setattr(
        ROUTER, "torch", SimpleNamespace(ops=SimpleNamespace())
    )
    assert ROUTER._has_native_moe_topk() is False
    monkeypatch.setattr(
        ROUTER, "torch",
        SimpleNamespace(ops=SimpleNamespace(_moe_C=SimpleNamespace(
            topk_softmax=object()))),
    )
    assert ROUTER._has_native_moe_topk() is True


# =========================================================================== #
# router.py -- fused_topk
# =========================================================================== #

def test_fused_topk_token_mismatch():
    with pytest.raises(AssertionError, match="Number of tokens"):
        ROUTER.fused_topk(
            torch.zeros(2, 4), torch.zeros(3, 8), 2, False
        )


def test_fused_topk_ascend(monkeypatch):
    ascend = STUB[
        "vllm_fl.dispatch.backends.vendor.ascend.impl.fused_moe"
    ].fused_topk
    ascend.reset_mock()
    ascend.return_value = ("w", "i", "t")
    monkeypatch.setattr(
        ROUTER, "_get_padding_mask", MagicMock(return_value="pm")
    )
    fake_gating = SimpleNamespace(
        device=SimpleNamespace(type="npu"), size=lambda *a: 2
    )
    out = ROUTER.fused_topk(torch.zeros(2, 4), fake_gating, 2, True)
    assert out == ("w", "i", "t")
    ascend.assert_called_once()
    assert ascend.call_args.args[-1] == "pm"


def test_fused_topk_non_softmax_uses_upstream(monkeypatch):
    upstream = MagicMock(return_value=("u",))
    monkeypatch.setattr(ROUTER, "upstream_fused_topk", upstream)
    monkeypatch.setattr(ROUTER, "_get_padding_mask", MagicMock(return_value=None))
    out = ROUTER.fused_topk(
        torch.zeros(2, 4), torch.zeros(2, 8), 2, False, scoring_func="sigmoid"
    )
    assert out == ("u",)
    upstream.assert_called_once()


def test_fused_topk_padding_and_native_uses_upstream(monkeypatch):
    upstream = MagicMock(return_value=("u",))
    monkeypatch.setattr(ROUTER, "upstream_fused_topk", upstream)
    monkeypatch.setattr(
        ROUTER, "_get_padding_mask",
        MagicMock(return_value=torch.tensor([False, True])),
    )
    monkeypatch.setattr(ROUTER, "_has_native_moe_topk", lambda: True)
    out = ROUTER.fused_topk(torch.zeros(2, 4), torch.zeros(2, 8), 2, False)
    assert out == ("u",)
    upstream.assert_called_once()


def test_fused_topk_default_softmax(monkeypatch):
    monkeypatch.setattr(ROUTER, "_get_padding_mask", MagicMock(return_value=None))
    tw = torch.zeros(3, 2)
    ti = torch.zeros(3, 2, dtype=torch.int32)
    topk_op = MagicMock(return_value=(tw, ti))
    monkeypatch.setattr(ROUTER, "_topk_softmax", topk_op)
    w, ids, tei = ROUTER.fused_topk(
        torch.zeros(3, 4), torch.randn(3, 8), 2, True
    )
    assert w is tw and ids is ti
    assert tei.shape == (3, 2)
    topk_op.assert_called_once()


def test_fused_topk_default_indices_type(monkeypatch):
    monkeypatch.setattr(ROUTER, "_get_padding_mask", MagicMock(return_value=None))
    tw = torch.zeros(3, 2)
    ti = torch.zeros(3, 2, dtype=torch.int64)
    monkeypatch.setattr(ROUTER, "_topk_softmax", MagicMock(return_value=(tw, ti)))
    _, ids, _ = ROUTER.fused_topk(
        torch.zeros(3, 4), torch.randn(3, 8), 2, False,
        indices_type=torch.int64,
    )
    assert ids.dtype == torch.int64


def test_fused_topk_padding_mask_applied(monkeypatch):
    pm = torch.tensor([False, True, False])
    monkeypatch.setattr(ROUTER, "_get_padding_mask", MagicMock(return_value=pm))
    monkeypatch.setattr(ROUTER, "_has_native_moe_topk", lambda: False)
    tw = torch.ones(3, 2)
    ti = torch.zeros(3, 2, dtype=torch.int32)
    monkeypatch.setattr(ROUTER, "_topk_softmax", MagicMock(return_value=(tw, ti)))
    w, ids, _ = ROUTER.fused_topk(
        torch.zeros(3, 4), torch.randn(3, 8), 2, False
    )
    assert w[1].tolist() == [0.0, 0.0]
    assert ids[1].tolist() == [-1, -1]


# =========================================================================== #
# router.py -- FusedTopKRouterFL
# =========================================================================== #

def test_fused_topk_router_compute_routing(monkeypatch):
    fake = MagicMock(return_value=("w", "i", "t"))
    monkeypatch.setattr(ROUTER, "fused_topk", fake)
    router = object.__new__(ROUTER.FusedTopKRouterFL)
    router.top_k = 2
    router.renormalize = True
    router.scoring_func = "softmax"
    w, i = router._compute_routing(torch.zeros(2, 4), torch.randn(2, 8),
                                   torch.int32)
    assert (w, i) == ("w", "i")
    fake.assert_called_once()


# =========================================================================== #
# router.py -- _fl_grouped_topk
# =========================================================================== #

def test_fl_grouped_topk_bias_sigmoid(monkeypatch):
    op = MagicMock(return_value=("v", "i"))
    monkeypatch.setattr(ROUTER, "_grouped_topk", op)
    out = ROUTER._fl_grouped_topk(
        torch.zeros(2, 4), torch.randn(2, 8), 2, True, 2, 1,
        "sigmoid", 1.5, torch.zeros(8),
    )
    assert out == ("v", "i")
    assert op.call_args.args[-1] == 1


def test_fl_grouped_topk_bias_softmax(monkeypatch):
    op = MagicMock(return_value=("v", "i"))
    monkeypatch.setattr(ROUTER, "_grouped_topk", op)
    out = ROUTER._fl_grouped_topk(
        torch.zeros(2, 4), torch.randn(2, 8), 2, True, 2, 1,
        "softmax", 1.0, torch.zeros(8),
    )
    assert out == ("v", "i")
    assert op.call_args.args[-1] == 0


def test_fl_grouped_topk_bias_unsupported():
    with pytest.raises(ValueError, match="Unsupported scoring"):
        ROUTER._fl_grouped_topk(
            torch.zeros(2, 4), torch.randn(2, 8), 2, True, 2, 1,
            "relu", 1.0, torch.zeros(8),
        )


def test_fl_grouped_topk_token_mismatch():
    with pytest.raises(AssertionError, match="Number of tokens"):
        ROUTER._fl_grouped_topk(
            torch.zeros(2, 4), torch.randn(3, 8), 2, True, 2, 1,
            "softmax", 1.0, torch.zeros(8),
        )


def test_fl_grouped_topk_no_bias_softmax():
    w, ids = ROUTER._fl_grouped_topk(
        torch.zeros(4, 4), torch.randn(4, 8), 2, True, 2, 1,
        "softmax", 2.0,
    )
    assert w.shape == (4, 2) and ids.shape == (4, 2)
    assert w.dtype == torch.float32 and ids.dtype == torch.int32


def test_fl_grouped_topk_no_bias_sigmoid_identity_scale():
    w, ids = ROUTER._fl_grouped_topk(
        torch.zeros(3, 4), torch.randn(3, 8), 2, False, 2, 1,
        "sigmoid", 1.0,
    )
    assert w.shape == (3, 2)


def test_fl_grouped_topk_no_bias_unsupported():
    with pytest.raises(ValueError, match="Unsupported scoring"):
        ROUTER._fl_grouped_topk(
            torch.zeros(2, 4), torch.randn(2, 8), 2, False, 2, 1,
            "gelu", 1.0,
        )


# =========================================================================== #
# router.py -- GroupedTopKRouterFL
# =========================================================================== #

def _grouped_router():
    r = object.__new__(ROUTER.GroupedTopKRouterFL)
    r.num_expert_group = 2
    r.topk_group = 1
    r.top_k = 2
    r.renormalize = True
    r.scoring_func = "softmax"
    r.routed_scaling_factor = 1.0
    r.e_score_correction_bias = None
    r.num_fused_shared_experts = 0
    return r


def test_grouped_router_invalid_grouping_with_bias(monkeypatch):
    bias = MagicMock(
        return_value=(torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.int32))
    )
    monkeypatch.setattr(ROUTER, "fused_topk_bias", bias)
    monkeypatch.setattr(ROUTER, "_fl_grouped_topk", MagicMock())
    r = _grouped_router()
    r.num_expert_group = 4
    r.e_score_correction_bias = SimpleNamespace(data=torch.zeros(4))
    r.routed_scaling_factor = 2.0
    # invalid: num_experts (4) <= num_expert_group (4)
    logits = torch.randn(3, 4)
    w, i = r._compute_routing(torch.zeros(3, 4), logits, None)
    assert w.shape == (3, 2) and i.shape == (3, 2)
    assert torch.all(w == 2.0)
    bias.assert_called_once()


def test_grouped_router_invalid_grouping_without_bias(monkeypatch):
    fake = MagicMock(return_value=("w", "i", "tei"))
    monkeypatch.setattr(ROUTER, "fused_topk", fake)
    r = _grouped_router()
    r.num_expert_group = 3  # 8 % 3 != 0
    w, i = r._compute_routing(torch.zeros(3, 4), torch.randn(3, 8), None)
    assert (w, i) == ("w", "i")
    fake.assert_called_once()


def test_grouped_router_aiter_path(monkeypatch):
    aiter_topk = MagicMock(return_value=("w", "i"))
    monkeypatch.setattr(ROUTER, "rocm_aiter_grouped_topk", aiter_topk)
    ops = SimpleNamespace(
        is_fused_moe_enabled=lambda: True,
        is_fusion_moe_shared_experts_enabled=lambda: True,
    )
    monkeypatch.setattr(ROUTER, "rocm_aiter_ops", ops)
    r = _grouped_router()
    r.num_fused_shared_experts = 1
    w, i = r._compute_routing(torch.zeros(3, 4), torch.randn(3, 8), None)
    assert (w, i) == ("w", "i")
    aiter_topk.assert_called_once()


def test_grouped_router_aiter_shared_assert(monkeypatch):
    ops = SimpleNamespace(
        is_fused_moe_enabled=lambda: True,
        is_fusion_moe_shared_experts_enabled=lambda: False,
    )
    monkeypatch.setattr(ROUTER, "rocm_aiter_ops", ops)
    r = _grouped_router()
    r.num_fused_shared_experts = 1
    with pytest.raises(AssertionError):
        r._compute_routing(torch.zeros(3, 4), torch.randn(3, 8), None)


def test_grouped_router_pure_torch_fallback(monkeypatch):
    monkeypatch.setattr(
        ROUTER, "rocm_aiter_ops",
        SimpleNamespace(is_fused_moe_enabled=lambda: False),
    )
    fake = MagicMock(return_value=("w", "i"))
    monkeypatch.setattr(ROUTER, "_fl_grouped_topk", fake)
    r = _grouped_router()
    w, i = r._compute_routing(torch.zeros(3, 4), torch.randn(3, 8), None)
    assert (w, i) == ("w", "i")
    fake.assert_called_once()


# =========================================================================== #
# router.py -- FusedTopKBiasRouterFL
# =========================================================================== #

def _bias_router(*, scoring="softmax", n_shared=0):
    r = object.__new__(ROUTER.FusedTopKBiasRouterFL)
    r.scoring_func = scoring
    r.top_k = 2
    r.renormalize = True
    r.routed_scaling_factor = 1.0
    r.e_score_correction_bias = SimpleNamespace(data=torch.zeros(8))
    r._hash_indices_table = None
    r.num_fused_shared_experts = n_shared
    r.global_num_experts = 8
    r.shared_expert_weight = 0.5
    r.bias_vl = None
    r.image_sentinel_lo = None
    return r


def test_bias_router_sqrtsoftplus_npu(monkeypatch):
    sqrt_op = MagicMock(return_value=("w", "i"))
    monkeypatch.setattr(ROUTER, "_sqrtsoftplus_topk", sqrt_op)
    r = _bias_router(scoring="sqrtsoftplus")
    logits = SimpleNamespace(device=SimpleNamespace(type="npu"))
    w, i = r._compute_routing(torch.zeros(3, 4), logits, None)
    assert (w, i) == ("w", "i")
    sqrt_op.assert_called_once()


def test_bias_router_fused_topk_bias_path(monkeypatch):
    fake = MagicMock(return_value=(torch.ones(3, 2), torch.zeros(3, 2, dtype=torch.int32)))
    monkeypatch.setattr(ROUTER, "fused_topk_bias", fake)
    r = _bias_router()
    w, i = r._compute_routing(torch.zeros(3, 4), torch.zeros(3, 8), None)
    assert w.shape == (3, 2)
    fake.assert_called_once()


def test_bias_router_shared_experts_appended(monkeypatch):
    fake = MagicMock(
        return_value=(
            torch.ones(3, 2),
            torch.zeros(3, 2, dtype=torch.int32),
        )
    )
    monkeypatch.setattr(ROUTER, "fused_topk_bias", fake)
    r = _bias_router(n_shared=2)
    w, i = r._compute_routing(torch.zeros(3, 4), torch.zeros(3, 8), None)
    assert w.shape == (3, 4) and i.shape == (3, 4)
    assert i[:, 2].tolist() == [8, 8, 8]
    assert w[:, 2].tolist() == [0.5, 0.5, 0.5]


# =========================================================================== #
# router.py -- replace_router_with_fl
# =========================================================================== #

def test_replace_router_with_fl(monkeypatch):
    base_topk = STUB[
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router"
    ].FusedTopKRouter
    base_grouped = STUB[
        "vllm.model_executor.layers.fused_moe.router.grouped_topk_router"
    ].GroupedTopKRouter
    base_bias = STUB[
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router"
    ].FusedTopKBiasRouter
    saved = {c: getattr(c, "_compute_routing", _MISSING)
             for c in (base_topk, base_grouped, base_bias)}
    try:
        ROUTER.replace_router_with_fl()
        assert base_topk._compute_routing is ROUTER.FusedTopKRouterFL._compute_routing
        assert base_grouped._compute_routing is ROUTER.GroupedTopKRouterFL._compute_routing
        assert base_bias._compute_routing is ROUTER.FusedTopKBiasRouterFL._compute_routing
    finally:
        for cls, prev in saved.items():
            if prev is _MISSING:
                delattr(cls, "_compute_routing")
            else:
                cls._compute_routing = prev
