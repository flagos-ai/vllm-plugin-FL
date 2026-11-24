# Copyright (c) 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage tests for FL patch / attention / compilation / hybrid cache.

No kernel is compiled and no accelerator is touched.  Heavy dependencies are
guarded and every vendor module is replaced with a fake before use.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import logging
import sys
import types
from collections import namedtuple
from importlib.metadata import PackageNotFoundError
from unittest.mock import Mock

import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in the test image
    torch = None

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected in the test image
    np = None


def _try_import(path):
    try:
        return importlib.import_module(path)
    except Exception:
        return None


GDN = _try_import("vllm_fl.patches.gdn_packed_decode")
FLA = _try_import("vllm_fl.patches.fla_module_relocation")
FLAGTREE = _try_import("vllm_fl.patches.flagtree_target_info")
WORKER = _try_import("vllm_fl.patches.worker_symbols_032")
QSA = _try_import("vllm_fl.patches.qsa_cache_hint")
EXP = _try_import("vllm_fl.patches.exponential_compat")
TK = _try_import("vllm_fl.patches.triton_kernel")
ARM = _try_import("vllm_fl.patches.arm_cpu_gdn")
ATTN = _try_import("vllm_fl.attention.utils")
GRAPH = _try_import("vllm_fl.compilation.graph")
GRT = _try_import("vllm_fl.compilation.graph_runtime")
GP = _try_import("vllm_fl.compilation.graph_params")
HYB = _try_import("vllm_fl.kv_cache.ascend.hybrid")


def _need(module):
    return pytest.mark.skipif(module is None, reason="target module not importable")


needs_torch = pytest.mark.skipif(torch is None, reason="torch is unavailable")
needs_numpy = pytest.mark.skipif(np is None, reason="numpy is unavailable")

BD = namedtuple("BD", "num_tokens")


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _install(monkeypatch, mapping):
    for name, module in mapping.items():
        monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# gdn_packed_decode.py
# ---------------------------------------------------------------------------
@_need(GDN)
def test_gdn_kernel_needs_patch_when_already_marked():
    assert GDN._kernel_needs_beta_patch(_ns(_fl_fp32_beta=True)) is False


@_need(GDN)
def test_gdn_kernel_needs_patch_vulnerable_and_fixed(monkeypatch):
    vulnerable = "x = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"
    fixed = "beta = tl.sigmoid(b_val)\n"
    monkeypatch.setattr(GDN.inspect, "getsource", lambda obj: vulnerable)
    assert GDN._kernel_needs_beta_patch(_ns(fn=lambda: None)) is True
    monkeypatch.setattr(GDN.inspect, "getsource", lambda obj: fixed)
    assert GDN._kernel_needs_beta_patch(_ns(fn=lambda: None)) is False


@_need(GDN)
@pytest.mark.parametrize("exc", [OSError("missing"), TypeError("bad")])
def test_gdn_kernel_needs_patch_inspection_failure(monkeypatch, exc):
    def fake_getsource(obj):
        raise exc

    monkeypatch.setattr(GDN.inspect, "getsource", fake_getsource)
    assert GDN._kernel_needs_beta_patch(_ns(fn=lambda: None)) is True


@_need(GDN)
def test_gdn_patch_replaces_and_is_idempotent(monkeypatch):
    vulnerable = "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"

    def original_kernel():
        return None

    target = _ns(fused_recurrent_gated_delta_rule_packed_decode_kernel=original_kernel)
    monkeypatch.setattr(GDN.inspect, "getsource", lambda obj: vulnerable)
    monkeypatch.setattr(GDN.importlib, "import_module", lambda name: target)
    assert GDN.patch_vllm_packed_gdn_beta() is True
    replacement = target.fused_recurrent_gated_delta_rule_packed_decode_kernel
    expected = GDN._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    assert replacement is expected
    assert replacement._fl_fp32_beta is True
    assert GDN.patch_vllm_packed_gdn_beta() is False


@_need(GDN)
def test_gdn_patch_import_error_returns_false(monkeypatch):
    def boom(name):
        raise ImportError("no fla")

    monkeypatch.setattr(GDN.importlib, "import_module", boom)
    assert GDN.patch_vllm_packed_gdn_beta() is False


@_need(GDN)
def test_gdn_patch_missing_symbol_returns_false(monkeypatch):
    monkeypatch.setattr(GDN.importlib, "import_module", lambda name: _ns())
    assert GDN.patch_vllm_packed_gdn_beta() is False


@_need(GDN)
def test_gdn_replacement_is_marked_fp32_beta():
    replacement = GDN._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    assert replacement._fl_fp32_beta is True
    assert "tl.sigmoid(b_val)" in GDN._VULNERABLE_BETA_EXPRESSION


# ---------------------------------------------------------------------------
# fla_module_relocation.py
# ---------------------------------------------------------------------------
@_need(FLA)
def test_fla_module_exists_true_and_false(monkeypatch):
    monkeypatch.setattr(FLA.importlib.util, "find_spec", lambda name: object())
    assert FLA._module_exists("anything") is True
    monkeypatch.setattr(FLA.importlib.util, "find_spec", lambda name: None)
    assert FLA._module_exists("anything") is False


@_need(FLA)
@pytest.mark.parametrize("exc", [ImportError("x"), AttributeError("y"), ValueError("z")])
def test_fla_module_exists_swallows_errors(monkeypatch, exc):
    def boom(name):
        raise exc

    monkeypatch.setattr(FLA.importlib.util, "find_spec", boom)
    assert FLA._module_exists("anything") is False


@_need(FLA)
def test_fla_install_noop_when_old_prefix_present(monkeypatch):
    monkeypatch.setattr(FLA, "_module_exists", lambda name: name == FLA._OLD_PREFIX)
    assert FLA.install() is False


@_need(FLA)
def test_fla_install_noop_when_new_prefix_absent(monkeypatch):
    monkeypatch.setattr(FLA, "_module_exists", lambda name: False)
    assert FLA.install() is False


def _fla_success_env(
    monkeypatch, existing_alias=None, layers_has_fla=False,
    submodule_raises=False, skip_ops=False,
):
    new_pkg = _ns(name="new")
    sub_pkg = _module("fla-fake-ops")
    child = _ns(name="child")
    layers = _ns()
    if layers_has_fla:
        layers.fla = "sentinel-fla"

    present = {FLA._NEW_PREFIX}
    if not skip_ops:
        present.add(FLA._NEW_PREFIX + ".ops")

    monkeypatch.setattr(FLA, "_module_exists", lambda name: name in present)

    def fake_import(name):
        if name == FLA._NEW_PREFIX:
            return new_pkg
        if name == FLA._NEW_PREFIX + ".ops":
            return sub_pkg
        if name == FLA._NEW_PREFIX + ".ops.child":
            if submodule_raises:
                raise RuntimeError("broken child")
            return child
        if name == "vllm.model_executor.layers":
            return layers
        raise ImportError(name)

    monkeypatch.setattr(FLA.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        FLA.pkgutil, "iter_modules", lambda path: iter([_ns(name="child")])
    )
    fake_modules = {}
    if existing_alias is not None:
        fake_modules[FLA._OLD_PREFIX] = existing_alias
    monkeypatch.setattr(FLA, "sys", _ns(modules=fake_modules))
    return new_pkg, sub_pkg, child, layers, fake_modules


@_need(FLA)
def test_fla_install_aliases_and_binds_layers(monkeypatch):
    new_pkg, sub_pkg, child, layers, fake_modules = _fla_success_env(monkeypatch)
    assert FLA.install() is True
    assert fake_modules[FLA._OLD_PREFIX] is new_pkg
    assert fake_modules[FLA._OLD_PREFIX + ".ops"] is sub_pkg
    assert fake_modules[FLA._OLD_PREFIX + ".ops.child"] is child
    assert layers.fla is new_pkg


@_need(FLA)
def test_fla_install_skips_missing_subprefix(monkeypatch):
    new_pkg, _sub, _child, layers, fake_modules = _fla_success_env(
        monkeypatch, skip_ops=True
    )
    assert FLA.install() is True
    assert fake_modules[FLA._OLD_PREFIX] is new_pkg
    assert FLA._OLD_PREFIX + ".ops" not in fake_modules
    assert layers.fla is new_pkg


@_need(FLA)
def test_fla_install_keeps_existing_alias(monkeypatch):
    sentinel = object()
    _new, _sub, _child, _layers, fake_modules = _fla_success_env(
        monkeypatch, existing_alias=sentinel
    )
    assert FLA.install() is True
    assert fake_modules[FLA._OLD_PREFIX] is sentinel


@_need(FLA)
def test_fla_install_skips_failed_submodule(monkeypatch):
    _new, sub_pkg, _child, _layers, fake_modules = _fla_success_env(
        monkeypatch, submodule_raises=True
    )
    assert FLA.install() is True
    assert fake_modules[FLA._OLD_PREFIX + ".ops"] is sub_pkg
    assert FLA._OLD_PREFIX + ".ops.child" not in fake_modules


@_need(FLA)
def test_fla_install_does_not_override_layers_fla(monkeypatch):
    _new, _sub, _child, layers, _modules = _fla_success_env(
        monkeypatch, layers_has_fla=True
    )
    assert FLA.install() is True
    assert layers.fla == "sentinel-fla"


@_need(FLA)
def test_fla_apply_alias_identity():
    assert FLA.apply_fla_module_relocation is FLA.install


# ---------------------------------------------------------------------------
# flagtree_target_info.py
# ---------------------------------------------------------------------------
@_need(FLAGTREE)
def test_flagtree_current_arch_variants():
    ti = _ns(current_target=lambda: _ns(arch="gfx942"))
    assert FLAGTREE._current_arch(ti) == "gfx942"
    assert FLAGTREE._current_arch(_ns(current_target=lambda: None)) == ""
    assert FLAGTREE._current_arch(_ns(current_target=lambda: _ns(arch=5))) == ""

    def boom():
        raise RuntimeError("no target")

    assert FLAGTREE._current_arch(_ns(current_target=boom)) == ""


@_need(FLAGTREE)
def test_flagtree_predicate_non_hip_and_prefix():
    ti = _ns(is_hip=lambda: False, current_target=lambda: _ns(arch="gfx1250"))
    predicate = FLAGTREE._make_predicate(ti, "is_hip_gfx1250", ("gfx1250",))
    assert predicate() is False
    assert predicate.__name__ == "is_hip_gfx1250"

    hip = _ns(is_hip=lambda: True, current_target=lambda: _ns(arch="gfx942"))
    assert FLAGTREE._make_predicate(hip, "p", ("gfx942", "gfx950"))() is True
    assert FLAGTREE._make_predicate(hip, "p", ("gfx90a",))() is False


@_need(FLAGTREE)
def test_flagtree_predicate_constexpr_decorator():
    calls = []

    def deco(fn):
        def wrapper():
            calls.append(True)
            return fn()

        return wrapper

    ti = _ns(
        is_hip=lambda: True,
        current_target=lambda: _ns(arch="gfx11"),
        constexpr_function=deco,
    )
    predicate = FLAGTREE._make_predicate(ti, "is_hip_gfx11", ("gfx11",))
    assert predicate() is True
    assert calls == [True]


@_need(FLAGTREE)
def test_flagtree_predicate_decorator_failure():
    def deco(fn):
        raise RuntimeError("cannot decorate")

    ti = _ns(
        is_hip=lambda: True,
        current_target=lambda: _ns(arch="gfx12"),
        constexpr_function=deco,
    )
    predicate = FLAGTREE._make_predicate(ti, "is_hip_gfx12", ("gfx12",))
    assert predicate() is True
    assert predicate.__name__ == "is_hip_gfx12"


def _install_fake_triton(monkeypatch, target_info):
    triton = _module("triton")
    lang = _module("triton.language")
    lang.target_info = target_info
    triton.language = lang
    _install(
        monkeypatch,
        {
            "triton": triton,
            "triton.language": lang,
            "triton.language.target_info": target_info,
        },
    )


@_need(FLAGTREE)
def test_flagtree_apply_import_error(monkeypatch):
    FLAGTREE.apply_flagtree_target_info.cache_clear()
    monkeypatch.setitem(sys.modules, "triton.language.target_info", None)
    assert FLAGTREE.apply_flagtree_target_info() is False
    FLAGTREE.apply_flagtree_target_info.cache_clear()


@_need(FLAGTREE)
def test_flagtree_apply_adds_missing(monkeypatch):
    FLAGTREE.apply_flagtree_target_info.cache_clear()
    ti = _module(
        "triton.language.target_info",
        is_hip=lambda: False,
        current_target=lambda: None,
        constexpr_function=None,
    )
    _install_fake_triton(monkeypatch, ti)
    assert FLAGTREE.apply_flagtree_target_info() is True
    for name in FLAGTREE._ARCH_PREDICATES:
        assert callable(getattr(ti, name))
        assert getattr(ti, name)() is False
    FLAGTREE.apply_flagtree_target_info.cache_clear()


@_need(FLAGTREE)
def test_flagtree_apply_keeps_existing(monkeypatch):
    FLAGTREE.apply_flagtree_target_info.cache_clear()
    sentinel = lambda: "kept"  # noqa: E731
    ti = _module(
        "triton.language.target_info",
        is_hip=lambda: False,
        current_target=lambda: None,
        constexpr_function=None,
        is_hip_gfx1250=sentinel,
    )
    _install_fake_triton(monkeypatch, ti)
    assert FLAGTREE.apply_flagtree_target_info() is True
    assert ti.is_hip_gfx1250 is sentinel
    assert callable(ti.is_hip_cdna)
    FLAGTREE.apply_flagtree_target_info.cache_clear()


# ---------------------------------------------------------------------------
# worker_symbols_032.py
# ---------------------------------------------------------------------------
def _sp_config(enable_sp=True, tp=8, use_inductor=False, splitting_ops=("x",)):
    return _ns(
        compilation_config=_ns(
            pass_config=_ns(enable_sp=enable_sp),
            use_inductor_graph_partition=use_inductor,
            splitting_ops=list(splitting_ops),
        ),
        parallel_config=_ns(tensor_parallel_size=tp),
    )


@_need(WORKER)
def test_sp_residual_disabled_and_single_tp():
    assert WORKER._is_residual_scattered_for_sp(_sp_config(enable_sp=False), 8) is False
    assert WORKER._is_residual_scattered_for_sp(_sp_config(tp=1), 8) is False


@_need(WORKER)
def test_sp_residual_requires_full_graph():
    with pytest.raises(AssertionError):
        WORKER._is_residual_scattered_for_sp(_sp_config(use_inductor=False), 8)


@_need(WORKER)
def test_sp_residual_true_paths():
    assert WORKER._is_residual_scattered_for_sp(_sp_config(use_inductor=True), 16) is True
    assert WORKER._is_residual_scattered_for_sp(
        _sp_config(use_inductor=False, splitting_ops=()), 16
    ) is True


@_need(WORKER)
def test_sp_residual_unaligned_tokens():
    with pytest.raises(AssertionError):
        WORKER._is_residual_scattered_for_sp(_sp_config(use_inductor=True), 15)


@_need(WORKER)
def test_supports_xdrope_probe():
    class Model:
        def get_xdrope_input_positions(self):
            return None

    assert WORKER._supports_xdrope(Model()) is True
    assert WORKER._supports_xdrope(object()) is False
    assert WORKER._supports_xdrope(_ns(get_xdrope_input_positions=5)) is False


@_need(WORKER)
@needs_numpy
def test_xdrotary_positions_write_in_place():
    out = np.zeros((2, 5), dtype=np.int64)
    WORKER._XDRotaryEmbedding.get_next_input_positions_tensor(out, 1, 3, 2)
    assert out[0, 1:3].tolist() == [3, 4]
    assert out[1, 1:3].tolist() == [3, 4]
    assert out[0, 0] == 0 and out[0, 4] == 0


@_need(WORKER)
@needs_torch
def test_dtype_size_helper():
    assert WORKER._dtype_size(torch.float16) == 2
    assert WORKER._dtype_size(torch.int8) == 1


@_need(WORKER)
@needs_torch
def test_kv_cache_packing_branch():
    spec = _ns(dtype=torch.int8, page_size_padded=None, page_size_bytes=0)
    raw = torch.arange(24, dtype=torch.uint8)
    view = WORKER._reshape_attention_kv_cache(
        raw, spec, (2, 3, 4), (0, 1, 2), 2, packing=(0, 12)
    )
    assert tuple(view.shape) == (2, 3, 4)
    assert view.flatten().tolist() == list(range(24))


@_need(WORKER)
@needs_torch
def test_kv_cache_padded_branch():
    spec = _ns(dtype=torch.int8, page_size_padded=6, page_size_bytes=6)
    raw = torch.zeros(20, dtype=torch.uint8)
    view = WORKER._reshape_attention_kv_cache(
        raw, spec, (2, 3, 4), (0, 1, 2), 2, packing=None
    )
    assert tuple(view.shape) == (2, 3, 4)
    assert view.stride(0) == 6


@_need(WORKER)
@needs_torch
def test_kv_cache_page_aligned_branch():
    spec = _ns(dtype=torch.int8, page_size_padded=None, page_size_bytes=0)
    raw = torch.arange(24, dtype=torch.uint8)
    view = WORKER._reshape_attention_kv_cache(
        raw, spec, (2, 3, 4), (0, 1, 2), 3, packing=None, page_aligned_blocks=True
    )
    assert tuple(view.shape) == (2, 3, 4)
    assert view[0, 0, 0].item() == 0
    assert view[1, 0, 0].item() == 4


@_need(WORKER)
@needs_torch
def test_kv_cache_contiguous_branch():
    spec = _ns(dtype=torch.int8, page_size_padded=None, page_size_bytes=0)
    raw = torch.arange(24, dtype=torch.uint8)
    view = WORKER._reshape_attention_kv_cache(
        raw, spec, (2, 3, 4), (0, 1, 2), 2, packing=None
    )
    assert view.flatten().tolist() == list(range(24))


@_need(WORKER)
@needs_torch
def test_kv_cache_invalid_layouts():
    spec = _ns(dtype=torch.int8, page_size_padded=None, page_size_bytes=0)
    raw = torch.zeros(64, dtype=torch.uint8)
    with pytest.raises(AssertionError):
        WORKER._reshape_attention_kv_cache(
            raw, spec, (2, 3, 4), (1, 0, 2), 2, packing=(0, 24)
        )
    padded = _ns(dtype=torch.int8, page_size_padded=6, page_size_bytes=6)
    with pytest.raises(AssertionError):
        WORKER._reshape_attention_kv_cache(
            raw, padded, (2, 3, 4), (0, 1, 2), 3, packing=None
        )
    with pytest.raises(AssertionError):
        WORKER._reshape_attention_kv_cache(
            raw, spec, (2, 3, 4), (0, 1, 2), 2, packing=None,
            page_aligned_blocks=True,
        )


@_need(WORKER)
@needs_torch
def test_routed_experts_cpu_identity_and_tolists():
    routing = torch.arange(4).reshape(2, 2)
    slots = torch.arange(2)
    tensors = WORKER._RoutedExpertsTensors(routing, slots)
    assert tensors.to_cpu_nonblocking() is tensors
    lists = tensors.tolists()
    assert isinstance(lists, WORKER._RoutedExpertsLists)
    assert lists.routing_data.tolist() == routing.tolist()
    assert lists.slot_mapping.tolist() == slots.tolist()


@_need(WORKER)
def test_routed_experts_non_cpu_d2h():
    sentinel = object()

    class FakeTensor:
        def __init__(self):
            self.device = _ns(type="cuda")

        def to(self, *args, **kwargs):
            return sentinel

    tensors = WORKER._RoutedExpertsTensors(FakeTensor(), FakeTensor())
    moved = tensors.to_cpu_nonblocking()
    assert moved is not tensors
    assert moved.routing_data is sentinel
    assert moved.slot_mapping is sentinel


@_need(WORKER)
def test_set_missing_only_when_absent():
    module = _ns()
    WORKER._set_missing(module, "a", 1)
    assert module.a == 1
    WORKER._set_missing(module, "a", 2)
    assert module.a == 1


@_need(WORKER)
def test_apply_worker_symbols_installs(monkeypatch):
    WORKER.apply_worker_symbols_032.cache_clear()
    keep_xd = object()
    keep_lists = object()
    worker_utils = _module("vllm.v1.worker.utils")
    rotary = _module("vllm.model_executor.layers.rotary_embedding")
    rotary.XDRotaryEmbedding = keep_xd
    interfaces = _module("vllm.model_executor.models.interfaces")
    attn = _module("vllm.v1.worker.gpu.attn_utils")
    outputs = _module("vllm.v1.outputs")
    outputs.RoutedExpertsLists = keep_lists
    layers = _module("vllm.model_executor.layers", rotary_embedding=rotary)
    models = _module("vllm.model_executor.models", interfaces=interfaces)
    v1 = _module("vllm.v1", outputs=outputs)
    worker = _module("vllm.v1.worker", utils=worker_utils)
    gpu = _module("vllm.v1.worker.gpu", attn_utils=attn)
    _install(
        monkeypatch,
        {
            "vllm.model_executor.layers": layers,
            "vllm.model_executor.models": models,
            "vllm.v1": v1,
            "vllm.v1.worker": worker,
            "vllm.v1.worker.gpu": gpu,
        },
    )
    assert WORKER.apply_worker_symbols_032() is True
    assert worker_utils.is_residual_scattered_for_sp is WORKER._is_residual_scattered_for_sp
    assert rotary.XDRotaryEmbedding is keep_xd
    assert interfaces.SupportsXDRoPE is WORKER._SupportsXDRoPE
    assert interfaces.supports_xdrope is WORKER._supports_xdrope
    assert attn._reshape_attention_kv_cache is WORKER._reshape_attention_kv_cache
    assert outputs.RoutedExpertsLists is keep_lists
    assert outputs.RoutedExpertsTensors is WORKER._RoutedExpertsTensors
    WORKER.apply_worker_symbols_032.cache_clear()


# ---------------------------------------------------------------------------
# qsa_cache_hint.py
# ---------------------------------------------------------------------------
def _qsa_source(policy='"evict_first"', extra_load=False):
    body = [
        "def kern(k_cache_ptr, offs):",
        "    keys = tl.load(k_cache_ptr + offs, mask=offs < 4, eviction_policy=%s)"
        % policy,
        "    return keys",
    ]
    if extra_load:
        body.insert(
            2,
            "    keys = tl.load(k_cache_ptr + offs, mask=offs < 4, "
            'eviction_policy="evict_first")',
        )
    return "\n".join(body) + "\n"


@_need(QSA)
def test_qsa_strip_hint_ok():
    out = QSA._without_k_cache_eviction_hint(_ns(src=_qsa_source()))
    assert isinstance(out, str)
    assert "eviction_policy" not in out
    assert out.endswith("\n")


@_need(QSA)
def test_qsa_strip_hint_none_when_absent():
    source = "def kern(k_cache_ptr, offs):\n    keys = tl.load(k_cache_ptr + offs)\n"
    assert QSA._without_k_cache_eviction_hint(_ns(src=source)) is None


@_need(QSA)
@pytest.mark.parametrize(
    "source",
    [
        _qsa_source(extra_load=True),
        "def kern(p):\n    keys = tl.load(p)\n",
        "def kern(p):\n    x = tl.load(p)\n",
    ],
)
def test_qsa_strip_hint_rejects_unexpected_load(source):
    with pytest.raises(RuntimeError):
        QSA._without_k_cache_eviction_hint(_ns(src=source))


@_need(QSA)
def test_qsa_strip_hint_rejects_other_policy():
    with pytest.raises(RuntimeError):
        QSA._without_k_cache_eviction_hint(_ns(src=_qsa_source('"evict_last"')))


class _QsaKernel:
    def __init__(self, src):
        self.src = src
        self.device_caches = {}
        self.updated = None

    def _unsafe_update_src(self, source):
        self.updated = source
        self.src = source


def _qsa_env(monkeypatch, triton_version="3.6.0", flagtree_missing=False):
    kernels = {name: _QsaKernel(_qsa_source()) for name in QSA._KERNEL_NAMES}
    qsa_indexer = _module("qsa_indexer", **kernels)
    ops = _module("vllm.models.qwen4_exp.nvidia.ops", qsa_indexer=qsa_indexer)
    _install(monkeypatch, {"vllm.models.qwen4_exp.nvidia.ops": ops})
    monkeypatch.setitem(sys.modules, "triton", _ns(__version__=triton_version))

    def fake_version(name):
        if flagtree_missing:
            raise PackageNotFoundError(name)
        return "0.7.0"

    monkeypatch.setattr(QSA, "version", fake_version)
    return kernels


@_need(QSA)
def test_qsa_architecture_mismatch():
    assert QSA.patch_qsa_cache_hint(_ns(architecture="Other"), _ns(type="cuda")) is False


@_need(QSA)
def test_qsa_triton_version_gate(monkeypatch):
    _qsa_env(monkeypatch, triton_version="3.5.1")
    assert QSA.patch_qsa_cache_hint(
        _ns(architecture="Qwen4ExpForCausalLM"), _ns(type="cuda")
    ) is False


@_need(QSA)
def test_qsa_flagtree_missing(monkeypatch):
    _qsa_env(monkeypatch, flagtree_missing=True)
    assert QSA.patch_qsa_cache_hint(
        _ns(architecture="Qwen4ExpForCausalLM"), _ns(type="cuda")
    ) is False


@_need(QSA)
@needs_torch
@pytest.mark.parametrize(
    "device_type,hip,cap", [("cpu", None, (9, 0)), ("cuda", "6.0", (9, 0)),
                             ("cuda", None, (8, 0))]
)
def test_qsa_device_gate(monkeypatch, device_type, hip, cap):
    _qsa_env(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: cap)
    assert QSA.patch_qsa_cache_hint(
        _ns(architecture="Qwen4ExpForCausalLM"), _ns(type=device_type)
    ) is False


@_need(QSA)
@needs_torch
def test_qsa_success_and_second_call(monkeypatch):
    kernels = _qsa_env(monkeypatch)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: (9, 0))
    config = _ns(architecture="Qwen4ExpForConditionalGeneration")
    assert QSA.patch_qsa_cache_hint(config, _ns(type="cuda")) is True
    for kernel in kernels.values():
        assert kernel._fl_qsa_original_src == _qsa_source()
        assert "eviction_policy" not in kernel.src
        assert kernel.updated is not None
    assert QSA.patch_qsa_cache_hint(config, _ns(type="cuda")) is False


@_need(QSA)
@needs_torch
def test_qsa_refuses_already_compiled_kernel(monkeypatch):
    kernels = _qsa_env(monkeypatch)
    kernels[QSA._KERNEL_NAMES[1]].device_caches = {"compiled": True}
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: (9, 0))
    with pytest.raises(RuntimeError):
        QSA.patch_qsa_cache_hint(
            _ns(architecture="Qwen4ExpForCausalLM"), _ns(type="cuda")
        )


# ---------------------------------------------------------------------------
# exponential_compat.py
# ---------------------------------------------------------------------------
class _FakeAten:
    def __init__(self):
        self.calls = []
        self.reject_generator = False

    def exponential_(self, tensor, lambd=1.0, **kwargs):
        self.calls.append((tensor, lambd, kwargs))
        if "generator" in kwargs and self.reject_generator:
            raise TypeError("unexpected keyword generator")
        return "result"


class _FakeTensor:
    pass


def _exponential_env(monkeypatch):
    monkeypatch.setattr(EXP, "_warned", False)
    monkeypatch.setattr(EXP, "_installed", False)
    aten = _FakeAten()
    fake = _ns(Tensor=_FakeTensor, ops=_ns(aten=aten))
    monkeypatch.setattr(EXP, "torch", fake)
    return aten


@_need(EXP)
def test_exponential_generator_path(monkeypatch):
    aten = _exponential_env(monkeypatch)
    tensor = _FakeTensor()
    generator = object()
    assert EXP._exponential_compat(tensor, 2.0, generator=generator) == "result"
    assert aten.calls[-1] == (tensor, 2.0, {"generator": generator})


@_need(EXP)
def test_exponential_generator_typeerror_warns_once(monkeypatch, caplog):
    aten = _exponential_env(monkeypatch)
    aten.reject_generator = True
    tensor = _FakeTensor()
    with caplog.at_level(logging.WARNING):
        assert EXP._exponential_compat(tensor, generator=object()) == "result"
        assert EXP._exponential_compat(tensor, generator=object()) == "result"
    warnings = [r for r in caplog.records if "ignores the generator" in r.getMessage()]
    assert len(warnings) == 1
    assert aten.calls[-1] == (tensor, 1.0, {})


@_need(EXP)
def test_exponential_no_generator(monkeypatch):
    aten = _exponential_env(monkeypatch)
    tensor = _FakeTensor()
    assert EXP._exponential_compat(tensor, 3.0) == "result"
    assert aten.calls[-1] == (tensor, 3.0, {})


@_need(EXP)
def test_exponential_apply_once(monkeypatch):
    _exponential_env(monkeypatch)
    assert EXP.apply_exponential_compat() is True
    assert EXP.torch.Tensor.exponential_ is EXP._exponential_compat
    assert EXP.apply_exponential_compat() is False


# ---------------------------------------------------------------------------
# triton_kernel.py
# ---------------------------------------------------------------------------
@_need(TK)
def test_proxy_requires_overrides():
    with pytest.raises(ValueError):
        TK.KernelLaunchMetaProxy(object(), {})


@_need(TK)
def test_proxy_property_returns_copy():
    proxy = TK.KernelLaunchMetaProxy(_ns(), {"num_warps": 4})
    view = proxy.launch_overrides
    view["num_warps"] = 8
    assert proxy.launch_overrides == {"num_warps": 4}


@_need(TK)
def test_proxy_getitem_merges_overrides():
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append((grid, args, kwargs))
                return "launched"

            return launch

    proxy = TK.KernelLaunchMetaProxy(Kernel(), {"num_warps": 4})
    launcher = proxy[(1, 1)]
    assert launcher(3, num_warps=8, other=1) == "launched"
    assert calls == [((1, 1), (3,), {"num_warps": 4, "other": 1})]


@_need(TK)
def test_proxy_delegates_attributes():
    proxy = TK.KernelLaunchMetaProxy(_ns(answer=lambda: 42), {"x": 1})
    assert proxy.answer() == 42
    with pytest.raises(AttributeError):
        proxy.missing


@_need(TK)
def test_patch_kernel_launch_is_idempotent():
    module = _ns(kernel=object())
    TK.patch_kernel_launch_meta(module, "kernel", {"num_stages": 2})
    first = module.kernel
    assert isinstance(first, TK.KernelLaunchMetaProxy)
    assert TK.patch_kernel_launch_meta(module, "kernel", {"num_stages": 2}) is None
    assert module.kernel is first


@_need(TK)
def test_patch_kernel_launch_conflict():
    module = _ns(kernel=object())
    TK.patch_kernel_launch_meta(module, "kernel", {"num_stages": 2})
    with pytest.raises(RuntimeError):
        TK.patch_kernel_launch_meta(module, "kernel", {"num_stages": 3})


# ---------------------------------------------------------------------------
# arm_cpu_gdn.py
# ---------------------------------------------------------------------------
@_need(ARM)
@needs_torch
def test_arm_patch_clones_strided_state_indices():
    strided = torch.arange(12).reshape(3, 4)[:, ::2]

    class Builder:
        def build(self, *args, **kwargs):
            return _ns(non_spec_state_indices_tensor=strided)

    assert ARM.apply_arm_cpu_gdn_state_indices_patch(Builder) is True
    cloned = Builder().build().non_spec_state_indices_tensor
    assert cloned.stride(-1) == 1
    assert torch.equal(cloned, strided)


@_need(ARM)
@needs_torch
def test_arm_patch_leaves_contiguous_tensor():
    contiguous = torch.arange(4)

    class Builder:
        def build(self, *args, **kwargs):
            return _ns(non_spec_state_indices_tensor=contiguous)

    ARM.apply_arm_cpu_gdn_state_indices_patch(Builder)
    assert Builder().build().non_spec_state_indices_tensor is contiguous


@_need(ARM)
@needs_torch
def test_arm_patch_is_idempotent():
    class Builder:
        def build(self, *args, **kwargs):
            return _ns(non_spec_state_indices_tensor=None)

    assert ARM.apply_arm_cpu_gdn_state_indices_patch(Builder) is True
    assert ARM.apply_arm_cpu_gdn_state_indices_patch(Builder) is False


@_need(ARM)
@needs_torch
def test_arm_patch_default_builder(monkeypatch):
    class Builder:
        def build(self, *args, **kwargs):
            return _ns(non_spec_state_indices_tensor=None)

    module = _module(
        "vllm.v1.attention.backends.gdn_attn", GDNAttentionMetadataBuilder=Builder
    )
    _install(monkeypatch, {"vllm.v1.attention.backends.gdn_attn": module})
    assert ARM.apply_arm_cpu_gdn_state_indices_patch() is True
    assert getattr(Builder.build, ARM._PATCH_MARKER) is True


# ---------------------------------------------------------------------------
# attention/utils.py
# ---------------------------------------------------------------------------
def _attention_sdpa_env(monkeypatch, existing_layout=None):
    def rearrange(x, pattern):
        if "b s h d -> b h s d" in pattern or "b h s d -> b s h d" in pattern:
            return x.permute(0, 2, 1, 3)
        raise ValueError(pattern)

    vit = _module(
        "vit_attn_wrappers", einops=_ns(rearrange=rearrange), F=torch.nn.functional
    )
    if existing_layout is not None:
        sentinel = _ns(_fl_hygon_sdpa_layout=existing_layout)
        vit.apply_sdpa = sentinel
    else:
        sentinel = None
        vit.apply_sdpa = lambda *a, **k: None
    ops = _module("vllm.v1.attention.ops", vit_attn_wrappers=vit)
    _install(monkeypatch, {"vllm.v1.attention.ops": ops})
    monkeypatch.setattr(ATTN, "logger", Mock())
    return vit, sentinel


@_need(ATTN)
@needs_torch
def test_hygon_sdpa_pads_d72(monkeypatch):
    vit, _sentinel = _attention_sdpa_env(monkeypatch)
    ATTN._patch_hygon_sdpa_contiguous()
    patched = vit.apply_sdpa
    assert patched._fl_hygon_sdpa_layout == "bhsd-pad-d72"
    q = torch.rand(1, 3, 2, 72)
    assert tuple(patched(q, q, q).shape) == (1, 3, 2, 72)


@_need(ATTN)
@needs_torch
def test_hygon_sdpa_non_d72(monkeypatch):
    vit, _sentinel = _attention_sdpa_env(monkeypatch)
    ATTN._patch_hygon_sdpa_contiguous()
    q = torch.rand(2, 3, 2, 64)
    assert tuple(vit.apply_sdpa(q, q, q).shape) == (2, 3, 2, 64)


@_need(ATTN)
@needs_torch
def test_hygon_sdpa_idempotent(monkeypatch):
    vit, sentinel = _attention_sdpa_env(monkeypatch, existing_layout="bhsd-pad-d72")
    ATTN._patch_hygon_sdpa_contiguous()
    assert vit.apply_sdpa is sentinel
    ATTN.logger.info_once.assert_not_called()


class _Enum:
    FLASH_ATTN = object()
    ROCM_AITER_FA = object()


def _gcu_env(monkeypatch):
    registry = _module("vllm.v1.attention.backends.registry", AttentionBackendEnum=_Enum)

    @contextlib.contextmanager
    def gpu_sync_allowed():
        yield

    sync = _module("vllm.utils.gpu_sync_debug", gpu_sync_allowed=gpu_sync_allowed)

    def flash_attn_varlen_func(**kwargs):
        return kwargs["q"]

    flash = _module(
        "flash_attn.vllm_flash_attn", flash_attn_varlen_func=flash_attn_varlen_func
    )
    _install(
        monkeypatch,
        {
            "vllm.v1.attention.backends.registry": registry,
            "vllm.utils.gpu_sync_debug": sync,
            "flash_attn.vllm_flash_attn": flash,
        },
    )


def _gcu_self(attn_backend, num_heads, head_size, num_kv_heads=None):
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads

    def view_qkv_to_4d(query, key, value, batch, q_len, kv_len):
        q = query.reshape(batch, q_len, num_heads, head_size)
        k = key.reshape(batch, kv_len, kv_heads, head_size)
        v = value.reshape(batch, kv_len, kv_heads, head_size)
        return q, k, v

    return _ns(
        attn_backend=attn_backend,
        scale=0.5,
        num_heads=num_heads,
        head_size=head_size,
        num_kv_heads=kv_heads,
        view_qkv_to_4d=view_qkv_to_4d,
        _fl_gcu_original_forward_native=Mock(return_value="original"),
    )


@_need(ATTN)
@needs_torch
def test_gcu_forward_non_flash_passthrough(monkeypatch):
    _gcu_env(monkeypatch)
    self = _gcu_self(object(), 2, 4)
    out = ATTN._gcu_forward_mm_attention(
        self, torch.zeros(1), torch.zeros(1), torch.zeros(1), None, None, None
    )
    assert out == "original"
    self._fl_gcu_original_forward_native.assert_called_once()


@_need(ATTN)
@needs_torch
def test_gcu_forward_varlen_without_cu_seqlens(monkeypatch):
    _gcu_env(monkeypatch)
    self = _gcu_self(_Enum.FLASH_ATTN, 2, 4)
    query = torch.rand(1, 3, 8)
    key = torch.rand(1, 5, 8)
    out = ATTN._gcu_forward_mm_attention(self, query, key, key, None, None, None)
    assert tuple(out.shape) == (1, 3, 8)


@_need(ATTN)
@needs_torch
def test_gcu_forward_varlen_with_cu_seqlens(monkeypatch):
    _gcu_env(monkeypatch)
    self = _gcu_self(_Enum.FLASH_ATTN, 1, 4)
    query = torch.rand(1, 2, 4)
    key = torch.rand(1, 2, 4)
    out = ATTN._gcu_forward_mm_attention(
        self, query, key, key, torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor(2), None,
    )
    assert tuple(out.shape) == (1, 2, 4)


@_need(ATTN)
@needs_torch
def test_gcu_forward_cu_seqlens_assert(monkeypatch):
    _gcu_env(monkeypatch)
    self = _gcu_self(_Enum.FLASH_ATTN, 1, 4)
    with pytest.raises(AssertionError):
        ATTN._gcu_forward_mm_attention(
            self, torch.zeros(1, 1, 4), torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 4), torch.tensor([0, 1]), None, None,
        )


def _mm_env(monkeypatch, vendor, is_cuda=False):
    # Import the real parent package before faking ``vllm.platforms`` so its
    # own imports still see the real platform object, then rebind the leaf
    # attribute that ``import ... as mm_mod`` resolves through.
    try:
        import vllm.model_executor.layers.attention as attention_pkg
    except Exception:
        attention_pkg = None

    class MMEncoderAttention:
        forward_native = staticmethod(lambda *a, **k: "native")
        forward_oot = staticmethod(lambda *a, **k: "oot")
        forward_cuda = staticmethod(lambda *a, **k: "cuda")

    mm = _module(
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        MMEncoderAttention=MMEncoderAttention,
        maybe_get_vit_flash_attn_backend=lambda backend: None,
    )
    platform = _ns(vendor_name=vendor, is_cuda=lambda: is_cuda)
    platforms = _module("vllm.platforms", current_platform=platform)
    registry = _module(
        "vllm.v1.attention.backends.registry", AttentionBackendEnum=_Enum
    )
    _install(
        monkeypatch,
        {
            "vllm.model_executor.layers.attention.mm_encoder_attention": mm,
            "vllm.platforms": platforms,
            "vllm.v1.attention.backends.registry": registry,
        },
    )
    # ``import ...mm_encoder_attention as mm_mod`` walks attributes from the
    # real top-level package, so rebind the parent attribute as well.
    if attention_pkg is not None:
        monkeypatch.setattr(attention_pkg, "mm_encoder_attention", mm)
    monkeypatch.setattr(ATTN, "logger", Mock())
    return mm, platform


@_need(ATTN)
def test_patch_mm_hygon_invokes_sdpa_patch(monkeypatch):
    mm, _platform = _mm_env(monkeypatch, "hygon")
    called = []
    monkeypatch.setattr(
        ATTN, "_patch_hygon_sdpa_contiguous", lambda: called.append(True)
    )
    ATTN.patch_mm_encoder_attention()
    assert called == [True]
    assert mm.maybe_get_vit_flash_attn_backend is not None


@_need(ATTN)
def test_patch_mm_gcu_sets_forward(monkeypatch):
    mm, _platform = _mm_env(monkeypatch, "gcu")
    sentinel = lambda *a, **k: "gcu"  # noqa: E731
    monkeypatch.setattr(ATTN, "_gcu_forward_mm_attention", sentinel)
    original = mm.MMEncoderAttention.forward_native
    ATTN.patch_mm_encoder_attention()
    assert mm.MMEncoderAttention.forward_native is sentinel
    assert mm.MMEncoderAttention.forward_oot is sentinel
    assert mm.MMEncoderAttention._fl_gcu_original_forward_native is original


@_need(ATTN)
def test_patch_mm_gcu_keeps_saved_original(monkeypatch):
    mm, _platform = _mm_env(monkeypatch, "enflame")
    monkeypatch.setattr(ATTN, "_gcu_forward_mm_attention", lambda *a, **k: None)
    saved = object()
    mm.MMEncoderAttention._fl_gcu_original_forward_native = saved
    ATTN.patch_mm_encoder_attention()
    assert mm.MMEncoderAttention._fl_gcu_original_forward_native is saved


@_need(ATTN)
@needs_torch
def test_patch_mm_kunlunxin_chunked_sdpa(monkeypatch):
    mm, _platform = _mm_env(monkeypatch, "kunlunxin")
    ATTN.patch_mm_encoder_attention()
    forward = mm.MMEncoderAttention._forward_sdpa
    self = _gcu_self(object(), 2, 4)
    query = torch.rand(1, 4, 8)
    assert tuple(forward(self, query, query, query).shape) == (1, 4, 8)
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    assert tuple(forward(self, query, query, query, cu).shape) == (1, 4, 8)


@_need(ATTN)
def test_patch_mm_cuda_binds_oot(monkeypatch):
    mm, _platform = _mm_env(monkeypatch, "nvidia", is_cuda=True)
    ATTN.patch_mm_encoder_attention()
    assert mm.MMEncoderAttention.forward_oot is mm.MMEncoderAttention.forward_cuda


@_need(ATTN)
def test_patched_vit_backend_selection(monkeypatch):
    mm, _platform = _mm_env(monkeypatch, "nvidia", is_cuda=True)
    ATTN.patch_mm_encoder_attention()
    fn = mm.maybe_get_vit_flash_attn_backend

    sentinel = object()
    _install(
        monkeypatch,
        {"vllm.vllm_flash_attn": _module(
            "vllm.vllm_flash_attn", flash_attn_varlen_func=sentinel)},
    )
    assert fn(_Enum.FLASH_ATTN) is sentinel

    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)
    fallback = object()
    _install(
        monkeypatch,
        {"flash_attn": _module("flash_attn", flash_attn_varlen_func=fallback)},
    )
    assert fn(_Enum.FLASH_ATTN) is fallback

    aiter_fn = object()
    _install(monkeypatch, {"aiter": _module("aiter", flash_attn_varlen_func=aiter_fn)})
    assert fn(_Enum.ROCM_AITER_FA) is aiter_fn
    assert fn(object()) is None


# ---------------------------------------------------------------------------
# compilation/graph.py
# ---------------------------------------------------------------------------
def _fake_platform(**overrides):
    base = dict(
        device_type="cuda",
        vendor_name=None,
        get_global_graph_pool=lambda: "pool-token",
        graph_pool_handle=lambda: "handle-token",
        is_cuda=lambda: False,
        torch_device_fn=_ns(
            graph=lambda *a, **k: contextlib.nullcontext(),
            current_stream=lambda: "current-stream",
            synchronize=lambda: None,
            Stream=lambda: "new-stream",
        ),
    )
    base.update(overrides)
    return _ns(**base)


class _FakeStorage:
    def nbytes(self):
        return 16

    def data_ptr(self):
        return 1234


class _FakeTorchTensor:
    def __init__(self, device_type="gcu"):
        self.device = _ns(type=device_type)
        self.shape = (2, 2)
        self.dtype = "float32"

    def untyped_storage(self):
        return _FakeStorage()

    def stride(self):
        return (2, 1)

    def storage_offset(self):
        return 0

    def is_conj(self):
        return False

    def is_neg(self):
        return False


@_need(GRAPH)
def test_weak_ref_npu_tensor(monkeypatch):
    monkeypatch.setattr(GRAPH, "torch", _ns(Tensor=_FakeTorchTensor))
    sentinel = object()
    torch_npu = _module("torch_npu", _C=_ns(_weak_ref_tensor=lambda v: sentinel))
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    assert GRAPH._weak_ref_npu_tensor(_FakeTorchTensor("npu")) is sentinel
    assert GRAPH._weak_ref_npu_tensor(object()) is not sentinel


@_need(GRAPH)
def test_weak_ref_npu_missing_runtime(monkeypatch):
    monkeypatch.setattr(GRAPH, "torch", _ns(Tensor=_FakeTorchTensor))
    monkeypatch.setitem(sys.modules, "torch_npu", _module("torch_npu"))
    with pytest.raises(RuntimeError):
        GRAPH._weak_ref_npu_tensor(_FakeTorchTensor("npu"))


@_need(GRAPH)
def test_weak_ref_npu_tensors_recursive(monkeypatch):
    monkeypatch.setattr(GRAPH, "torch", _ns(Tensor=_FakeTorchTensor))
    sentinel = object()
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        _module("torch_npu", _C=_ns(_weak_ref_tensor=lambda v: sentinel)),
    )

    class IntermediateTensors:
        def __init__(self, tensors):
            self.tensors = tensors

    monkeypatch.setitem(
        sys.modules,
        "vllm.sequence",
        _module("vllm.sequence", IntermediateTensors=IntermediateTensors),
    )
    value = _FakeTorchTensor("npu")
    assert GRAPH._weak_ref_npu_tensors([value]) == [sentinel]
    assert GRAPH._weak_ref_npu_tensors({"a": value}) == {"a": sentinel}
    assert GRAPH._weak_ref_npu_tensors(7) == 7
    converted = GRAPH._weak_ref_npu_tensors(IntermediateTensors({"a": value}))
    assert converted.tensors == {"a": sentinel}


@_need(GRAPH)
def test_weak_ref_gcu_tensor(monkeypatch):
    sentinel = object()
    fake_torch = _ns(
        Tensor=_FakeTorchTensor,
        _C=_ns(
            _construct_storage_from_data_pointer=lambda *a: object(),
            _set_conj=lambda *a: None,
            _set_neg=lambda *a: None,
        ),
    )
    monkeypatch.setattr(GRAPH, "torch", fake_torch)
    torch_gcu = _module(
        "torch_gcu",
        _C=_ns(_construct_GCU_Tensor_From_Storage_And_Metadata=lambda m, s: sentinel),
    )
    monkeypatch.setitem(sys.modules, "torch_gcu", torch_gcu)
    assert GRAPH._weak_ref_gcu_tensor(_FakeTorchTensor("gcu")) is sentinel
    other = _FakeTorchTensor("cpu")
    assert GRAPH._weak_ref_gcu_tensor(other) is other


@_need(GRAPH)
def test_weak_ref_gcu_tensors_recursive(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        GRAPH,
        "torch",
        _ns(
            Tensor=_FakeTorchTensor,
            _C=_ns(
                _construct_storage_from_data_pointer=lambda *a: object(),
                _set_conj=lambda *a: None,
                _set_neg=lambda *a: None,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_gcu",
        _module(
            "torch_gcu",
            _C=_ns(_construct_GCU_Tensor_From_Storage_And_Metadata=lambda m, s: sentinel),
        ),
    )

    class IntermediateTensors:
        def __init__(self, tensors):
            self.tensors = tensors

    monkeypatch.setitem(
        sys.modules,
        "vllm.sequence",
        _module("vllm.sequence", IntermediateTensors=IntermediateTensors),
    )
    value = _FakeTorchTensor("gcu")
    assert GRAPH._weak_ref_gcu_tensors((value,)) == (sentinel,)
    assert GRAPH._weak_ref_gcu_tensors({"a": [value]}) == {"a": [sentinel]}
    assert GRAPH._weak_ref_gcu_tensors(3) == 3
    converted = GRAPH._weak_ref_gcu_tensors(IntermediateTensors({"a": value}))
    assert converted.tensors == {"a": sentinel}


@_need(GRAPH)
def test_weak_ref_gcu_missing_runtime(monkeypatch):
    monkeypatch.setattr(GRAPH, "torch", _ns(Tensor=_FakeTorchTensor, _C=_ns()))
    monkeypatch.setitem(sys.modules, "torch_gcu", _module("torch_gcu"))
    with pytest.raises(RuntimeError):
        GRAPH._weak_ref_gcu_tensor(_FakeTorchTensor("gcu"))


@_need(GRAPH)
@pytest.mark.parametrize(
    "device_type,vendor,expect",
    [
        ("cuda", "iluvatar", "iluvatar"),
        ("npu", None, "npu"),
        ("gcu", None, "gcu"),
        ("cpu", None, "upstream"),
    ],
)
def test_weak_ref_tensors_dispatch(monkeypatch, device_type, vendor, expect):
    monkeypatch.setattr(
        GRAPH, "current_platform", _ns(device_type=device_type, vendor_name=vendor)
    )
    monkeypatch.setattr(GRAPH, "_weak_ref_npu_tensors", lambda v: "npu")
    monkeypatch.setattr(GRAPH, "_weak_ref_gcu_tensors", lambda v: "gcu")
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.dispatch.backends.vendor.iluvatar.graph",
        _module(
            "vllm_fl.dispatch.backends.vendor.iluvatar.graph",
            _weak_ref_iluvatar_tensors=lambda v: "iluvatar",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.torch_utils",
        _module("vllm.utils.torch_utils", weak_ref_tensors=lambda v: "upstream"),
    )
    assert GRAPH.weak_ref_tensors(object()) == expect


@_need(GRAPH)
def test_weak_ref_tensors_upstream_failure(monkeypatch):
    monkeypatch.setattr(
        GRAPH, "current_platform", _ns(device_type="cpu", vendor_name=None)
    )
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", None)
    value = object()
    assert GRAPH.weak_ref_tensors(value) is value


@_need(GRAPH)
@pytest.mark.parametrize(
    "device_type,expected", [("npu", "native-stream"), ("cpu", "cached-stream")]
)
def test_graph_capture_stream(monkeypatch, device_type, expected):
    platform = _fake_platform(
        device_type=device_type,
        torch_device_fn=_ns(current_stream=lambda: "native-stream"),
    )
    monkeypatch.setattr(GRAPH, "current_platform", platform)
    monkeypatch.setattr(GRAPH, "current_stream", lambda: "cached-stream")
    assert GRAPH._graph_capture_stream() == expected


@_need(GRAPH)
def test_graph_entry_and_options_defaults():
    entry = GRAPH.GraphEntry(batch_descriptor=BD(4))
    assert entry.graph is None and entry.output is None
    assert entry.input_addresses is None
    options = GRAPH.GraphOptions()
    assert options.debug_log_enable is True
    assert options.gc_disable is False
    assert options.weak_ref_output is True


def _make_wrapper(monkeypatch, debug=False, **option_overrides):
    monkeypatch.setattr(
        GRAPH, "envs", _ns(VLLM_LOGGING_LEVEL="DEBUG" if debug else "INFO")
    )
    mode = object()
    none_mode = object()
    monkeypatch.setattr(GRAPH, "CUDAGraphMode", _ns(NONE=none_mode))
    monkeypatch.setattr(GRAPH, "current_platform", _fake_platform())
    monkeypatch.setattr(GRAPH, "current_stream", lambda: "cached-stream")
    runnable = lambda *a, **k: "ran"  # noqa: E731
    config = _ns(compilation_config=_ns())
    wrapper = GRAPH.GraphWrapper(runnable, config, mode, GRAPH.GraphOptions(**option_overrides))
    return wrapper, mode, none_mode


@_need(GRAPH)
def test_graphwrapper_init_asserts_none_mode(monkeypatch):
    monkeypatch.setattr(GRAPH, "envs", _ns(VLLM_LOGGING_LEVEL="INFO"))
    none_mode = object()
    monkeypatch.setattr(GRAPH, "CUDAGraphMode", _ns(NONE=none_mode))
    monkeypatch.setattr(GRAPH, "current_platform", _fake_platform())
    with pytest.raises(AssertionError):
        GRAPH.GraphWrapper(lambda: None, _ns(compilation_config=_ns()), none_mode)


@_need(GRAPH)
def test_graphwrapper_getattr_unwrap_and_clear(monkeypatch):
    wrapper, _mode, _none = _make_wrapper(monkeypatch)
    assert wrapper.unwrap()() == "ran"
    assert wrapper.cudagraph_wrapper is wrapper
    with pytest.raises(AttributeError):
        wrapper.unknown

    debug_wrapper, _m, _n = _make_wrapper(monkeypatch, debug=True)
    with pytest.raises(AttributeError) as excinfo:
        debug_wrapper.unknown
    assert "not exists in the runnable" in str(excinfo.value)

    delegate = GRAPH.GraphWrapper(
        _ns(foo=42), _ns(compilation_config=_ns()), object()
    )
    assert delegate.foo == 42
    delegate.concrete_graph_entries[BD(1)] = object()
    delegate.clear_graphs()
    assert delegate.concrete_graph_entries == {}


@_need(GRAPH)
def test_graphwrapper_clear_all_instances(monkeypatch):
    first, _m, _n = _make_wrapper(monkeypatch)
    second, _m2, _n2 = _make_wrapper(monkeypatch)
    first.concrete_graph_entries[BD(1)] = object()
    second.concrete_graph_entries[BD(2)] = object()
    GRAPH.GraphWrapper.clear_all_graphs()
    assert first.concrete_graph_entries == {}
    assert second.concrete_graph_entries == {}


@_need(GRAPH)
def test_graphwrapper_call_passthrough_branches(monkeypatch):
    wrapper, _mode, none_mode = _make_wrapper(monkeypatch)
    monkeypatch.setattr(GRAPH, "is_forward_context_available", lambda: False)
    assert wrapper(1) == "ran"

    monkeypatch.setattr(GRAPH, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        GRAPH,
        "get_forward_context",
        lambda: _ns(batch_descriptor=BD(4), cudagraph_runtime_mode=none_mode),
    )
    assert wrapper(1) == "ran"

    monkeypatch.setattr(
        GRAPH,
        "get_forward_context",
        lambda: _ns(batch_descriptor=BD(4), cudagraph_runtime_mode=object()),
    )
    assert wrapper(1) == "ran"


@_need(GRAPH)
def test_graphwrapper_call_capture_then_replay(monkeypatch):
    wrapper, mode, _none = _make_wrapper(monkeypatch)
    captured, replayed = [], []
    monkeypatch.setattr(
        wrapper,
        "_capture_graph",
        lambda entry, args, kwargs: captured.append(entry.batch_descriptor) or "captured",
    )
    monkeypatch.setattr(
        wrapper,
        "_replay_graph",
        lambda entry, args, kwargs: replayed.append(entry.batch_descriptor) or "replayed",
    )
    monkeypatch.setattr(GRAPH, "is_forward_context_available", lambda: True)
    descriptor = BD(8)
    monkeypatch.setattr(
        GRAPH,
        "get_forward_context",
        lambda: _ns(batch_descriptor=descriptor, cudagraph_runtime_mode=mode),
    )
    assert wrapper(1) == "captured"
    assert captured == [descriptor]
    wrapper.concrete_graph_entries[descriptor].graph = object()
    assert wrapper(1) == "replayed"
    assert replayed == [descriptor]


@_need(GRAPH)
@needs_torch
def test_graphwrapper_capture_graph(monkeypatch):
    wrapper, _mode, _none = _make_wrapper(monkeypatch)
    wrapper._graph_type = Mock(side_effect=lambda: _ns(replay=Mock()))

    captured = {}
    monkeypatch.setattr(
        GRAPH,
        "validate_cudagraph_capturing_enabled",
        lambda: captured.setdefault("validated", True),
    )
    monkeypatch.setattr(
        GRAPH, "set_graph_pool_id", lambda pool: captured.setdefault("pool", pool)
    )
    monkeypatch.setattr(GRAPH, "weak_ref_tensors", lambda value: ("weak", value))
    monkeypatch.setattr(GRAPH, "compilation_counter", _ns(num_cudagraph_captured=0))
    offloader = _ns(sync_prev_onload=Mock(), join_after_forward=Mock())
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.offloader.base",
        _module("vllm.model_executor.offloader.base", get_offloader=lambda: offloader),
    )
    output = torch.zeros(2)
    wrapper.runnable = lambda *a, **k: output
    arg = torch.zeros(3)
    entry = GRAPH.GraphEntry(batch_descriptor=BD(4))
    result = wrapper._capture_graph(entry, (arg,), {})
    assert result[0] == "weak" and result[1] is output
    assert entry.graph is not None
    assert entry.output[0] == "weak" and entry.output[1][1] is output
    assert entry.input_addresses == [arg.data_ptr()]
    assert captured["validated"] is True
    assert captured["pool"] == "pool-token"
    assert GRAPH.compilation_counter.num_cudagraph_captured == 1
    offloader.sync_prev_onload.assert_called()


@_need(GRAPH)
@needs_torch
def test_graphwrapper_validate_replay_addresses(monkeypatch):
    wrapper, _mode, _none = _make_wrapper(monkeypatch, debug=True)
    tensor = torch.zeros(3)
    entry = GRAPH.GraphEntry(
        batch_descriptor=BD(4), input_addresses=[tensor.data_ptr()]
    )
    wrapper._validate_replay_addresses(entry, (tensor,))
    with pytest.raises(AssertionError):
        wrapper._validate_replay_addresses(entry, (torch.zeros(3),))


@_need(GRAPH)
def test_graphwrapper_replay_graph(monkeypatch):
    wrapper, _mode, _none = _make_wrapper(monkeypatch)
    graph = _ns(replay=Mock())
    entry = GRAPH.GraphEntry(batch_descriptor=BD(4), graph=graph, output="out")
    offloader = _ns(sync_prev_onload=Mock(), join_after_forward=Mock())
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.offloader.base",
        _module("vllm.model_executor.offloader.base", get_offloader=lambda: offloader),
    )
    monkeypatch.setattr(GRAPH, "current_platform", _fake_platform(device_type="cpu"))
    assert wrapper._replay_graph(entry, (), {}) == "out"
    graph.replay.assert_called_once()

    sync = Mock()
    platform = _fake_platform(device_type="npu")
    platform.torch_device_fn.synchronize = sync
    monkeypatch.setattr(GRAPH, "current_platform", platform)
    wrapper._replay_graph(entry, (), {})
    sync.assert_called_once()

    monkeypatch.setattr(GRAPH, "get_forward_context", lambda: "ctx")
    used = []
    wrapper.graph_runtime = _ns(
        replay_scope=lambda ctx: (used.append(ctx), contextlib.nullcontext())[1]
    )
    assert wrapper._replay_graph(entry, (), {}) == "out"
    assert used == ["ctx"]


# ---------------------------------------------------------------------------
# compilation/graph_runtime.py and graph_params.py
# ---------------------------------------------------------------------------
@_need(GRT)
def test_batch_descriptor_validation():
    controller = GRT.GraphRuntimeController(vllm_config=object())
    descriptor = _ns(num_tokens=8)
    assert controller._batch_descriptor(_ns(batch_descriptor=descriptor)) is descriptor
    for bad in (None, True, "8", 1.5):
        with pytest.raises(RuntimeError):
            controller._batch_descriptor(_ns(batch_descriptor=_ns(num_tokens=bad)))


def _fake_graph_module():
    return _module("vllm_fl.compilation.graph", weak_ref_tensors=lambda v: ("weak", v))


@_need(GRT)
def test_capture_scope_prepares_and_weak_refs(monkeypatch):
    fake_graph = _fake_graph_module()
    monkeypatch.setitem(sys.modules, "vllm_fl.compilation.graph", fake_graph)
    calls = {}
    monkeypatch.setattr(
        GP, "prepare_graph_params", lambda d: calls.setdefault("prepared", d)
    )
    monkeypatch.setattr(
        GP, "weak_ref_workspace", lambda d, f: calls.setdefault("weak", (d, f))
    )
    controller = GRT.GraphRuntimeController(vllm_config=object())
    descriptor = _ns(num_tokens=16)
    context = _ns(batch_descriptor=descriptor, capturing=False)
    with controller.capture_scope(context):
        assert controller._active is True
        assert context.capturing is True
    assert controller._active is False
    assert context.capturing is False
    assert calls["prepared"] is descriptor
    assert calls["weak"] == (descriptor, fake_graph.weak_ref_tensors)


@_need(GRT)
def test_capture_scope_rejects_reentrancy(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_fl.compilation.graph", _fake_graph_module())
    monkeypatch.setattr(GP, "prepare_graph_params", lambda d: None)
    monkeypatch.setattr(GP, "weak_ref_workspace", lambda d, f: None)
    controller = GRT.GraphRuntimeController(vllm_config=object())
    controller._active = True
    with pytest.raises(AssertionError):
        with controller.capture_scope(
            _ns(batch_descriptor=_ns(num_tokens=4), capturing=False)
        ):
            pass


@_need(GRT)
def test_capture_scope_releases_on_exception(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm_fl.compilation.graph", _fake_graph_module())
    released = []
    monkeypatch.setattr(GP, "prepare_graph_params", lambda d: None)
    monkeypatch.setattr(GP, "weak_ref_workspace", lambda d, f: released.append(d))
    controller = GRT.GraphRuntimeController(vllm_config=object())
    context = _ns(batch_descriptor=_ns(num_tokens=4), capturing=False)
    with pytest.raises(ValueError):
        with controller.capture_scope(context):
            raise ValueError("boom")
    assert released
    assert controller._active is False
    assert context.capturing is False


@_need(GRT)
def test_replay_scope_synchronizes_and_updates(monkeypatch):
    sync = Mock()
    platform = _fake_platform()
    platform.torch_device_fn.current_stream = lambda: _ns(synchronize=sync)
    monkeypatch.setattr(GRT, "current_platform", platform)
    controller = GRT.GraphRuntimeController(vllm_config=object())
    updated = []
    monkeypatch.setattr(controller, "_update_attention", lambda ctx: updated.append(ctx))
    context = _ns(batch_descriptor=_ns(num_tokens=4))
    with controller.replay_scope(context):
        assert controller._active is True
    assert controller._active is False
    sync.assert_called_once()
    assert updated == [context]


@_need(GRT)
def test_replay_scope_propagates_update_failure(monkeypatch):
    platform = _fake_platform()
    platform.torch_device_fn.current_stream = lambda: _ns(synchronize=Mock())
    monkeypatch.setattr(GRT, "current_platform", platform)
    controller = GRT.GraphRuntimeController(vllm_config=object())
    with pytest.raises(RuntimeError):
        with controller.replay_scope(_ns(batch_descriptor=_ns(num_tokens=4))):
            pass
    assert controller._active is False


@_need(GRT)
def test_update_attention_requires_bound_impls():
    controller = GRT.GraphRuntimeController(vllm_config=object())
    with pytest.raises(RuntimeError):
        controller._update_attention(_ns(batch_descriptor=_ns(num_tokens=4)))
    controller.attention_impls = ()
    controller._update_attention(_ns(batch_descriptor=_ns(num_tokens=4)))


@_need(GRT)
def test_update_attention_creates_stream_once(monkeypatch):
    streams = []

    def make_stream():
        streams.append(object())
        return streams[-1]

    platform = _fake_platform()
    platform.torch_device_fn.Stream = make_stream
    monkeypatch.setattr(GRT, "current_platform", platform)
    impls = [_ns(update_graph_params=Mock()), _ns(update_graph_params=Mock())]
    config = object()
    controller = GRT.GraphRuntimeController(vllm_config=config)
    controller.attention_impls = tuple(impls)
    context = _ns(batch_descriptor=_ns(num_tokens=4))
    controller._update_attention(context)
    controller._update_attention(context)
    assert len(streams) == 1
    for impl in impls:
        assert impl.update_graph_params.call_count == 2
        args = impl.update_graph_params.call_args.args
        assert args[0] is streams[0]
        assert args[1] is context
        assert args[3] is config


@_need(GRT)
def test_bind_attention_groups(monkeypatch):
    backend_mod = _module("vllm.v1.attention.backend")

    class AttentionBackend:
        @staticmethod
        def get_impl_cls():
            return None

    backend_mod.AttentionBackend = AttentionBackend
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", backend_mod)

    impl_a = _ns(update_graph_params=lambda *a: None)
    impl_b = _ns(update_graph_params=lambda *a: None)

    class BackendA:
        @classmethod
        def get_impl_cls(cls):
            return impl_a

    class BackendB:
        @classmethod
        def get_impl_cls(cls):
            return impl_b

    class BackendBase:
        get_impl_cls = AttentionBackend.get_impl_cls

    class BackendNone:
        get_impl_cls = None

    controller = GRT.GraphRuntimeController(vllm_config=object())
    groups = [
        [
            _ns(backend=BackendA),
            _ns(backend=BackendB),
            _ns(backend=BackendBase),
            _ns(backend=BackendNone),
        ],
        [_ns(backend=BackendA)],
    ]
    controller.bind_attention_groups(groups)
    assert controller.attention_impls == (impl_a, impl_b)


@_need(GRT)
def test_bind_attention_groups_errors(monkeypatch):
    backend_mod = _module("vllm.v1.attention.backend")

    class AttentionBackend:
        @staticmethod
        def get_impl_cls():
            return None

    backend_mod.AttentionBackend = AttentionBackend
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backend", backend_mod)

    class Raising:
        @classmethod
        def get_impl_cls(cls):
            raise RuntimeError("nope")

    class NoUpdate:
        @classmethod
        def get_impl_cls(cls):
            return _ns()

    controller = GRT.GraphRuntimeController(vllm_config=object())
    with pytest.raises(RuntimeError):
        controller.bind_attention_groups([[_ns(backend=Raising)]])
    with pytest.raises(RuntimeError):
        controller.bind_attention_groups([[_ns(backend=NoUpdate)]])


@_need(GRT)
def test_clear_decoder_graphs(monkeypatch):
    cleared = []
    monkeypatch.setattr(GP, "clear_graph_params", lambda: cleared.append(True))
    wrapper_type = _ns(clear_all_graphs=Mock())
    controller = GRT.GraphRuntimeController(
        vllm_config=object(), full_graph_wrapper_type=wrapper_type
    )
    controller.clear_decoder_graphs()
    wrapper_type.clear_all_graphs.assert_called_once()
    assert cleared == [True]
    controller.wrapper_type = None
    controller.clear_decoder_graphs()
    assert cleared == [True, True]


@pytest.fixture
def fresh_graph_params():
    if GP is not None:
        GP.clear_graph_params()
    yield GP
    if GP is not None:
        GP.clear_graph_params()


@_need(GP)
def test_graph_params_prepare_validation(fresh_graph_params):
    params = GP.GraphParams()
    for bad in (0, -1, True, "4", 2.5):
        with pytest.raises(ValueError):
            params.prepare_capture(BD(bad))


@_need(GP)
def test_graph_params_prepare_and_clear(fresh_graph_params):
    params = GP.GraphParams()
    descriptor = BD(4)
    params.prepare_capture(descriptor)
    assert params.events[descriptor] == []
    assert params.workspaces[descriptor] is None
    assert params.handles[descriptor] == []
    assert params.attn_params[descriptor] == []
    params.prepare_capture(descriptor)
    params.attn_params[descriptor].append("x")
    with pytest.raises(RuntimeError):
        params.prepare_capture(descriptor)
    params.clear()
    assert params.events == {}


@_need(GP)
def test_graph_params_require_capture(fresh_graph_params):
    params = GP.GraphParams()
    descriptor = BD(8)
    with pytest.raises(RuntimeError):
        params.require_capture(descriptor)
    params.prepare_capture(descriptor)
    params.require_capture(descriptor)


@_need(GP)
def test_graph_params_module_registry(fresh_graph_params):
    descriptor = BD(2)
    assert GP.get_graph_params() is GP._GRAPH_PARAMS
    assert GP.prepare_graph_params(descriptor) is GP._GRAPH_PARAMS
    GP.clear_graph_params()
    assert GP.get_graph_params().attn_params == {}


@_need(GP)
def test_weak_ref_workspace(fresh_graph_params):
    descriptor = BD(16)
    with pytest.raises(RuntimeError):
        GP.weak_ref_workspace(descriptor, lambda v: v)

    GP.prepare_graph_params(descriptor)
    GP.weak_ref_workspace(descriptor, lambda v: ("weak", v))
    assert GP.get_graph_params().workspaces[descriptor] is None

    GP.get_graph_params().workspaces[descriptor] = "workspace"
    GP.weak_ref_workspace(descriptor, lambda v: ("weak", v))
    assert GP.get_graph_params().workspaces[descriptor] == ("weak", "workspace")


# ---------------------------------------------------------------------------
# kv_cache/ascend/hybrid.py
# ---------------------------------------------------------------------------
@_need(HYB)
def test_uses_dense_hybrid_cache():
    model = _ns(is_hybrid=True, hf_text_config=_ns(model_type="qwen3_5_text"))
    assert HYB.uses_dense_hybrid_cache(model) is True
    model.hf_text_config.model_type = "qwen3_5_moe_text"
    assert HYB.uses_dense_hybrid_cache(model) is True
    model.hf_text_config.model_type = "llama"
    assert HYB.uses_dense_hybrid_cache(model) is False
    assert HYB.uses_dense_hybrid_cache(
        _ns(is_hybrid=False, hf_text_config=_ns(model_type="qwen3_5_text"))
    ) is False


def _align_env(
    monkeypatch,
    shapes=((2, 3), (8, 16)),
    dtypes=None,
    use_mla=False,
    head_size=1,
    kv_heads=1,
    model_dtype=None,
):
    if dtypes is None:
        dtypes = (torch.float16, torch.float32)
    if model_dtype is None:
        model_dtype = torch.float32
    model_cls = _ns(
        get_mamba_state_shape_from_config=lambda cfg: shapes,
        get_mamba_state_dtype_from_config=lambda cfg: dtypes,
    )
    monkeypatch.setattr(
        HYB,
        "ModelRegistry",
        _ns(resolve_model_cls=lambda arch, model_config=None: (model_cls, None)),
    )
    model = _ns(
        use_mla=use_mla,
        dtype=model_dtype,
        get_head_size=lambda: head_size,
        get_num_kv_heads=lambda parallel: kv_heads,
        max_model_len=2048,
    )
    cache = _ns(
        cache_dtype="auto",
        user_specified_block_size=False,
        block_size=0,
        mamba_cache_mode="align",
        mamba_page_size_padded=None,
        mamba_block_size=None,
    )
    config = _ns(
        cache_config=cache, model_config=model, parallel_config=_ns()
    )
    monkeypatch.setattr(
        HYB,
        "get_dtype_size",
        lambda dtype: torch.tensor([], dtype=dtype).element_size(),
    )
    monkeypatch.setattr(HYB, "cdiv", lambda a, b: -(-a // b))
    monkeypatch.setattr(
        HYB,
        "STR_DTYPE_TO_TORCH_DTYPE",
        {"bfloat16": torch.bfloat16, "float16": torch.float16},
    )
    return config, cache


@_need(HYB)
@needs_torch
def test_align_hybrid_rejects_unsupported_layout(monkeypatch):
    config, _cache = _align_env(monkeypatch, use_mla=True)
    with pytest.raises(NotImplementedError):
        HYB.align_hybrid_block_size(config)
    config, _cache = _align_env(monkeypatch, shapes=((2, 3),))
    with pytest.raises(NotImplementedError):
        HYB.align_hybrid_block_size(config)
    config, _cache = _align_env(
        monkeypatch, shapes=((64, 4), (2, 2)), dtypes=(torch.float32, torch.float32)
    )
    with pytest.raises(NotImplementedError):
        HYB.align_hybrid_block_size(config)


@_need(HYB)
@needs_torch
def test_align_hybrid_success_auto_dtype(monkeypatch):
    config, cache = _align_env(monkeypatch)
    HYB.align_hybrid_block_size(config)
    assert cache.block_size == 128
    assert cache.mamba_page_size_padded == 2 * 512 + 12
    assert cache.mamba_block_size == 128


@_need(HYB)
@needs_torch
def test_align_hybrid_success_non_align(monkeypatch):
    config, cache = _align_env(monkeypatch)
    cache.mamba_cache_mode = "max"
    HYB.align_hybrid_block_size(config)
    assert cache.mamba_block_size == 2048


@_need(HYB)
@needs_torch
def test_align_hybrid_explicit_cache_dtype(monkeypatch):
    config, cache = _align_env(monkeypatch)
    cache.cache_dtype = "bfloat16"
    HYB.align_hybrid_block_size(config)
    assert cache.block_size == 256


@_need(HYB)
@needs_torch
def test_align_hybrid_alignment_and_user_size_errors(monkeypatch):
    config, _cache = _align_env(
        monkeypatch, shapes=((2, 3), (5, 5)), dtypes=(torch.float16, torch.float32)
    )
    with pytest.raises(ValueError):
        HYB.align_hybrid_block_size(config)

    config, cache = _align_env(monkeypatch)
    cache.user_specified_block_size = True
    cache.block_size = 999
    with pytest.raises(ValueError):
        HYB.align_hybrid_block_size(config)


@_need(HYB)
def test_pad_attention_specs(monkeypatch):
    @dataclasses.dataclass
    class _AttnSpec:
        page_size_bytes: int
        page_size_padded: object = None

    monkeypatch.setattr(HYB, "AttentionSpec", _AttnSpec)
    small = _AttnSpec(page_size_bytes=100)
    large = _AttnSpec(page_size_bytes=9000)
    other = _ns(page_size_bytes=1)
    result = HYB.pad_attention_specs({"a": small, "b": large, "c": other}, 8192)
    assert result["a"].page_size_padded == 8192
    assert result["b"] is large
    assert result["c"] is other
    assert small.page_size_padded is None


@_need(HYB)
@needs_torch
def test_dense_attention_view(monkeypatch):
    monkeypatch.setattr(
        HYB,
        "get_dtype_size",
        lambda dtype: torch.tensor([], dtype=dtype).element_size(),
    )
    spec = _ns(dtype=torch.float32)
    raw = torch.zeros(64, dtype=torch.uint8)
    assert tuple(HYB.dense_attention_view(raw, spec, (2, 4, 1, 1, 1)).shape) == (
        2, 4, 1, 1, 1,
    )
    with pytest.raises(ValueError):
        HYB.dense_attention_view(raw.to(torch.float32), spec, (2, 4, 1, 1, 1))
    with pytest.raises(ValueError):
        HYB.dense_attention_view(raw, spec, (4, 8, 1, 1, 1))
    with pytest.raises(ValueError):
        HYB.dense_attention_view(raw, spec, (2, 3, 5))


@_need(HYB)
@needs_torch
def test_dense_mamba_views(monkeypatch):
    monkeypatch.setattr(
        HYB,
        "get_dtype_size",
        lambda dtype: torch.tensor([], dtype=dtype).element_size(),
    )
    raw = torch.zeros(152, dtype=torch.uint8)
    conv, state = HYB.dense_mamba_views(
        raw, ((2, 3), (4, 4)), (torch.float16, torch.float32), 2
    )
    assert tuple(conv.shape) == (2, 2, 3)
    assert tuple(state.shape) == (2, 4, 4)

    with pytest.raises(ValueError):
        HYB.dense_mamba_views(
            raw.to(torch.float32), ((2, 3), (4, 4)),
            (torch.float16, torch.float32), 2,
        )
    with pytest.raises(ValueError):
        HYB.dense_mamba_views(
            torch.zeros(200, dtype=torch.uint8), ((3, 3), (4, 4)),
            (torch.float16, torch.float32), 1,
        )
    with pytest.raises(ValueError):
        HYB.dense_mamba_views(
            torch.zeros(4, dtype=torch.uint8), ((2, 3), (4, 4)),
            (torch.float16, torch.float32), 2,
        )


@_need(HYB)
@needs_torch
def test_deduplicate_block_planes():
    plane = torch.zeros(4, 8, dtype=torch.uint8)
    alias = plane.view(torch.uint8)
    assert len(HYB._deduplicate_block_planes([plane, alias])) == 1
    other = torch.zeros(4, 8, dtype=torch.uint8)
    assert len(HYB._deduplicate_block_planes([plane, other])) == 2


@_need(HYB)
def test_layer_specs(monkeypatch):
    uniform_cls = type("UniformTypeKVCacheSpecs", (), {})

    class Uniform(uniform_cls):
        def __init__(self, mapping):
            self.kv_cache_specs = mapping

    monkeypatch.setattr(HYB, "UniformTypeKVCacheSpecs", uniform_cls)
    spec_a, spec_b = object(), object()
    config = _ns(
        kv_cache_groups=[
            _ns(layer_names=["a", "b"],
                kv_cache_spec=Uniform({"a": spec_a, "b": spec_b})),
            _ns(layer_names=["c"], kv_cache_spec=object()),
        ]
    )
    specs, group_ids = HYB._layer_specs(config)
    assert specs["a"] is spec_a and specs["b"] is spec_b
    assert group_ids["a"] == 0 and group_ids["c"] == 1


@_need(HYB)
@needs_torch
def test_attention_kernel_view(monkeypatch):
    class AttentionSpec:
        pass

    monkeypatch.setattr(HYB, "AttentionSpec", AttentionSpec)
    spec = AttentionSpec()
    spec.block_size = 4
    spec.num_kv_heads = 2
    spec.head_size = 3
    spec.tokens_per_state = 1
    spec.head_size_v = 3
    manager = torch.arange(1 * 2 * 4 * 6).reshape(1, 2, 4, 6).float()
    assert tuple(HYB._attention_kernel_view(manager, spec, 2).shape) == (
        2, 2, 2, 2, 3,
    )
    with pytest.raises(NotImplementedError):
        HYB._attention_kernel_view(manager.reshape(2, 4, 6), spec, 2)
    with pytest.raises(NotImplementedError):
        HYB._attention_kernel_view(manager.transpose(1, 2), spec, 2)
    with pytest.raises(ValueError):
        HYB._attention_kernel_view(manager, spec, 3)


@_need(HYB)
@needs_torch
def test_allocate_empty_and_guards(monkeypatch):
    class AttentionSpec:
        pass

    class MambaSpec:
        pass

    monkeypatch.setattr(HYB, "AttentionSpec", AttentionSpec)
    monkeypatch.setattr(HYB, "MambaSpec", MambaSpec)

    empty = _ns(kv_cache_tensors=[], kv_cache_groups=[])
    assert HYB.allocate_ascend_kv_cache(empty, "cpu", _ns(name="x"), [], [], False) == (
        {}, {}, [],
    )

    host = _ns(kv_cache_tensors=[_ns(host_resident=True)], kv_cache_groups=[])
    with pytest.raises(NotImplementedError):
        HYB.allocate_ascend_kv_cache(host, "cpu", _ns(name="x"), [], [], False)

    attn = AttentionSpec()
    mismatched = _ns(
        kv_cache_tensors=[_ns(host_resident=False)],
        kv_cache_groups=[_ns(layer_names=["a"], kv_cache_spec=attn)],
    )
    with pytest.raises(NotImplementedError):
        HYB.allocate_ascend_kv_cache(mismatched, "cpu", _ns(name="LBNHC"), [], [], False)

    mamba = MambaSpec()
    mixed = _ns(
        kv_cache_tensors=[_ns(host_resident=False)],
        kv_cache_groups=[
            _ns(layer_names=["a"], kv_cache_spec=attn),
            _ns(layer_names=["m"], kv_cache_spec=mamba),
        ],
    )
    with pytest.raises(NotImplementedError):
        HYB.allocate_ascend_kv_cache(mixed, "cpu", _ns(name="LHBNC"), [], [], False)


@_need(HYB)
@needs_torch
def test_allocate_non_dense_attention(monkeypatch):
    class AttentionSpec:
        pass

    class MambaSpec:
        pass

    monkeypatch.setattr(HYB, "AttentionSpec", AttentionSpec)
    monkeypatch.setattr(HYB, "MambaSpec", MambaSpec)

    spec = AttentionSpec()
    spec.block_size = 4
    spec.num_kv_heads = 2
    spec.head_size = 3
    spec.tokens_per_state = 1
    spec.head_size_v = 3
    manager = torch.arange(1 * 2 * 4 * 6).reshape(1, 2, 4, 6).float()
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.utils",
        _module("vllm.v1.worker.utils", allocate_kv_cache=lambda *a, **k: {"a": manager}),
    )
    config = _ns(
        kv_cache_tensors=[_ns(host_resident=False)],
        kv_cache_groups=[_ns(layer_names=["a"], kv_cache_spec=spec)],
    )
    bind_views, manager_views, planes = HYB.allocate_ascend_kv_cache(
        config, "cpu", _ns(name="LHBNC"), [2], [], False
    )
    assert tuple(bind_views["a"].shape) == (2, 2, 2, 2, 3)
    assert manager_views["a"] is manager
    assert len(planes) == 2


@_need(HYB)
@needs_torch
def test_allocate_non_dense_mamba_and_unknown(monkeypatch):
    class AttentionSpec:
        pass

    class MambaSpec:
        pass

    monkeypatch.setattr(HYB, "AttentionSpec", AttentionSpec)
    monkeypatch.setattr(HYB, "MambaSpec", MambaSpec)

    manager = torch.zeros(3, 4, dtype=torch.uint8)
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.utils",
        _module("vllm.v1.worker.utils", allocate_kv_cache=lambda *a, **k: {"m": manager}),
    )
    mamba = MambaSpec()
    config = _ns(
        kv_cache_tensors=[_ns(host_resident=False)],
        kv_cache_groups=[_ns(layer_names=["m"], kv_cache_spec=mamba)],
    )
    bind_views, _manager_views, planes = HYB.allocate_ascend_kv_cache(
        config, "cpu", _ns(name="LHBNC"), [], [], False
    )
    assert bind_views["m"] is manager
    assert len(planes) == 1

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.utils",
        _module(
            "vllm.v1.worker.utils",
            allocate_kv_cache=lambda *a, **k: {"u": torch.zeros(2, 2, dtype=torch.uint8)},
        ),
    )
    bad_config = _ns(
        kv_cache_tensors=[_ns(host_resident=False)],
        kv_cache_groups=[_ns(layer_names=["u"], kv_cache_spec=object())],
    )
    with pytest.raises(NotImplementedError):
        HYB.allocate_ascend_kv_cache(bad_config, "cpu", _ns(name="LHBNC"), [], [], False)


@_need(HYB)
@needs_torch
def test_allocate_dense_attention(monkeypatch):
    class AttentionSpec:
        pass

    class MambaSpec:
        pass

    monkeypatch.setattr(HYB, "AttentionSpec", AttentionSpec)
    monkeypatch.setattr(HYB, "MambaSpec", MambaSpec)
    monkeypatch.setattr(
        HYB,
        "get_dtype_size",
        lambda dtype: torch.tensor([], dtype=dtype).element_size(),
    )

    spec = AttentionSpec()
    spec.block_size = 2
    spec.num_kv_heads = 1
    spec.head_size = 1
    spec.dtype = torch.float32
    spec.page_size_bytes = 32

    info = _ns(
        size=64, layers=["a"], block_stride=32, layer_stride=64,
        offset=0, host_resident=False,
    )
    config = _ns(
        kv_cache_tensors=[info],
        kv_cache_groups=[_ns(layer_names=["a"], kv_cache_spec=spec)],
        num_blocks_of=lambda tensor: 2,
    )
    bound_group = _ns(kv_cache_spec=spec, kernel_block_size=1, kv_cache_group_id=0)
    bind_views, manager_views, planes = HYB.allocate_ascend_kv_cache(
        config, "cpu", _ns(name="LBNHC"), [1], [bound_group], True
    )
    assert tuple(bind_views["a"].shape) == (2, 4, 1, 1, 1)
    assert tuple(manager_views["a"].shape) == (2, 2, 2, 1)
    assert len(planes) == 3
    assert bound_group.kernel_block_stride == 2


@_need(HYB)
@needs_torch
def test_allocate_dense_errors(monkeypatch):
    class AttentionSpec:
        pass

    class MambaSpec:
        pass

    monkeypatch.setattr(HYB, "AttentionSpec", AttentionSpec)
    monkeypatch.setattr(HYB, "MambaSpec", MambaSpec)
    monkeypatch.setattr(
        HYB,
        "get_dtype_size",
        lambda dtype: torch.tensor([], dtype=dtype).element_size(),
    )

    spec = AttentionSpec()
    spec.block_size = 2
    spec.num_kv_heads = 1
    spec.head_size = 1
    spec.dtype = torch.float32
    spec.page_size_bytes = 16

    info = _ns(size=32, layers=["a"], block_stride=16, layer_stride=32,
               offset=0, host_resident=False)
    config = _ns(
        kv_cache_tensors=[info],
        kv_cache_groups=[_ns(layer_names=["a"], kv_cache_spec=spec)],
        num_blocks_of=lambda tensor: 2,
    )
    with pytest.raises(ValueError):
        HYB.allocate_ascend_kv_cache(config, "cpu", _ns(name="LBNHC"), [1], [], True)

    tensor_a = _ns(size=64, layers=["a"], block_stride=32, layer_stride=64,
                   offset=0, host_resident=False)
    tensor_b = _ns(size=32, layers=["b"], block_stride=32, layer_stride=32,
                   offset=0, host_resident=False)
    mismatched = _ns(
        kv_cache_tensors=[tensor_a, tensor_b],
        kv_cache_groups=[_ns(layer_names=["a", "b"], kv_cache_spec=spec)],
        num_blocks_of=lambda tensor: 2,
    )
    with pytest.raises(ValueError):
        HYB.allocate_ascend_kv_cache(
            mismatched, "cpu", _ns(name="LBNHC"), [1], [], True
        )


@_need(HYB)
@needs_torch
def test_copy_block_planes_aligned_and_unaligned():
    aligned = torch.arange(16, dtype=torch.uint8).reshape(4, 4).clone()
    original = aligned.clone()
    HYB.copy_block_planes([aligned], [(0, 1), (1, 2)])
    assert torch.equal(aligned[1], original[0])
    assert torch.equal(aligned[2], original[1])
    assert torch.equal(aligned[0], original[0])
    assert torch.equal(aligned[3], original[3])

    unaligned = torch.arange(9, dtype=torch.uint8).reshape(3, 3).clone()
    before = unaligned.clone()
    HYB.copy_block_planes([unaligned], [(0, 2)])
    assert torch.equal(unaligned[2], before[0])
    assert torch.equal(unaligned[1], before[1])

    HYB.copy_block_planes([], [(0, 1)])
    HYB.copy_block_planes([aligned], [])


@_need(HYB)
@needs_torch
def test_zero_block_planes():
    aligned = torch.ones(4, 4, dtype=torch.uint8)
    HYB.zero_block_planes([aligned], [1, 3])
    assert torch.count_nonzero(aligned[1]) == 0
    assert torch.count_nonzero(aligned[3]) == 0
    assert torch.count_nonzero(aligned[0]) == 4
    assert torch.count_nonzero(aligned[2]) == 4

    unaligned = torch.ones(3, 3, dtype=torch.uint8)
    HYB.zero_block_planes([unaligned], [1])
    assert torch.count_nonzero(unaligned[1]) == 0
    assert torch.count_nonzero(unaligned[0]) == 4

    HYB.zero_block_planes([], [1])
    HYB.zero_block_planes([aligned], [])


