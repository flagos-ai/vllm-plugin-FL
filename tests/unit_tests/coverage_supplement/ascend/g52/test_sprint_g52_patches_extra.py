# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G52: coverage for five vllm_fl.patches compatibility modules.

Targets (814b line):
  * patches/gdn_packed_decode.py
  * patches/fla_module_relocation.py
  * patches/flagtree_target_info.py
  * patches/worker_symbols_032.py
  * patches/qsa_cache_hint.py

All vLLM/Triton collaborators are replaced at the module or sys.modules
boundary; no accelerator, Triton runtime, or FLA package is required beyond
what importing the target modules already needs.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.patches import fla_module_relocation as fla

try:
    from vllm_fl.patches import flagtree_target_info as fti
except Exception:
    if "vllm" not in sys.modules:
        raise
    _logger = types.ModuleType("vllm.logger")
    _logger.init_logger = logging.getLogger
    sys.modules["vllm.logger"] = _logger
    setattr(sys.modules["vllm"], "logger", _logger)
    fti = importlib.import_module("vllm_fl.patches.flagtree_target_info")

from vllm_fl.patches import qsa_cache_hint as qsa
from vllm_fl.patches import worker_symbols_032 as ws


def _stub_gdn_dependencies():
    """Provide the Triton/FLA modules gdn_packed_decode imports at module load."""
    if "vllm" not in sys.modules:
        return False
    triton = sys.modules.get("triton")
    if triton is None or not callable(getattr(triton, "jit", None)):
        triton = types.ModuleType("triton")
        triton.jit = lambda fn: fn
        sys.modules["triton"] = triton
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.tl = SimpleNamespace()
    triton_utils.triton = triton
    sys.modules["vllm.triton_utils"] = triton_utils

    parts = [
        "vllm",
        "vllm.third_party",
        "vllm.third_party.flash_linear_attention",
        "vllm.third_party.flash_linear_attention.ops",
    ]
    parent = sys.modules["vllm"]
    for part in parts[1:]:
        module = sys.modules.get(part) or types.ModuleType(part)
        sys.modules[part] = module
        setattr(parent, part.rsplit(".", 1)[1], module)
        parent = module
    op_module = types.ModuleType(parts[-1] + ".op")
    op_module.exp = lambda value: value
    sys.modules[parts[-1] + ".op"] = op_module
    parent.op = op_module
    return True


try:
    from vllm_fl.patches import gdn_packed_decode as gdn
except Exception:
    if not _stub_gdn_dependencies():
        raise
    gdn = importlib.import_module("vllm_fl.patches.gdn_packed_decode")


@pytest.fixture(autouse=True)
def _clear_caches():
    fti.apply_flagtree_target_info.cache_clear()
    ws.apply_worker_symbols_032.cache_clear()
    yield
    fti.apply_flagtree_target_info.cache_clear()
    ws.apply_worker_symbols_032.cache_clear()


# ---------------------------------------------------------------------------
# gdn_packed_decode
# ---------------------------------------------------------------------------
def _vulnerable_kernel_impl():
    return "tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)"


def _fixed_kernel_impl():
    return "tl.sigmoid(b_val)"


def test_gdn_module_constants_mark_fp32_replacement():
    assert gdn._TARGET_MODULE == (
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
    )
    assert gdn._TARGET_NAME == "fused_recurrent_gated_delta_rule_packed_decode_kernel"
    assert "tl.sigmoid(b_val).to(b.dtype.element_ty)" in gdn._VULNERABLE_BETA_EXPRESSION
    assert gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta._fl_fp32_beta


def test_gdn_kernel_needs_patch_variants():
    assert gdn._kernel_needs_beta_patch(SimpleNamespace(_fl_fp32_beta=True)) is False
    assert gdn._kernel_needs_beta_patch(SimpleNamespace(fn=_vulnerable_kernel_impl)) is True
    assert gdn._kernel_needs_beta_patch(SimpleNamespace(fn=_fixed_kernel_impl)) is False
    # Source inspection unavailable (builtin) -> prefer the corrected kernel.
    assert gdn._kernel_needs_beta_patch(SimpleNamespace(fn=len)) is True


def test_gdn_patch_skips_on_import_error_or_missing_symbol(monkeypatch):
    def _boom(_module):
        raise ImportError("no FLA")

    monkeypatch.setattr(gdn.importlib, "import_module", _boom)
    assert gdn.patch_vllm_packed_gdn_beta() is False

    empty = SimpleNamespace()
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _m: empty)
    assert gdn.patch_vllm_packed_gdn_beta() is False


def test_gdn_patch_replaces_vulnerable_kernel_and_is_idempotent(monkeypatch):
    target = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_vulnerable_kernel_impl
    )
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _m: target)

    assert gdn.patch_vllm_packed_gdn_beta() is True
    replaced = target.fused_recurrent_gated_delta_rule_packed_decode_kernel
    assert replaced is gdn._fused_recurrent_gated_delta_rule_packed_decode_kernel_fp32_beta
    assert replaced._fl_fp32_beta is True
    assert gdn.patch_vllm_packed_gdn_beta() is False


def test_gdn_patch_preserves_upstream_fixed_kernel(monkeypatch):
    target = SimpleNamespace(
        fused_recurrent_gated_delta_rule_packed_decode_kernel=_fixed_kernel_impl
    )
    monkeypatch.setattr(gdn.importlib, "import_module", lambda _m: target)

    assert gdn.patch_vllm_packed_gdn_beta() is False
    assert target.fused_recurrent_gated_delta_rule_packed_decode_kernel is _fixed_kernel_impl


# ---------------------------------------------------------------------------
# flagtree_target_info
# ---------------------------------------------------------------------------
def _target_info(**attrs):
    mod = types.ModuleType("triton.language.target_info")
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _inject_target_info(monkeypatch, module):
    triton = types.ModuleType("triton")
    triton.__path__ = []
    language = types.ModuleType("triton.language")
    language.__path__ = []
    triton.language = language
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", language)
    if module is None:
        monkeypatch.setitem(sys.modules, "triton.language.target_info", None)
    else:
        language.target_info = module
        monkeypatch.setitem(sys.modules, "triton.language.target_info", module)


def test_fti_current_arch_handles_failures_and_non_strings():
    assert fti._current_arch(SimpleNamespace(current_target=None)) == ""

    def _raise():
        raise RuntimeError("no target")

    assert fti._current_arch(SimpleNamespace(current_target=_raise)) == ""
    assert fti._current_arch(
        SimpleNamespace(current_target=lambda: None)
    ) == ""
    assert fti._current_arch(
        SimpleNamespace(current_target=lambda: SimpleNamespace(arch=123))
    ) == ""
    assert fti._current_arch(
        SimpleNamespace(current_target=lambda: SimpleNamespace(arch="gfx942"))
    ) == "gfx942"


def test_fti_predicate_requires_hip_and_matching_arch():
    info = _target_info(
        is_hip=lambda: False,
        current_target=lambda: SimpleNamespace(arch="gfx942"),
    )
    assert fti._make_predicate(info, "is_hip_cdna", ("gfx942",))() is False

    info.is_hip = lambda: True
    assert fti._make_predicate(info, "is_hip_cdna", ("gfx942",))() is True
    assert fti._make_predicate(info, "is_hip_gfx11", ("gfx11",))() is False


def test_fti_predicate_uses_constexpr_decorator_when_available():
    info = _target_info(
        is_hip=lambda: True,
        current_target=lambda: SimpleNamespace(arch="gfx942"),
        constexpr_function=lambda fn: ("wrapped", fn),
    )
    wrapped = fti._make_predicate(info, "is_hip_cdna", ("gfx942",))
    assert wrapped[0] == "wrapped"
    assert wrapped[1].__name__ == "is_hip_cdna"
    assert wrapped[1]() is True

    def _raising(_fn):
        raise ValueError("no constexpr")

    info.constexpr_function = _raising
    plain = fti._make_predicate(info, "is_hip_cdna", ("gfx942",))
    assert callable(plain) and plain.__name__ == "is_hip_cdna" and plain() is True


def test_fti_apply_returns_false_when_triton_missing(monkeypatch):
    _inject_target_info(monkeypatch, None)
    assert fti.apply_flagtree_target_info() is False


def test_fti_apply_adds_missing_predicates_and_keeps_existing(monkeypatch):
    sentinel = lambda: "existing"
    info = _target_info(
        is_hip=lambda: True,
        current_target=lambda: SimpleNamespace(arch="gfx942"),
        is_hip_cdna=sentinel,
    )
    _inject_target_info(monkeypatch, info)

    assert fti.apply_flagtree_target_info() is True
    assert info.is_hip_cdna is sentinel
    for name in fti._ARCH_PREDICATES:
        assert getattr(info, name) is not None
    assert info.is_hip_cdna1() is False
    assert info.is_hip_gfx12() is False

    # Second call is served from lru_cache and changes nothing.
    before = {name: getattr(info, name) for name in fti._ARCH_PREDICATES}
    assert fti.apply_flagtree_target_info() is True
    assert before == {name: getattr(info, name) for name in fti._ARCH_PREDICATES}


def test_fti_apply_noops_when_all_predicates_present(monkeypatch):
    existing = {name: (lambda name=name: name) for name in fti._ARCH_PREDICATES}
    info = _target_info(is_hip=lambda: True, current_target=lambda: None, **existing)
    _inject_target_info(monkeypatch, info)
    assert fti.apply_flagtree_target_info() is True
    assert {name: getattr(info, name) for name in fti._ARCH_PREDICATES} == existing


# ---------------------------------------------------------------------------
# fla_module_relocation
# ---------------------------------------------------------------------------
def _fake_importlib(find_spec, import_module):
    return SimpleNamespace(
        util=SimpleNamespace(find_spec=find_spec),
        import_module=import_module,
    )


def _found(predicate):
    return lambda name: object() if predicate(name) else None


def _fake_sys():
    return SimpleNamespace(modules={})


def test_fla_noop_when_old_or_new_prefix_inapplicable(monkeypatch):
    fake_sys = _fake_sys()
    monkeypatch.setattr(fla, "sys", fake_sys)
    # Old package exists (vLLM <= 0.24) -> leave it alone.
    monkeypatch.setattr(
        fla,
        "importlib",
        _fake_importlib(_found(lambda name: name == fla._OLD_PREFIX), lambda _n: object()),
    )
    assert fla.install() is False
    assert fake_sys.modules == {}

    # New package missing -> nothing to alias onto.
    monkeypatch.setattr(
        fla,
        "importlib",
        _fake_importlib(lambda _name: None, lambda _n: object()),
    )
    assert fla.install() is False
    assert fake_sys.modules == {}


def test_fla_aliases_base_and_ops_and_binds_parent(monkeypatch):
    new_pkg = object()
    ops_pkg = SimpleNamespace(__path__=["/ops"])
    op_mod = object()
    layers = types.SimpleNamespace()
    fake_sys = _fake_sys()
    monkeypatch.setattr(fla, "sys", fake_sys)
    monkeypatch.setattr(
        fla,
        "importlib",
        _fake_importlib(
            _found(lambda name: name in (fla._NEW_PREFIX, fla._NEW_PREFIX + ".ops")),
            lambda name: {
                fla._NEW_PREFIX: new_pkg,
                fla._NEW_PREFIX + ".ops": ops_pkg,
                fla._NEW_PREFIX + ".ops.op": op_mod,
                "vllm.model_executor.layers": layers,
            }[name],
        ),
    )
    monkeypatch.setattr(
        fla, "pkgutil", SimpleNamespace(iter_modules=lambda _path: [SimpleNamespace(name="op")])
    )

    assert fla.install() is True
    assert fake_sys.modules[fla._OLD_PREFIX] is new_pkg
    assert fake_sys.modules[fla._OLD_PREFIX + ".ops"] is ops_pkg
    assert fake_sys.modules[fla._OLD_PREFIX + ".ops.op"] is op_mod
    assert layers.fla is new_pkg


def test_fla_skips_ops_when_subpackage_absent(monkeypatch):
    new_pkg = object()
    layers = types.SimpleNamespace()
    fake_sys = _fake_sys()
    monkeypatch.setattr(fla, "sys", fake_sys)
    monkeypatch.setattr(
        fla,
        "importlib",
        _fake_importlib(
            _found(lambda name: name == fla._NEW_PREFIX),
            lambda name: {
                fla._NEW_PREFIX: new_pkg,
                "vllm.model_executor.layers": layers,
            }[name],
        ),
    )

    def _no_iter(_path):
        raise AssertionError("iter_modules must not run")

    monkeypatch.setattr(fla, "pkgutil", SimpleNamespace(iter_modules=_no_iter))
    assert fla.install() is True
    assert set(fake_sys.modules) == {fla._OLD_PREFIX}
    assert layers.fla is new_pkg


def test_fla_skips_submodule_that_fails_to_import(monkeypatch):
    new_pkg = object()
    ops_pkg = SimpleNamespace(__path__=["/ops"])
    layers = types.SimpleNamespace()
    fake_sys = _fake_sys()
    monkeypatch.setattr(fla, "sys", fake_sys)

    def _import(name):
        if name == fla._NEW_PREFIX + ".ops.bad":
            raise RuntimeError("backend exploded")
        return {
            fla._NEW_PREFIX: new_pkg,
            fla._NEW_PREFIX + ".ops": ops_pkg,
            "vllm.model_executor.layers": layers,
        }[name]

    monkeypatch.setattr(
        fla,
        "importlib",
        _fake_importlib(
            _found(lambda name: name in (fla._NEW_PREFIX, fla._NEW_PREFIX + ".ops")), _import
        ),
    )
    monkeypatch.setattr(
        fla, "pkgutil", SimpleNamespace(iter_modules=lambda _path: [SimpleNamespace(name="bad")])
    )
    assert fla.install() is True
    assert fla._OLD_PREFIX + ".ops.bad" not in fake_sys.modules
    assert fake_sys.modules[fla._OLD_PREFIX + ".ops"] is ops_pkg


def test_fla_preserves_existing_alias_and_parent_attr(monkeypatch):
    new_pkg = object()
    layers = types.SimpleNamespace(fla="pre-existing")
    fake_sys = _fake_sys()
    fake_sys.modules[fla._OLD_PREFIX] = "keep-me"
    monkeypatch.setattr(fla, "sys", fake_sys)
    monkeypatch.setattr(
        fla,
        "importlib",
        _fake_importlib(
            _found(lambda name: name == fla._NEW_PREFIX),
            lambda name: {
                fla._NEW_PREFIX: new_pkg,
                "vllm.model_executor.layers": layers,
            }[name],
        ),
    )
    assert fla.install() is True
    assert fake_sys.modules[fla._OLD_PREFIX] == "keep-me"
    assert layers.fla == "pre-existing"


def test_fla_module_exists_swallows_lookup_errors(monkeypatch):
    for exc in (ImportError("x"), AttributeError("x"), ValueError("x")):
        def _raise(_name, exc=exc):
            raise exc

        monkeypatch.setattr(
            fla, "importlib", SimpleNamespace(util=SimpleNamespace(find_spec=_raise))
        )
        assert fla._module_exists("whatever") is False
    assert fla.apply_fla_module_relocation is fla.install


# ---------------------------------------------------------------------------
# worker_symbols_032
# ---------------------------------------------------------------------------
def _residual_config(sp, tp, splitting_ops=(), inductor=False):
    return SimpleNamespace(
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=sp),
            splitting_ops=splitting_ops,
            use_inductor_graph_partition=inductor,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp),
    )


def test_ws_residual_scattered_disabled_and_tp_one():
    assert ws._is_residual_scattered_for_sp(_residual_config(False, 4), 8) is False
    assert ws._is_residual_scattered_for_sp(_residual_config(True, 1), 8) is False


def test_ws_residual_scattered_true_paths():
    assert ws._is_residual_scattered_for_sp(
        _residual_config(True, 2, inductor=True), 8
    ) is True
    assert ws._is_residual_scattered_for_sp(
        _residual_config(True, 2, splitting_ops=()), 8
    ) is True


def test_ws_residual_scattered_assertions():
    with pytest.raises(AssertionError):
        ws._is_residual_scattered_for_sp(
            _residual_config(True, 2, splitting_ops=("op",)), 8
        )
    with pytest.raises(AssertionError):
        ws._is_residual_scattered_for_sp(_residual_config(True, 2, inductor=True), 7)


def test_ws_supports_xdrope_and_positions():
    assert ws._supports_xdrope(SimpleNamespace(get_xdrope_input_positions=lambda: None))
    assert ws._supports_xdrope(SimpleNamespace(get_xdrope_input_positions=1)) is False
    assert ws._supports_xdrope(object()) is False

    out = np.zeros((2, 6), dtype=np.int64)
    ws._XDRotaryEmbedding.get_next_input_positions_tensor(out, 2, 10, 3)
    assert out[:, 2:5].tolist() == [[10, 11, 12], [10, 11, 12]]
    assert out[:, :2].tolist() == [[0, 0], [0, 0]]
    assert out[:, 5].tolist() == [0, 0]


def test_ws_dtype_size_and_contiguous_view():
    assert ws._dtype_size(torch.float16) == 2
    assert ws._dtype_size(torch.float32) == 4
    spec = SimpleNamespace(dtype=torch.float16, page_size_padded=None)
    raw = torch.zeros(8, dtype=torch.uint8)
    view = ws._reshape_attention_kv_cache(raw, spec, (2, 1, 2), (0, 1, 2), 2, None)
    assert tuple(view.shape) == (2, 1, 2)
    assert view.dtype == torch.float16


def test_ws_reshape_packing_and_stride_order_assert():
    spec = SimpleNamespace(dtype=torch.float16, page_size_padded=None)
    raw = torch.zeros(16, dtype=torch.uint8)
    view = ws._reshape_attention_kv_cache(raw, spec, (2, 1, 2), (0, 1, 2), 2, (2, 8))
    assert tuple(view.shape) == (2, 1, 2)
    with pytest.raises(AssertionError):
        ws._reshape_attention_kv_cache(raw, spec, (2, 1, 2), (1, 0, 2), 2, (0, 8))


def test_ws_reshape_padded_pages_and_kv_first_assert():
    spec = SimpleNamespace(
        dtype=torch.float16, page_size_padded=16, page_size_bytes=8
    )
    raw = torch.zeros(16, dtype=torch.uint8)
    view = ws._reshape_attention_kv_cache(raw, spec, (2, 1, 2), (0, 1, 2), 2, None)
    assert tuple(view.shape) == (2, 1, 2)
    assert view.stride()[0] == 4
    with pytest.raises(AssertionError):
        ws._reshape_attention_kv_cache(raw, spec, (2, 2, 2), (0, 1, 2), 3, None)


def test_ws_reshape_page_aligned_and_assert():
    spec = SimpleNamespace(dtype=torch.float16, page_size_padded=None)
    raw = torch.zeros(16, dtype=torch.uint8)
    view = ws._reshape_attention_kv_cache(
        raw, spec, (2, 2, 2), (0, 1, 2), 2, None, page_aligned_blocks=True
    )
    assert tuple(view.shape) == (2, 2, 2)
    with pytest.raises(AssertionError):
        ws._reshape_attention_kv_cache(
            raw, spec, (2, 2, 2), (1, 0, 2), 2, None, page_aligned_blocks=True
        )
    with pytest.raises(AssertionError):
        ws._reshape_attention_kv_cache(
            raw, spec, (2, 5, 2), (0, 1, 2), 2, None, page_aligned_blocks=True
        )


def test_ws_routed_experts_tensors_roundtrip():
    tensors = ws._RoutedExpertsTensors(
        torch.tensor([[1, 2], [3, 4]]), torch.tensor([5, 6])
    )
    assert tensors.to_cpu_nonblocking() is tensors
    lists = tensors.tolists()
    assert isinstance(lists, ws._RoutedExpertsLists)
    assert lists.routing_data.tolist() == [[1, 2], [3, 4]]
    assert lists.slot_mapping.tolist() == [5, 6]

    class _FakeTensor:
        def __init__(self):
            self.device = SimpleNamespace(type="cuda")
            self.calls = []

        def to(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return ("cpu", args)

    data = _FakeTensor()
    slots = _FakeTensor()
    moved = ws._RoutedExpertsTensors(data, slots).to_cpu_nonblocking()
    assert data.calls == [(("cpu",), {"non_blocking": True})]
    assert slots.calls == [(("cpu",), {"non_blocking": True})]
    assert moved.routing_data == ("cpu", ("cpu",))


def test_ws_set_missing_and_apply_worker_symbols(monkeypatch):
    module = SimpleNamespace(existing=1)
    ws._set_missing(module, "existing", 2)
    ws._set_missing(module, "added", 3)
    assert module.existing == 1 and module.added == 3

    root = types.ModuleType("vllm")
    model_executor = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    models = types.ModuleType("vllm.model_executor.models")
    v1 = types.ModuleType("vllm.v1")
    worker = types.ModuleType("vllm.v1.worker")
    gpu = types.ModuleType("vllm.v1.worker.gpu")
    rotary = types.ModuleType("vllm.model_executor.layers.rotary_embedding")
    interfaces = types.ModuleType("vllm.model_executor.models.interfaces")
    outputs = types.ModuleType("vllm.v1.outputs")
    worker_utils = types.ModuleType("vllm.v1.worker.utils")
    attn_utils = types.ModuleType("vllm.v1.worker.gpu.attn_utils")
    marker = object()
    attn_utils._reshape_attention_kv_cache = marker
    layers.rotary_embedding = rotary
    models.interfaces = interfaces
    worker.utils = worker_utils
    gpu.attn_utils = attn_utils
    worker.gpu = gpu
    v1.worker = worker
    v1.outputs = outputs
    model_executor.layers = layers
    model_executor.models = models
    root.model_executor = model_executor
    modules = {
        "vllm": root,
        "vllm.model_executor": model_executor,
        "vllm.model_executor.layers": layers,
        "vllm.model_executor.models": models,
        "vllm.model_executor.layers.rotary_embedding": rotary,
        "vllm.model_executor.models.interfaces": interfaces,
        "vllm.v1": v1,
        "vllm.v1.outputs": outputs,
        "vllm.v1.worker": worker,
        "vllm.v1.worker.utils": worker_utils,
        "vllm.v1.worker.gpu": gpu,
        "vllm.v1.worker.gpu.attn_utils": attn_utils,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    assert ws.apply_worker_symbols_032() is True
    assert worker_utils.is_residual_scattered_for_sp is ws._is_residual_scattered_for_sp
    assert rotary.XDRotaryEmbedding is ws._XDRotaryEmbedding
    assert interfaces.SupportsXDRoPE is ws._SupportsXDRoPE
    assert interfaces.supports_xdrope is ws._supports_xdrope
    assert attn_utils._reshape_attention_kv_cache is marker
    assert outputs.RoutedExpertsLists is ws._RoutedExpertsLists
    assert outputs.RoutedExpertsTensors is ws._RoutedExpertsTensors
    assert ws.apply_worker_symbols_032() is True


# ---------------------------------------------------------------------------
# qsa_cache_hint
# ---------------------------------------------------------------------------
_SRC_WITH_HINT = """
@triton.jit
def kernel(k_cache_ptr, offs, out):
    keys = tl.load(k_cache_ptr + offs, eviction_policy="evict_first", other=0)
    tl.store(out + offs, keys)
"""

_SRC_NO_HINT = """
@triton.jit
def kernel(k_cache_ptr, offs, out):
    keys = tl.load(k_cache_ptr + offs, other=0)
    tl.store(out + offs, keys)
"""


def _recording_kernel(src, compiled=False):
    rec = {
        "src": src,
        "device_caches": {0: object()} if compiled else {},
        "updates": [],
    }

    def _update(new):
        rec["updates"].append(new)
        rec["src"] = new

    rec["_unsafe_update_src"] = _update
    return SimpleNamespace(**rec)


def test_qsa_removes_only_optional_eviction_hint():
    result = qsa._without_k_cache_eviction_hint(SimpleNamespace(src=_SRC_WITH_HINT))
    assert result is not None
    assert "eviction_policy" not in result
    assert "k_cache_ptr" in result and "keys =" in result
    assert qsa._without_k_cache_eviction_hint(
        SimpleNamespace(src=_SRC_NO_HINT)
    ) is None


def test_qsa_rejects_policy_change():
    src = _SRC_WITH_HINT.replace("evict_first", "evict_last")
    with pytest.raises(RuntimeError, match="eviction policy changed"):
        qsa._without_k_cache_eviction_hint(SimpleNamespace(src=src))


def test_qsa_rejects_structural_drift():
    _two_loads = """
def kernel(k_cache_ptr, offs):
    keys = tl.load(k_cache_ptr + offs, eviction_policy="evict_first")
    keys = tl.load(k_cache_ptr + offs, eviction_policy="evict_first")
"""
    drifted = [
        _SRC_WITH_HINT.replace("k_cache_ptr", "other_ptr"),
        "x = 1\n",
        _two_loads,
    ]
    for src in drifted:
        with pytest.raises(RuntimeError, match="K-cache load changed"):
            qsa._without_k_cache_eviction_hint(SimpleNamespace(src=src))


def _install_indexer(monkeypatch, kernels):
    names = [
        "vllm.models",
        "vllm.models.qwen4_exp",
        "vllm.models.qwen4_exp.nvidia",
        "vllm.models.qwen4_exp.nvidia.ops",
    ]
    mods = {name: types.ModuleType(name) for name in names}
    indexer = types.ModuleType("vllm.models.qwen4_exp.nvidia.ops.qsa_indexer")
    for name, kernel in kernels.items():
        setattr(indexer, name, kernel)
    for i in range(1, len(names)):
        setattr(mods[names[i - 1]], names[i].rsplit(".", 1)[1], mods[names[i]])
    setattr(mods[names[-1]], "qsa_indexer", indexer)
    root = sys.modules.get("vllm")
    if root is not None:
        monkeypatch.setattr(root, "models", mods["vllm.models"], raising=False)
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(
        sys.modules, "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer", indexer
    )
    return indexer


def _fake_triton(monkeypatch, version="3.6.4"):
    triton = types.ModuleType("triton")
    triton.__version__ = version
    monkeypatch.setitem(sys.modules, "triton", triton)


def _config(architecture="Qwen4ExpForConditionalGeneration"):
    return SimpleNamespace(architecture=architecture)


def test_qsa_patch_early_returns(monkeypatch):
    device = SimpleNamespace(type="cuda")
    assert qsa.patch_qsa_cache_hint(_config("LlamaForCausalLM"), device) is False

    _fake_triton(monkeypatch, "3.7.0")
    assert qsa.patch_qsa_cache_hint(_config(), device) is False

    _fake_triton(monkeypatch, "3.6.4")

    def _missing(_name):
        raise qsa.PackageNotFoundError("flagtree")

    monkeypatch.setattr(qsa, "version", _missing)
    assert qsa.patch_qsa_cache_hint(_config(), device) is False

    monkeypatch.setattr(qsa, "version", lambda _name: "0.7.0")
    seen = []
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda dev: seen.append(dev) or (9, 0),
        raising=False,
    )

    assert qsa.patch_qsa_cache_hint(_config(), SimpleNamespace(type="cpu")) is False
    assert seen == []

    monkeypatch.setattr(torch.version, "hip", "test-rocm", raising=False)
    assert qsa.patch_qsa_cache_hint(_config(), device) is False

    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _dev: (8, 0), raising=False
    )
    assert qsa.patch_qsa_cache_hint(_config(), device) is False




def _patch_env(monkeypatch):
    _fake_triton(monkeypatch)
    monkeypatch.setattr(qsa, "version", lambda _name: "0.7.0")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda _dev: (9, 0), raising=False
    )
    return SimpleNamespace(type="cuda")


def test_qsa_patch_skips_already_patched_kernel(monkeypatch):
    device = _patch_env(monkeypatch)
    patched = _recording_kernel(_SRC_NO_HINT)
    pending = _recording_kernel(_SRC_WITH_HINT)
    _install_indexer(
        monkeypatch, {"_qsa_mqa_paged_prefill_kernel": patched, "_qsa_mqa_paged_uniform_kernel": pending}
    )
    assert qsa.patch_qsa_cache_hint(_config(), device) is True
    assert not hasattr(patched, "_fl_qsa_original_src")
    assert pending._fl_qsa_original_src == _SRC_WITH_HINT

    all_patched = {
        name: _recording_kernel(_SRC_NO_HINT) for name in qsa._KERNEL_NAMES
    }
    _install_indexer(monkeypatch, all_patched)
    assert qsa.patch_qsa_cache_hint(_config(), device) is False


def test_qsa_patch_refuses_compiled_kernel_atomically(monkeypatch):
    device = _patch_env(monkeypatch)
    first = _recording_kernel(_SRC_WITH_HINT)
    second = _recording_kernel(_SRC_WITH_HINT, compiled=True)
    _install_indexer(
        monkeypatch,
        {
            "_qsa_mqa_paged_prefill_kernel": first,
            "_qsa_mqa_paged_uniform_kernel": second,
        },
    )
    with pytest.raises(RuntimeError, match="already compiled"):
        qsa.patch_qsa_cache_hint(_config(), device)
    assert first.src == _SRC_WITH_HINT
    assert not hasattr(first, "_fl_qsa_original_src")
    assert second.src == _SRC_WITH_HINT

